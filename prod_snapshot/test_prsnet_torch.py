#!/usr/bin/env python3
"""
test_prsnet_torch.py -- checks for the pure-PyTorch PRS-Net port. No DGL, no GPU.

  python3 test_prsnet_torch.py [path/to/ggi_graph.bin]

1. graph decode: the authors' file gives 19,836 nodes / 250,236 stored edges,
   a symmetric edge multiset, one self-loop per node, and the edge-count
   matrix row/column sums equal the in/out degrees with multiplicity.
2. GIN layer: sparse-mm aggregation == brute-force per-stored-edge loop
   (DGL 'sum' semantics incl. duplicates and self-loops) on a random small
   multigraph AND on the real graph for a 3-sample batch.
3. readout: sum_nodes == explicit per-graph loop.
4. full forward: PRSNet(A, x) == a reference forward written as an explicit
   loop over the batch with dense numpy (the DGL computation spelled out);
   also checks batch independence (sample i's logit does not depend on the
   other samples in the batch, in eval mode).
5. authors' quirk: MLP batch norms are inert (output unchanged when their
   affine parameters are randomized); the outer GIN batch norm is NOT inert.
6. state_dict keys == the key set the DGL model produces (hard-coded list,
   taken from src/model.py's module tree).
7. LD extension (Oct 19 2026): use_ld=True with ld_weight = 0 gives the same
   logits as the plain model with the same weights; with ld_weight != 0 the
   aggregation equals A_ppi h + w * A_ld h by brute force; the only new
   state_dict key is gnn_layer_list.0.ld_weight.
"""
import sys, os
import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prsnet_torch import PRSNet, MLP, GINConv, load_ggi_graph, read_dgl_graph_edges, edge_count_matrix, sum_nodes

torch.manual_seed(0); np.random.seed(0)
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def brute_gin(src, dst, n, h, apply_func, eps=0.0):
    """DGL GINConv 'sum' on B identical graphs, one stored edge at a time."""
    B = h.shape[0] // n
    out = (1 + eps) * h.clone()
    for b in range(B):
        for s, d in zip(src, dst):
            out[b * n + d] += h[b * n + s]
    return apply_func(out)


def to_sparse(src, dst, n):
    C = edge_count_matrix(src, dst, n)
    return torch.sparse_csr_tensor(torch.from_numpy(C.indptr.astype(np.int64)), torch.from_numpy(C.indices.astype(np.int64)),
                                   torch.from_numpy(C.data), size=(n, n))


# ---------------------------------------------------------------- 1. graph
path = sys.argv[1] if len(sys.argv) > 1 else "data/ggi_graph.bin"
print("1. graph decode")
if path.endswith(".npz"):
    z = np.load(path); src, dst, n = z["src"].astype(np.int64), z["dst"].astype(np.int64), int(z["n_nodes"])
else:
    src, dst, n = read_dgl_graph_edges(path)
authors = (n == 19836 and len(src) == 250236)
print(f"  graph: {n:,} nodes, {len(src):,} stored edges -- "
      + ("the authors' ggi_graph.bin" if authors else "NOT the authors' file (a rebuilt graph; counts not asserted)"))
if path.endswith(".bin"):
    check("nodes == 19836", n == 19836, n)
    check("stored edges == 250236", len(src) == 250236, len(src))
pairs = {}
for s, d in zip(src.tolist(), dst.tolist()):
    pairs[(s, d)] = pairs.get((s, d), 0) + 1
check("edge multiset symmetric", all(pairs.get((d, s), 0) == m for (s, d), m in pairs.items()))
check("every node has a self-loop", len({s for s, d in pairs if s == d}) == n)
C = edge_count_matrix(src, dst, n)
check("count matrix sums to stored edges", C.sum() == len(src))
check("row sums == in-degree with multiplicity", np.array_equal(np.asarray(C.sum(1)).ravel(), np.bincount(dst, minlength=n)))
check("col sums == out-degree with multiplicity", np.array_equal(np.asarray(C.sum(0)).ravel(), np.bincount(src, minlength=n)))
check("duplicates present (multigraph)", max(pairs.values()) > 1, f"max multiplicity {max(pairs.values())}")

# ---------------------------------------------------------------- 2. GIN layer
print("2. GIN aggregation == brute force")
D = 5
# (a) small random multigraph with duplicates and self-loops
ns = 7
rs = np.random.randint(0, ns, 40); rd = np.random.randint(0, ns, 40)
rs = np.concatenate([rs, np.arange(ns), np.arange(ns)[:2]]); rd = np.concatenate([rd, np.arange(ns), np.arange(ns)[:2]])
mlp = MLP(D, D, D, n_layers=1, bias=False, use_batchnorm=True, out_batchnorm=False)
gin = GINConv(mlp)
h = torch.randn(3 * ns, D)
a = gin(to_sparse(rs, rd, ns), h, ns); b = brute_gin(rs, rd, ns, h, mlp)
check("small multigraph, B=3", torch.allclose(a, b, atol=1e-6), f"max |diff| {(a - b).abs().max():.2e}")
# (b) the real graph (brute force over 250k edges x 3 samples, a few seconds)
mlp = MLP(D, D, D, n_layers=1, bias=False, use_batchnorm=True, out_batchnorm=False)
gin = GINConv(mlp)
A, _ = load_ggi_graph(path, expect_nodes=None, expect_edges=None, dtype=torch.float32)
h = torch.randn(3 * n, D)
a = gin(A, h, n); b = brute_gin(src, dst, n, h, mlp)
check("real graph, B=3 (float32 summation order)", torch.allclose(a, b, atol=1e-4), f"max |diff| {(a - b).abs().max():.2e}")

# ---------------------------------------------------------------- 3. readout
print("3. sum_nodes == per-graph loop")
B = 4
v = torch.randn(B * n, D); w = torch.rand(B * n, 1)
ref = torch.stack([(v[i * n:(i + 1) * n] * w[i * n:(i + 1) * n]).sum(0) for i in range(B)])
check("weighted node sum", torch.allclose(sum_nodes(v, w, n), ref, atol=1e-4), f"max |diff| {(sum_nodes(v, w, n) - ref).abs().max():.2e}")

# ---------------------------------------------------------------- 4. full forward
print("4. full forward == dense reference, batch independence")
model = PRSNet(n_genes=n).eval()
x = torch.randn(B, n * 11)
with torch.no_grad():
    preds, gates = model(A, x)

def reference_forward(model, Cdense_mm, x):
    """The authors' forward, written out per sample with explicit matrices."""
    out = []
    for b in range(x.shape[0]):
        xb = x[b].reshape(n, 11)
        hb = model.gene_encoder.linear_list[0](xb)           # n_gene_encode_layer=1, BN inert
        hb = torch.nn.functional.gelu(hb)                    # out_batchnorm=True -> dropout(gelu), p=0
        hb = hb + model.gene_embeddings.weight
        agg = Cdense_mm(hb)                                  # sum over stored in-edges
        hb = model.gnn_layer_list[0].apply_func.linear_list[0](hb + agg)   # eps=0
        bn = model.batch_norm_list[0]
        hb = (hb - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps) * bn.weight + bn.bias
        hb = torch.nn.functional.gelu(hb)
        keys = model.readout.key_layer(hb)
        wb = torch.sigmoid(model.readout.query_layer[0](keys))
        vb = model.readout.value_layer(hb)
        g = (wb * vb).sum(0)
        p = model.predictor.linear_list[0](g); p = torch.nn.functional.gelu(p)
        p = model.predictor.linear_list[1](p)
        out.append(p)
    return torch.stack(out)

with torch.no_grad():
    Ct = torch.from_numpy(C.toarray()).float()
    ref = reference_forward(model, lambda hb: Ct @ hb, x)
check("logits == reference", torch.allclose(preds, ref, atol=1e-4), f"max |diff| {(preds - ref).abs().max():.2e}")
with torch.no_grad():
    p1, _ = model(A, x[:1]); p2, _ = model(A, x[1:3])
check("batch independence (eval)", torch.allclose(preds[0], p1[0], atol=1e-5) and torch.allclose(preds[1:3], p2, atol=1e-5))
check("gate shape (B*n, 1) in (0,1)", gates.shape == (B * n, 1) and gates.min() > 0 and gates.max() < 1)

# ---------------------------------------------------------------- 5. quirks
print("5. authors' quirks preserved")
with torch.no_grad():
    for m in [model.gene_encoder, model.gnn_layer_list[0].apply_func, model.predictor]:
        for bn in m.batch_norm_list:
            bn.weight.mul_(5.0); bn.bias.add_(3.0)
    p_after, _ = model(A, x)
check("MLP-internal batch norms are inert", torch.allclose(p_after, preds, atol=1e-5))
with torch.no_grad():
    model.batch_norm_list[0].weight.mul_(5.0)
    p_outer, _ = model(A, x)
check("outer GIN batch norm is active", not torch.allclose(p_outer, preds, atol=1e-3))

# ---------------------------------------------------------------- 6. state dict
print("6. state_dict keys match the DGL model")
expected = {
    "gene_encoder.linear_list.0.weight", "gene_encoder.linear_list.0.bias",
    "gene_encoder.batch_norm_list.0.weight", "gene_encoder.batch_norm_list.0.bias",
    "gene_encoder.batch_norm_list.0.running_mean", "gene_encoder.batch_norm_list.0.running_var",
    "gene_encoder.batch_norm_list.0.num_batches_tracked",
    "gene_embeddings.weight",
    "gnn_layer_list.0.eps", "gnn_layer_list.0.apply_func.linear_list.0.weight",
    "gnn_layer_list.0.apply_func.batch_norm_list.0.weight", "gnn_layer_list.0.apply_func.batch_norm_list.0.bias",
    "gnn_layer_list.0.apply_func.batch_norm_list.0.running_mean", "gnn_layer_list.0.apply_func.batch_norm_list.0.running_var",
    "gnn_layer_list.0.apply_func.batch_norm_list.0.num_batches_tracked",
    "batch_norm_list.0.weight", "batch_norm_list.0.bias", "batch_norm_list.0.running_mean",
    "batch_norm_list.0.running_var", "batch_norm_list.0.num_batches_tracked",
    "readout.key_layer.weight", "readout.key_layer.bias", "readout.query_layer.0.weight",
    "readout.value_layer.weight", "readout.value_layer.bias", "readout.proj.weight", "readout.proj.bias",
    "predictor.linear_list.0.weight", "predictor.linear_list.0.bias",
    "predictor.linear_list.1.weight", "predictor.linear_list.1.bias",
    "predictor.batch_norm_list.0.weight", "predictor.batch_norm_list.0.bias",
    "predictor.batch_norm_list.0.running_mean", "predictor.batch_norm_list.0.running_var",
    "predictor.batch_norm_list.0.num_batches_tracked",
    "predictor.batch_norm_list.1.weight", "predictor.batch_norm_list.1.bias",
    "predictor.batch_norm_list.1.running_mean", "predictor.batch_norm_list.1.running_var",
    "predictor.batch_norm_list.1.num_batches_tracked",
}
got = set(model.state_dict().keys())
check("key set identical", got == expected, f"missing {sorted(expected - got)}, extra {sorted(got - expected)}")
n_par = sum(p.numel() for p in model.parameters())
check("parameter count (1,295,873 at 19,836 genes; 64 per gene)", n_par == 1295873 - 64 * (19836 - n), n_par)

# ---------------------------------------------------------------- 7. LD extension
print("7. LD edge type")
import prsnet_ld
model0 = PRSNet(n_genes=n).eval()
model_ld = PRSNet(n_genes=n, use_ld=True, ld_init=0.0).eval()
model_ld.load_state_dict(model0.state_dict(), strict=False)
got_ld = set(model_ld.state_dict().keys())
check("only new key is gnn_layer_list.0.ld_weight", got_ld - expected == {"gnn_layer_list.0.ld_weight"} and expected <= got_ld)
# a random symmetric LD graph
m_ld = 50
la = np.random.randint(0, n, m_ld); lb = np.random.randint(0, n, m_ld); keep = la != lb
la, lb = la[keep], lb[keep]; wr = np.random.uniform(-1, 1, len(la)).astype(np.float32)
A_ld = prsnet_ld._csr(np.concatenate([la, lb]), np.concatenate([lb, la]), np.concatenate([wr, wr]), n)
with torch.no_grad():
    p0, _ = model0(A, x); pl, _ = model_ld((A, A_ld), x)
check("ld_weight = 0 reproduces the plain model", torch.allclose(p0, pl, atol=1e-5), f"max |diff| {(p0 - pl).abs().max():.2e}")
with torch.no_grad():
    model_ld.gnn_layer_list[0].ld_weight.fill_(-0.7)
    h = torch.randn(2 * n, D)
    mlp = MLP(D, D, D, n_layers=1, bias=False, use_batchnorm=True, out_batchnorm=False)
    g = GINConv(mlp, use_ld=True, ld_init=-0.7)
    out = g((A, A_ld), h, n)
    Ld = torch.from_numpy(np.asarray(__import__("scipy.sparse").sparse.coo_matrix(
        (np.concatenate([wr, wr]), (np.concatenate([lb, la]), np.concatenate([la, lb]))), shape=(n, n)).toarray())).float()
    ref = torch.cat([mlp(h[i * n:(i + 1) * n] + Ct @ h[i * n:(i + 1) * n] - 0.7 * (Ld @ h[i * n:(i + 1) * n])) for i in range(2)])
check("aggregation == A_ppi h + w_ld * A_ld h (dense reference)", torch.allclose(out, ref, atol=1e-4), f"max |diff| {(out - ref).abs().max():.2e}")
with torch.no_grad():
    p_id, _ = model0(prsnet_ld.identity_graph(n), x)
    ref_id = reference_forward(model0, lambda hb: hb, x)        # identity: agg = h itself
check("identity graph == self-loop-only reference", torch.allclose(p_id, ref_id, atol=1e-4), f"max |diff| {(p_id - ref_id).abs().max():.2e}")
A_sh, done, E = prsnet_ld.shuffled_graph(src, dst, n, seed=0)
check("shuffled graph keeps the stored edge count", int(A_sh.values().sum()) == 4 * E + n and done > 0, f"{done} swaps over {E} pairs")
_, _, hid = model0(A, x, return_hidden=True)
check("return_hidden gives [enc, gnn] of shape (B*n, 64)", len(hid) == 2 and tuple(hid[0].shape) == (B * n, 64))

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
