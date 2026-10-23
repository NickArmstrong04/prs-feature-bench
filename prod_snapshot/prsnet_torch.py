#!/usr/bin/env python3
"""
prsnet_torch.py -- PRS-Net (lihan97/PRS-Net, src/model.py) in pure PyTorch.

Why this exists: the authors' model needs DGL, which has no build for Blackwell
GPUs (prsnet.md section 8). The graph is identical for every sample, so the
DGL machinery (dgl.batch of 512 copies of the same 19,836-node graph, per-node
ndata scatter/sum) reduces to one sparse matrix product per GIN layer and one
weighted sum per readout. Everything else is the authors' code, kept line for
line where it matters -- including two quirks that affect the numbers:

  1. MLP.forward calls self.batch_norm_list[i](h) and DISCARDS the result, so
     the batch norms inside MLP never touch the activations (they only update
     their running statistics). The port keeps the modules (same state_dict
     keys) and keeps them inert.
  2. ggi_graph.bin is a multigraph: every PPI edge is stored twice per
     direction (43 pairs 4-5x) and one self-loop 5x. GINConv with
     aggregator 'sum' counts each stored edge, so the adjacency used here is
     the edge COUNT matrix, not a 0/1 matrix. load_ggi_graph() returns it.

Parameter names match model.py exactly, so a state_dict trained with the
authors' DGL code loads into PRSNet here and vice versa (the GIN layer's MLP
is stored under 'apply_func' like dgl.nn.GINConv).

  from prsnet_torch import PRSNet, load_ggi_graph
  A, n = load_ggi_graph("tools/PRS-Net/data/ggi_graph.bin")   # torch sparse CSR, (n, n)
  model = PRSNet(n_genes=n)
  preds, w = model(A, x)    # x: (B, n*11) float32 -> preds (B, 1) logits, w (B*n, 1) gates

DGL semantics reproduced (dgl.nn.pytorch.conv.GINConv, learn_eps=False,
aggregator_type='sum'; dgl.readout.sum_nodes(g, 'v', 'w')):
    GIN:      h_v' = apply_func( (1 + eps) * h_v + sum_{(u->v) in E, with multiplicity} h_u ),  eps = 0
    readout:  g_h  = sum_v w_v * v_v   per graph
test_prsnet_torch.py checks both against a brute-force per-edge loop, and
verify_against_dgl.py checks the whole forward pass against the DGL model when
DGL is importable (CPU is fine).

LD extension (Oct 19 2026, ld_experiments.md): PRSNet(use_ld=True) adds a
second edge type. A is then a pair (A_ppi, A_ld) and the GIN update becomes
    h_v' = apply_func( (1 + eps) * h_v + sum_u A_ppi[v,u] h_u + w_ld * sum_u A_ld[v,u] h_u )
with ONE learnable scalar w_ld per GIN layer (parameter gnn_layer_list.<i>.ld_weight,
initialised at --ld-init, default 0 so the model starts identical to PRS-Net).
With use_ld=False nothing changes and the state_dict is the authors' exactly.
forward(..., return_hidden=True) additionally returns the list of node
representations [after encoder, after each GIN layer] for the leakage metric.
"""
import struct
import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

THRESHOLDS = [1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]   # d_input = 11


# =============================================================================
#  graph
# =============================================================================
def read_dgl_graph_edges(path):
    """
    Decode the edge lists of a single homogeneous graph saved by dgl.save_graphs
    (DGL >= 0.5 'graph.bin' layout) WITHOUT importing dgl.

    Layout (as read from the authors' file; little-endian):
      several small NDArray blobs of metadata, the graph's number of nodes
      twice (int64), then the two edge blobs, src and dst, each: 8-byte magic
      0x3fa1b496f0405edd, 8 bytes reserved, int64 ndim (=1), 8 bytes dtype
      (int64), int64 length, int64 byte size, then length*8 bytes of ids.
    Returns (src, dst, n_nodes) as int64 numpy arrays. Every assumption is
    asserted, so an unexpected file fails loudly rather than silently.
    """
    b = open(path, "rb").read()
    magic = bytes.fromhex("3fa1b496f0405edd")
    blobs, i = [], b.find(magic)
    while i != -1:
        ndim = struct.unpack_from("<q", b, i + 16)[0]
        n = struct.unpack_from("<q", b, i + 32)[0]
        nbytes = struct.unpack_from("<q", b, i + 40)[0]
        blobs.append((i, ndim, n, nbytes))
        i = b.find(magic, i + 8)
    # the edge lists are the two largest blobs; they are int64, equal length, and consecutive
    big = sorted(blobs, key=lambda t: -t[2])[:2]
    big.sort()
    (p0, nd0, n0, nb0), (p1, nd1, n1, nb1) = big
    assert nd0 == 1 and nd1 == 1 and n0 == n1 and nb0 == 8 * n0 and nb1 == 8 * n1, \
        f"{path}: unexpected NDArray layout {big} (all blobs: {blobs})"
    assert p1 == p0 + 48 + nb0, f"{path}: src/dst blobs are not consecutive"
    src = np.frombuffer(b[p0 + 48:p0 + 48 + nb0], dtype="<i8").astype(np.int64)
    dst = np.frombuffer(b[p1 + 48:p1 + 48 + nb1], dtype="<i8").astype(np.int64)
    # number of nodes: the int64 written twice just before the src blob
    n_nodes = struct.unpack_from("<q", b, p0 - 16)[0]
    assert struct.unpack_from("<q", b, p0 - 8)[0] == n_nodes, "node-count fields disagree"
    assert src.min() >= 0 and dst.min() >= 0 and max(src.max(), dst.max()) < n_nodes, "edge id out of range"
    return src, dst, int(n_nodes)


def edge_count_matrix(src, dst, n_nodes):
    """Sparse (n, n) int-valued matrix C with C[v, u] = number of stored edges u -> v."""
    import scipy.sparse as sp
    C = sp.coo_matrix((np.ones(len(src), dtype=np.float32), (dst, src)), shape=(n_nodes, n_nodes))
    return C.tocsr()   # duplicates are summed -> multiplicities


def load_ggi_graph(path, expect_nodes=19836, expect_edges=250236, device=None, dtype=torch.float32):
    """
    The authors' GGI graph as a torch sparse CSR tensor A with
    A[v, u] = multiplicity of edge u -> v (self-loops included, as stored).
    Returns (A, n_nodes). Prints the same summary prsnet.md section 4 records.
    """
    if path.endswith(".npz"):                    # test graphs: np.savez(src=, dst=, n_nodes=)
        z = np.load(path); src, dst, n = z["src"].astype(np.int64), z["dst"].astype(np.int64), int(z["n_nodes"])
    else:
        src, dst, n = read_dgl_graph_edges(path)
    if expect_nodes is not None:
        assert n == expect_nodes, f"graph has {n} nodes, expected {expect_nodes}"
    if expect_edges is not None:
        assert len(src) == expect_edges, f"graph has {len(src)} edges, expected {expect_edges}"
    C = edge_count_matrix(src, dst, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # torch's "sparse CSR is beta" notices
        A = torch.sparse_csr_tensor(
            torch.from_numpy(C.indptr.astype(np.int64)), torch.from_numpy(C.indices.astype(np.int64)),
            torch.from_numpy(C.data).to(dtype), size=(n, n))
    if device is not None:
        A = A.to(device)
    return A, n


def graph_summary(path):
    src, dst, n = read_dgl_graph_edges(path)
    self_mask = src == dst
    pairs = {}
    for s, d in zip(src.tolist(), dst.tolist()):
        pairs[(s, d)] = pairs.get((s, d), 0) + 1
    nonself = [k for k in pairs if k[0] != k[1]]
    und = {(min(k), max(k)) for k in nonself}
    deg = np.bincount(src, minlength=n)
    return {
        "nodes": n, "stored_edges": int(len(src)),
        "stored_self_loops": int(self_mask.sum()), "nodes_with_self_loop": int(len(set(src[self_mask].tolist()))),
        "unique_directed_ppi": len(nonself), "undirected_ppi": len(und),
        "symmetric": all((d, s) in pairs for s, d in nonself),
        "multiplicity_hist": {m: sum(1 for v in pairs.values() if v == m) for m in sorted(set(pairs.values()))},
        "nodes_with_only_self_loop": int((deg == 1).sum()), "max_out_degree": int(deg.max()),
    }


# =============================================================================
#  model  (names and defaults as in src/model.py)
# =============================================================================
def bert_init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.BatchNorm1d):
        torch.nn.init.normal_(module.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(module.bias.data, 0)


class MLP(nn.Module):
    """Verbatim port. NOTE the authors' forward never assigns the batch-norm
    output back to h (`self.batch_norm_list[i](h)` on its own line), so the
    norms are inert; we call them the same way so running stats evolve
    identically and state_dicts stay interchangeable."""
    def __init__(self, d_input, d_hidden, d_output, n_layers, activation=nn.GELU(), bias=False,
                 dropout=0, use_batchnorm=False, out_batchnorm=False):
        super().__init__()
        self.n_layers = n_layers
        self.use_batchnorm = use_batchnorm
        self.out_batchnorm = out_batchnorm
        self.linear_list = nn.ModuleList()
        if n_layers == 1:
            self.linear_list.append(nn.Linear(d_input, d_output, bias=bias))
        else:
            self.linear_list.append(nn.Linear(d_input, d_hidden, bias=bias))
            for _ in range(n_layers - 2):
                self.linear_list.append(nn.Linear(d_hidden, d_hidden, bias=bias))
            self.linear_list.append(nn.Linear(d_hidden, d_output, bias=bias))
        if use_batchnorm:
            self.batch_norm_list = nn.ModuleList()
            for _ in range(n_layers):
                self.batch_norm_list.append(nn.BatchNorm1d((d_hidden)))
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        for i in range(self.n_layers - 1):
            h = self.linear_list[i](h)
            if self.use_batchnorm:
                self.batch_norm_list[i](h)          # result discarded -- as in the authors' code
            h = self.dropout(self.activation(h))
        h = self.linear_list[-1](h)
        if self.use_batchnorm and self.out_batchnorm:
            self.batch_norm_list[-1](h)              # result discarded -- as in the authors' code
        if self.out_batchnorm:
            h = self.dropout(self.activation(h))
        return h


def _spmm_batch(A, h, n_nodes):
    """sum_u A[v,u] h_u for a batch of B identical graphs; h (B*N, D) graph-major."""
    B = h.shape[0] // n_nodes
    D = h.shape[1]
    hb = h.reshape(B, n_nodes, D).permute(1, 0, 2).reshape(n_nodes, B * D)
    return torch.sparse.mm(A, hb).reshape(n_nodes, B, D).permute(1, 0, 2).reshape(B * n_nodes, D)


class GINConv(nn.Module):
    """dgl.nn.pytorch.conv.GINConv(apply_func, aggregator_type='sum', learn_eps=False)
    for a batch of B identical graphs: h is (B*N, D) in dgl.batch order
    (graph-major), A is the (N, N) edge-count matrix with A[v, u] = #edges u->v.
    With use_ld=True, A is a pair (A_ppi, A_ld) and the aggregation gains
    ld_weight * A_ld h (LD extension; see module docstring)."""
    def __init__(self, apply_func, init_eps=0.0, learn_eps=False, use_ld=False, ld_init=0.0):
        super().__init__()
        self.apply_func = apply_func
        self.use_ld = use_ld
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))
        if use_ld:
            self.ld_weight = nn.Parameter(torch.FloatTensor([ld_init]))

    def forward(self, A, h, n_nodes):
        if self.use_ld:
            A_ppi, A_ld = A
            agg = _spmm_batch(A_ppi, h, n_nodes) + self.ld_weight * _spmm_batch(A_ld, h, n_nodes)
        else:
            agg = _spmm_batch(A, h, n_nodes)
        rst = (1 + self.eps) * h + agg
        return self.apply_func(rst)


def sum_nodes(v, w, n_nodes):
    """dgl.readout.sum_nodes(g, 'v', 'w') for B identical graphs: v (B*N, D), w (B*N, 1) -> (B, D)."""
    B = v.shape[0] // n_nodes
    return (v * w).reshape(B, n_nodes, -1).sum(dim=1)


class AttentiveReadout(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.in_feats = in_feats
        self.key_layer = nn.Linear(in_feats, in_feats)
        self.query_layer = nn.Sequential(nn.Linear(in_feats, 1, bias=False), nn.Sigmoid())
        self.value_layer = nn.Linear(in_feats, in_feats)
        self.proj = nn.Linear(2 * in_feats, in_feats)     # unused by the authors too; kept for state_dict parity

    def forward(self, feats, n_nodes):
        keys = self.key_layer(feats)
        w = self.query_layer(keys)
        v = self.value_layer(feats)
        return sum_nodes(v, w, n_nodes), w


class AttentiveReadoutMOE(nn.Module):
    """Multi-ancestry readout (not used for our EUR-only runs; ported for completeness)."""
    def __init__(self, in_feats):
        super().__init__()
        self.in_feats = in_feats
        self.ph_key_layer = nn.Linear(in_feats, in_feats)
        self.ph_value_layer = nn.Linear(in_feats, in_feats)
        self.ph_query_layer = nn.Embedding(1, in_feats)
        self.ancestry_key_layer = nn.Linear(in_feats, in_feats)
        self.ancestry_value_layer = nn.Linear(in_feats, in_feats)
        self.ancestry_query_layer = nn.Embedding(num_embeddings=4, embedding_dim=in_feats)

    def forward(self, feats, n_nodes, ancestries):
        B = feats.shape[0] // n_nodes
        ancestry_querys = self.ancestry_query_layer(ancestries)
        ph_querys = self.ph_query_layer.weight.reshape(1, -1)
        ph_keys = self.ph_key_layer(feats).reshape(B, -1, self.in_feats)
        ancestry_keys = self.ancestry_key_layer(feats).reshape(B, -1, self.in_feats)
        ph_w = torch.sigmoid(torch.einsum("BD,BND->BN", ph_querys, ph_keys).reshape(-1, 1))
        ph_v = self.ph_value_layer(feats)
        ph_h = sum_nodes(ph_v, ph_w, n_nodes)
        anc_w = torch.sigmoid(torch.einsum("BD,BND->BN", ancestry_querys, ancestry_keys).reshape(-1, 1))
        anc_v = self.ancestry_value_layer(feats)
        anc_h = sum_nodes(anc_v, anc_w, n_nodes)
        return ph_h + anc_h, ph_w, anc_w


class PRSNet(nn.Module):
    def __init__(self, multiple_ancestries=False, d_input=11, d_hidden=64, n_gene_encode_layer=1, n_layers=1,
                 n_genes=19836, n_predictor_layer=2, mlp_hidden_ratio=1, pre_norm=False, use_ld=False, ld_init=0.0):
        super().__init__()
        self.multiple_ancestries = multiple_ancestries
        self.activation = nn.GELU()
        self.n_layers = n_layers
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_genes = n_genes
        self.pre_norm = pre_norm
        self.use_ld = use_ld
        if self.pre_norm:
            self.pre_bn = nn.BatchNorm1d(d_input, affine=False)
        self.gene_encoder = MLP(d_input=d_input, d_hidden=d_hidden, d_output=d_hidden, n_layers=n_gene_encode_layer,
                                activation=self.activation, bias=True, use_batchnorm=True, out_batchnorm=True)
        self.gene_embeddings = nn.Embedding(n_genes, d_hidden)
        self.gnn_layer_list = nn.ModuleList()
        for _ in range(n_layers):
            mlp = MLP(d_hidden, d_hidden * mlp_hidden_ratio, d_hidden, n_layers=n_gene_encode_layer,
                      activation=self.activation, bias=False, use_batchnorm=True, out_batchnorm=False)
            self.gnn_layer_list.append(GINConv(mlp, learn_eps=False, use_ld=use_ld, ld_init=ld_init))
        self.batch_norm_list = nn.ModuleList()
        for _ in range(n_layers):
            self.batch_norm_list.append(nn.BatchNorm1d(d_hidden))
        if self.multiple_ancestries:
            self.readout = AttentiveReadoutMOE(d_hidden)
        else:
            self.readout = AttentiveReadout(d_hidden)
        self.predictor = MLP(d_input=d_hidden, d_hidden=d_hidden, d_output=1, n_layers=n_predictor_layer, dropout=0,
                             activation=self.activation, bias=True, use_batchnorm=True, out_batchnorm=False)
        self.apply(bert_init_params)
        if use_ld:                                   # bert_init does not touch it, but be explicit
            for g in self.gnn_layer_list:
                g.ld_weight.data.fill_(ld_init)

    def ld_weights(self):
        return [float(g.ld_weight.item()) for g in self.gnn_layer_list] if self.use_ld else []

    def forward(self, A, x, ancestries=None, return_hidden=False):
        """A: (n_genes, n_genes) sparse edge-count matrix (or (A_ppi, A_ld) when use_ld);
        x: (B, n_genes*d_input) or (B, n_genes, d_input)."""
        x = x.reshape(x.shape[0], -1, self.d_input)
        batch_size, n_gene, _ = x.shape
        assert n_gene == self.n_genes, f"features carry {n_gene} genes, model has {self.n_genes}"
        x = x.reshape(-1, self.d_input)
        if self.pre_norm:
            x = self.pre_bn(x)
        h = self.gene_encoder(x)
        h = h.reshape(batch_size, n_gene, self.d_hidden)
        h = h + self.gene_embeddings.weight
        h = h.reshape(-1, self.d_hidden)
        hidden_rep = [h]
        for i in range(self.n_layers):
            h = self.gnn_layer_list[i](A, h, n_gene)
            h = self.batch_norm_list[i](h)      # this one IS applied in the authors' code
            h = F.gelu(h)
            hidden_rep.append(h)
        if self.multiple_ancestries:
            g_h, ph_w, anc_w = self.readout(hidden_rep[-1], n_gene, ancestries)
            out = (self.predictor(g_h), ph_w, anc_w)
        else:
            g_h, weights = self.readout(hidden_rep[-1], n_gene)
            out = (self.predictor(g_h), weights)
        return out + (hidden_rep,) if return_hidden else out


if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else "data/ggi_graph.bin"
    print(json.dumps(graph_summary(path), indent=2))
    A, n = load_ggi_graph(path)
    m = PRSNet(n_genes=n)
    print("parameters:", sum(p.numel() for p in m.parameters()))
    with torch.no_grad():
        p, w = m(A, torch.randn(2, n * 11))
    print("forward ok:", tuple(p.shape), tuple(w.shape))
