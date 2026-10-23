#!/usr/bin/env python3
"""
prsnet_ld.py -- graph variants and the embedding-level LD-leakage metric for
the PRS-Net LD experiments (future_directions.md §2A.5, ld_experiments.md).

Graph variants (all return the same thing load_ggi_graph does: a torch sparse
CSR (n, n) matrix A with A[v, u] = stored multiplicity of u -> v, in the
authors' storage convention -- each undirected PPI twice per direction, one
self-loop per node):

  identity_graph(n)                     self-loops only; the GIN reduces to a per-gene MLP
  shuffled_graph(src, dst, n, seed)     degree-preserving rewiring of the PPI set
                                        (double-edge swaps on the simple undirected graph)
  ld_graph(pairs_tsv, gene_order, ...)  genomic-adjacency edges from 14_ld_leakage.py's
                                        pairs.tsv.gz, weighted by the measured r

Embedding leakage: embedding_leakage(model, A, store, rows, pairs, device)
streams the sampled people through the model and returns, per layer
('input_P<=1' = the raw feature, 'enc' = after the gene encoder, 'gnn' = after
the GIN layer, i.e. what the readout sees), the per-class summary of pairwise
node-embedding correlations: for a gene pair (a, b), r_ab = mean over the 64
embedding dims d of corr_people(h_a,d, h_b,d). Same classes and statistics as
14_ld_leakage.py's summary.tsv, so the two tables are directly comparable.
Nothing is stored per person; sums are accumulated on the GPU.
"""
import gzip
import warnings
import numpy as np
import torch

LEAK_CLASSES = ("ppi_far", "ppi_near", "adjacent_no_shared_snp", "adjacent_shared_snp", "random_diff_chr")
ADJ_CLASSES = ("adjacent_no_shared_snp", "adjacent_shared_snp", "adjacent", "ppi_near")


def _csr(src, dst, vals, n, device=None, dtype=torch.float32):
    import scipy.sparse as sp
    C = sp.coo_matrix((np.asarray(vals, dtype=np.float32), (np.asarray(dst), np.asarray(src))), shape=(n, n)).tocsr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = torch.sparse_csr_tensor(torch.from_numpy(C.indptr.astype(np.int64)), torch.from_numpy(C.indices.astype(np.int64)),
                                    torch.from_numpy(C.data).to(dtype), size=(n, n))
    return A.to(device) if device is not None else A


def load_edges(path):
    """(src, dst, n) of a stored graph: .npz (src, dst, n_nodes) or the authors' .bin."""
    if path.endswith(".npz"):
        z = np.load(path)
        return z["src"].astype(np.int64), z["dst"].astype(np.int64), int(z["n_nodes"])
    from prsnet_torch import read_dgl_graph_edges
    return read_dgl_graph_edges(path)


def undirected_simple(src, dst):
    """Unique undirected non-self pairs (a < b) of a stored multigraph."""
    m = src != dst
    a = np.minimum(src[m], dst[m]); b = np.maximum(src[m], dst[m])
    return np.unique(np.stack([a, b], 1), axis=0)


def stored_from_undirected(pairs, n, multiplicity=2):
    """Authors' convention: each pair `multiplicity` times per direction + one self-loop per node."""
    a, b = pairs[:, 0], pairs[:, 1]
    src = np.concatenate([np.tile(a, multiplicity), np.tile(b, multiplicity), np.arange(n)])
    dst = np.concatenate([np.tile(b, multiplicity), np.tile(a, multiplicity), np.arange(n)])
    return src, dst


def identity_graph(n, device=None):
    return _csr(np.arange(n), np.arange(n), np.ones(n), n, device)


def shuffled_graph(src, dst, n, seed=0, swaps_per_edge=10, device=None):
    """Degree-preserving rewiring: double-edge swaps on the simple undirected
    graph, rejecting self-loops and duplicate edges. Returns (A, n_swapped)."""
    pairs = undirected_simple(src, dst)
    E = len(pairs)
    edges = [tuple(p) for p in pairs.tolist()]
    present = set(edges)
    rng = np.random.default_rng(seed)
    done = 0
    for _ in range(swaps_per_edge * E):
        i, j = rng.integers(0, E, size=2)
        if i == j:
            continue
        a, b = edges[i]; c, d = edges[j]
        if rng.random() < 0.5:
            c, d = d, c
        e1 = (min(a, d), max(a, d)); e2 = (min(c, b), max(c, b))
        if a == d or c == b or e1 in present or e2 in present:
            continue
        present.discard(edges[i]); present.discard(edges[j])
        present.add(e1); present.add(e2)
        edges[i], edges[j] = e1, e2
        done += 1
    new = np.array(edges, dtype=np.int64)
    s, d = stored_from_undirected(new, n)
    # sanity: degree sequence preserved
    deg0 = np.bincount(pairs.ravel(), minlength=n); deg1 = np.bincount(new.ravel(), minlength=n)
    assert np.array_equal(deg0, deg1), "degree sequence changed"
    return _csr(s, d, np.ones(len(s)), n, device), done, E


def read_pairs(path, gene_index, classes=None, r_col="r_P<=1"):
    """Rows of pairs.tsv.gz -> dict class -> (A idx, B idx, r). Pairs with a
    gene outside gene_index or a nan r are dropped."""
    out = {}
    with gzip.open(path, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ci, ca, cb = hdr.index("class"), hdr.index("gene_a"), hdr.index("gene_b")
        if r_col not in hdr:
            raise SystemExit(f"{path}: column {r_col} not found (have {hdr})")
        cr = hdr.index(r_col)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            c = f[ci]
            if classes is not None and c not in classes:
                continue
            a, b = gene_index.get(f[ca]), gene_index.get(f[cb])
            if a is None or b is None or f[cr] == "nan":
                continue
            out.setdefault(c, [[], [], []])
            out[c][0].append(a); out[c][1].append(b); out[c][2].append(float(f[cr]))
    return {c: (np.array(v[0]), np.array(v[1]), np.array(v[2], dtype=np.float32)) for c, v in out.items()}


def ld_graph(pairs_path, gene_order, weight="signed", min_abs_r=0.0, r_col="r_P<=1", device=None):
    """Genomic-adjacency graph from the diagnostic's adjacent pairs.
    weight: signed (the measured r), abs (|r|), binary (1). Symmetric, no self-loops.
    Returns (A, n_edges_undirected, mean |r| of the kept edges)."""
    gi = {g: i for i, g in enumerate(gene_order)}
    pairs = read_pairs(pairs_path, gi, classes=set(ADJ_CLASSES), r_col=r_col)
    A_, B_, R_ = [], [], []
    for c, (a, b, r) in pairs.items():
        A_.append(a); B_.append(b); R_.append(r)
    a = np.concatenate(A_); b = np.concatenate(B_); r = np.concatenate(R_)
    keep = np.abs(r) >= min_abs_r
    a, b, r = a[keep], b[keep], r[keep]
    w = r if weight == "signed" else (np.abs(r) if weight == "abs" else np.ones_like(r))
    src = np.concatenate([a, b]); dst = np.concatenate([b, a]); vals = np.concatenate([w, w])
    return _csr(src, dst, vals, len(gene_order), device), int(len(a)), float(np.abs(r).mean()) if len(r) else 0.0


# ----------------------------------------------------------------------------
def load_leak_pairs(pairs_path, gene_order, classes=LEAK_CLASSES, max_random=50000, seed=0):
    gi = {g: i for i, g in enumerate(gene_order)}
    pairs = read_pairs(pairs_path, gi, classes=set(classes))
    if "random_diff_chr" in pairs and len(pairs["random_diff_chr"][0]) > max_random:
        a, b, r = pairs["random_diff_chr"]
        sel = np.random.default_rng(seed).choice(len(a), size=max_random, replace=False)
        pairs["random_diff_chr"] = (a[sel], b[sel], r[sel])
    return pairs


def _describe(r):
    ar = np.abs(r)
    return dict(n=int(len(r)), mean_r=float(r.mean()), mean_abs=float(ar.mean()), median_abs=float(np.median(ar)),
                p90_abs=float(np.percentile(ar, 90)), frac_gt_0_1=float((ar > 0.1).mean()), frac_gt_0_3=float((ar > 0.3).mean()))


@torch.no_grad()
def embedding_leakage(model, A, store, rows, pairs, device, batch=128, pair_chunk=None, input_threshold_idx=None):
    """See module docstring. Returns {layer: {class: describe-dict}}."""
    model.eval()
    if pair_chunk is None:                       # keep the per-gather temporaries near 1 GB: batch x chunk x 64 x 4 B x 2
        pair_chunk = max(1024, (1 << 22) // max(batch, 1))
    N, D = model.n_genes, model.d_hidden
    T = store.T
    if input_threshold_idx is None:
        input_threshold_idx = T - 1
    classes = list(pairs.keys())
    PA = torch.from_numpy(np.concatenate([pairs[c][0] for c in classes])).to(device)
    PB = torch.from_numpy(np.concatenate([pairs[c][1] for c in classes])).to(device)
    sizes = [len(pairs[c][0]) for c in classes]
    M = len(PA)
    layers = {"input_P<=%s" % store.meta["thresholds"][input_threshold_idx]: 1, "enc": D, "gnn": D}
    S1 = {k: torch.zeros(N, d, dtype=torch.float64, device=device) for k, d in layers.items()}
    S2 = {k: torch.zeros(N, d, dtype=torch.float64, device=device) for k, d in layers.items()}
    SP = {k: torch.zeros(M, d, dtype=torch.float64, device=device) for k, d in layers.items()}
    rows = np.asarray(rows)
    for i in range(0, len(rows), batch):
        x = store.rows(rows[i:i + batch]).to(device, non_blocking=True)
        B = x.shape[0]
        _, _, hidden = model(A, x, return_hidden=True)
        feats = {list(layers)[0]: x.reshape(B, N, T)[:, :, input_threshold_idx:input_threshold_idx + 1].float(),
                 "enc": hidden[0].reshape(B, N, D).float(), "gnn": hidden[-1].reshape(B, N, D).float()}
        for k, h in feats.items():
            S1[k] += h.sum(0).double(); S2[k] += (h * h).sum(0).double()
            for c0 in range(0, M, pair_chunk):
                a = PA[c0:c0 + pair_chunk]; b = PB[c0:c0 + pair_chunk]
                SP[k][c0:c0 + pair_chunk] += torch.einsum("bmd,bmd->md", h[:, a, :], h[:, b, :]).double()
    n = float(len(rows))
    out = {}
    for k in layers:
        mean = S1[k] / n
        var = (S2[k] / n - mean * mean).clamp_min(0)
        cov = SP[k] / n - mean[PA] * mean[PB]
        denom = torch.sqrt(var[PA] * var[PB])
        r = torch.where(denom > 1e-12, cov / denom.clamp_min(1e-12), torch.full_like(cov, float("nan")))
        r_pair = torch.nanmean(r, dim=1).cpu().numpy()          # mean over embedding dims
        res, off = {}, 0
        for c, sz in zip(classes, sizes):
            rc = r_pair[off:off + sz]; off += sz
            rc = rc[~np.isnan(rc)]
            if len(rc):
                res[c] = _describe(rc)
        out[k] = res
    return out


def write_leakage(path, res, split=None, extra=None):
    cols = ["layer", "class", "n", "mean_r", "mean_abs_r", "median_abs_r", "p90_abs_r", "frac_abs_gt_0.1", "frac_abs_gt_0.3"]
    new = not __import__("os").path.exists(path)
    with open(path, "a") as w:
        if new:
            w.write("\t".join((["split"] if split is not None else []) + cols) + "\n")
        for layer, classes in res.items():
            for c, d in classes.items():
                vals = [layer, c, d["n"], f"{d['mean_r']:.4f}", f"{d['mean_abs']:.4f}", f"{d['median_abs']:.4f}",
                        f"{d['p90_abs']:.4f}", f"{d['frac_gt_0_1']:.4f}", f"{d['frac_gt_0_3']:.4f}"]
                w.write("\t".join(([str(split)] if split is not None else []) + [str(v) for v in vals]) + "\n")


def format_leakage(res):
    lines = [f"  {'layer':14s}{'class':26s}{'n':>8}{'mean r':>8}{'mean|r|':>9}{'med|r|':>8}{'>0.1':>7}{'>0.3':>7}"]
    for layer, classes in res.items():
        for c, d in classes.items():
            lines.append(f"  {layer:14s}{c:26s}{d['n']:>8,}{d['mean_r']:>8.3f}{d['mean_abs']:>9.3f}{d['median_abs']:>8.3f}"
                         f"{d['frac_gt_0_1']:>7.3f}{d['frac_gt_0_3']:>7.3f}")
    return "\n".join(lines)
