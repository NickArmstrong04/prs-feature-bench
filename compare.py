#!/usr/bin/env python3
"""compare.py -- fidelity check of a candidate scores directory against a golden.

    python compare.py <golden_dir> <candidate_dir> [--rtol 1e-5] [--atol 1e-7] [--show 15]

Each directory is either the flat scores layout (samples.txt, columns.tsv,
scores.f32 float32 (n, m), README s2) or 12c's chunk layout (samples.txt,
scores_manifest.tsv, genes_chunk<k>.txt, scores_chunk<k>.npy (n, G_k, T),
gene_counts.tsv), told apart by scores_manifest.tsv. Columns are matched on
(gene, channel), samples on IID; order may differ. Exit 0 = PASS, 2 = FAIL.

Rows are compared in blocks (~256 MB of float64 at a time) so a 25 GB
full-cohort matrix is read once, sequentially; reading it column by column
through a memmap would touch every page of the file per column.
"""
import argparse, os, sys
import numpy as np


class Flat:
    def __init__(self, d, n, m):
        path = os.path.join(d, "scores.f32")
        size = os.path.getsize(path)
        if size != n * m * 4:
            sys.exit(f"{path}: {size} bytes, expected {n}x{m}x4 = {n*m*4}")
        self.S = np.memmap(path, dtype=np.float32, mode="r", shape=(n, m))

    def rows(self, r0, r1):
        return np.asarray(self.S[r0:r1])

    def take(self, idx):
        return np.asarray(self.S[idx])


class Chunks:
    """12c's layout as one (n, m) matrix: chunk k holds columns [c_k, c_k + G_k*T)."""
    def __init__(self, d, n, cols_per_chunk):
        self.A = []; self.m = 0
        for k, fn, G, T in cols_per_chunk:
            A = np.load(os.path.join(d, fn), mmap_mode="r")
            if A.shape != (n, G, T):
                sys.exit(f"{fn}: shape {A.shape}, expected ({n}, {G}, {T})")
            self.A.append(A); self.m += G * T

    def _cat(self, parts, k):
        out = np.empty((k, self.m), dtype=np.float32); c = 0
        for p in parts:
            w = p.shape[1] * p.shape[2]
            out[:, c:c + w] = p.reshape(k, w); c += w
        return out

    def rows(self, r0, r1):
        return self._cat([A[r0:r1] for A in self.A], r1 - r0)

    def take(self, idx):
        return self._cat([A[idx] for A in self.A], len(idx))


def load(d):
    with open(os.path.join(d, "samples.txt")) as fh:
        iids = [l.split()[1] for l in fh if l.strip()]
    n = len(iids)
    cols = []
    man = os.path.join(d, "scores_manifest.tsv")
    if os.path.exists(man):
        with open(os.path.join(d, "gene_counts.tsv")) as fh:
            labels = [h[2:] for h in fh.readline().split()[2:]]          # "n_P<=1e-5" -> "P<=1e-5"
        T = len(labels)
        chunks = []
        with open(man) as fh:
            fh.readline()
            for l in fh:
                f = l.split()
                if f:
                    chunks.append((int(f[0]), f[1], int(f[2])))
        chunks.sort()
        spec = []
        for k, fn, G in chunks:
            genes = [l.strip() for l in open(os.path.join(d, f"genes_chunk{k:03d}.txt")) if l.strip()]
            if len(genes) != G:
                sys.exit(f"{d}: genes_chunk{k:03d}.txt has {len(genes)} genes, manifest says {G}")
            cols += [(g, lab) for g in genes for lab in labels]
            spec.append((k, fn, G, T))
        return iids, cols, Chunks(d, n, spec)
    with open(os.path.join(d, "columns.tsv")) as fh:
        hdr = fh.readline().split()
        ig, ic = hdr.index("gene"), hdr.index("channel")
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if f:
                cols.append((f[ig], f[ic]))
    return iids, cols, Flat(d, n, len(cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden"); ap.add_argument("candidate")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-7)
    ap.add_argument("--show", type=int, default=15, help="worst columns to list")
    a = ap.parse_args()

    g_iid, g_cols, G = load(a.golden)
    c_iid, c_cols, C = load(a.candidate)

    # samples: candidate must contain every golden sample
    c_row = {s: i for i, s in enumerate(c_iid)}
    missing_s = [s for s in g_iid if s not in c_row]
    if missing_s:
        print(f"FAIL: {len(missing_s)} golden samples missing from candidate (e.g. {missing_s[:3]})")
        sys.exit(2)
    rows = np.array([c_row[s] for s in g_iid])
    same_order = bool(np.all(rows == np.arange(len(rows))))

    c_col = {c: j for j, c in enumerate(c_cols)}
    missing_c = [c for c in g_cols if c not in c_col]
    g_set = set(g_cols)
    extra_c = [c for c in c_cols if c not in g_set]
    gj = np.array([j for j, c in enumerate(g_cols) if c in c_col], dtype=np.int64)     # golden columns compared
    cj = np.array([c_col[g_cols[j]] for j in gj], dtype=np.int64)                       # their candidate columns
    n_cmp = len(gj)

    # row blocks: per-column running max |diff|, max rel, bad count
    n = len(g_iid)
    mabs = np.zeros(n_cmp); mrel = np.zeros(n_cmp); nbad = np.zeros(n_cmp, dtype=np.int64)
    step = max(1, (256 << 20) // max(1, n_cmp * 8))
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        g = G.rows(r0, r1)[:, gj].astype(np.float64)
        c = (C.rows(r0, r1) if same_order else C.take(rows[r0:r1]))[:, cj].astype(np.float64)
        d = np.abs(c - g)
        tol = a.atol + a.rtol * np.abs(g)
        nbad += np.count_nonzero(d > tol, axis=0)
        if d.size:
            np.maximum(mabs, d.max(axis=0), out=mabs)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(np.abs(g) > a.atol, d / np.abs(g), 0.0)   # cells below atol are judged by atol alone
            np.maximum(mrel, rel.max(axis=0), out=mrel)

    worst = sorted(((float(mabs[i]), float(mrel[i]), int(nbad[i]), g_cols[gj[i]][0], g_cols[gj[i]][1]) for i in range(n_cmp)), reverse=True)
    n_bad_cols = int(np.count_nonzero(nbad))
    print(f"golden {a.golden}: {len(g_iid)} samples x {len(g_cols)} columns")
    print(f"cand   {a.candidate}: {len(c_iid)} samples x {len(c_cols)} columns")
    print(f"compared {n_cmp} columns on {len(g_iid)} samples  (rtol {a.rtol:g}, atol {a.atol:g}; sample order {'identical' if same_order else 'remapped'})")
    if extra_c:
        print(f"note: {len(extra_c)} extra columns in candidate (not compared), e.g. {extra_c[:3]}")
    print(f"{'gene':<18}{'channel':<14}{'max|diff|':>12}{'max rel':>12}{'bad':>8}")
    for mx, mr, bad, gene, ch in worst[: a.show]:
        print(f"{gene:<18}{ch:<14}{mx:>12.3e}{mr:>12.3e}{bad:>8}")

    ok = not missing_c and n_bad_cols == 0
    gmax_abs = max((w[0] for w in worst), default=0.0)
    gmax_rel = max((w[1] for w in worst), default=0.0)
    if missing_c:
        print(f"MISSING: {len(missing_c)} golden columns absent from candidate, e.g. {missing_c[:5]}")
    line = (f"{'PASS' if ok else 'FAIL'}: {n_cmp} cols, {n_bad_cols} bad cols, "
            f"{len(missing_c)} missing, max|diff| {gmax_abs:.3e}, max rel {gmax_rel:.3e}")
    print(line)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
