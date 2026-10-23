#!/usr/bin/env python3
"""compare_dosage.py -- exact fidelity check for the dosage layout (PRSNet-2 features).

    python compare_dosage.py <golden_dir> <candidate_dir>

Both hold samples.txt, snps.tsv, X.npy (int8), pvalues.npy, gene2snps.tsv (see
README s2). Samples are matched on IID, SNP columns on snp id; order may
differ. PASS requires: every golden sample and SNP present; the counted allele
(a1) identical per SNP; X identical cell for cell (int8, 3 = missing); P
identical per SNP; gene -> set of SNPs identical for every golden gene.
Extra candidate SNPs/genes are reported, not compared. Exit 0 = PASS, 2 = FAIL.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from x2bit import unpack


class _Unpacked:
    """Row-sliceable view of an X2.npy memmap that unpacks on access (compare reads row blocks)."""
    def __init__(self, X2, m):
        self.X2, self.m = X2, m

    def __getitem__(self, idx):
        if isinstance(idx, tuple):                         # X[rows][:, cols] is two steps; X[r0:r1] is one
            rows, cols = idx
            return unpack(self.X2[rows], self.m)[:, cols]
        return unpack(self.X2[idx], self.m)


def load(d):
    with open(os.path.join(d, "samples.txt")) as fh:
        iids = [l.split()[1] for l in fh if l.strip()]
    snps, a1 = [], []
    with open(os.path.join(d, "snps.tsv")) as fh:
        hdr = fh.readline().split(); isnp, ia1 = hdr.index("snp"), hdr.index("a1")
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if f:
                snps.append(f[isnp]); a1.append(f[ia1])
    if os.path.exists(os.path.join(d, "X.npy")):
        X = np.load(os.path.join(d, "X.npy"), mmap_mode="r")
        if X.shape != (len(iids), len(snps)) or X.dtype != np.int8:
            sys.exit(f"{d}/X.npy: shape {X.shape} dtype {X.dtype}, expected ({len(iids)}, {len(snps)}) int8")
    else:                                                  # 2-bit store: unpacked row block by row block
        X2 = np.load(os.path.join(d, "X2.npy"), mmap_mode="r")
        B = (len(snps) + 3) // 4
        if X2.shape != (len(iids), B) or X2.dtype != np.uint8:
            sys.exit(f"{d}/X2.npy: shape {X2.shape} dtype {X2.dtype}, expected ({len(iids)}, {B}) uint8")
        X = _Unpacked(X2, len(snps))
    p = np.load(os.path.join(d, "pvalues.npy"))
    g2 = {}
    with open(os.path.join(d, "gene2snps.tsv")) as fh:
        fh.readline()
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if len(f) >= 3:
                g2[f[0]] = set(f[2].split(",")) if f[2] else set()
    return iids, snps, a1, X, p, g2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden"); ap.add_argument("candidate")
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args()
    for d in (a.golden, a.candidate):
        if not (os.path.exists(os.path.join(d, "X.npy")) or os.path.exists(os.path.join(d, "X2.npy"))):
            sys.exit(f"{d}: no X.npy / X2.npy -- not a dosage-layout output (golden and candidate must share a contract)")
    g_iid, g_snp, g_a1, G, g_p, g_g2 = load(a.golden)
    c_iid, c_snp, c_a1, C, c_p, c_g2 = load(a.candidate)

    c_row = {s: i for i, s in enumerate(c_iid)}
    missing_s = [s for s in g_iid if s not in c_row]
    if missing_s:
        print(f"FAIL: {len(missing_s)} golden samples missing from candidate (e.g. {missing_s[:3]})"); sys.exit(2)
    rows = np.array([c_row[s] for s in g_iid])
    same_rows = bool(np.all(rows == np.arange(len(rows))))
    c_col = {s: j for j, s in enumerate(c_snp)}
    missing_c = [s for s in g_snp if s not in c_col]
    extra_c = [s for s in c_snp if s not in set(g_snp)]
    cols = np.array([c_col[s] for s in g_snp if s in c_col])
    gcols = np.array([j for j, s in enumerate(g_snp) if s in c_col])
    same_cols = bool(len(cols) == len(c_snp) and np.all(cols == np.arange(len(cols))))

    bad_a1 = [s for j, s in enumerate(g_snp) if s in c_col and c_a1[c_col[s]] != g_a1[j]]
    bad_p = int(np.count_nonzero(g_p[gcols] != c_p[cols])) if len(cols) else 0

    # X, in row blocks to bound memory
    n_bad_cells, worst = 0, []       # worst: (bad cells, snp)
    bad_per_col = np.zeros(len(cols), dtype=np.int64)
    step = max(1, (256 << 20) // max(1, len(cols)))
    for r0 in range(0, len(g_iid), step):
        g = np.asarray(G[r0:r0 + step][:, gcols]) if not same_cols else np.asarray(G[r0:r0 + step])
        if same_rows and same_cols:
            c = np.asarray(C[r0:r0 + step])
        else:
            c = np.asarray(C[rows[r0:r0 + step]][:, cols])
        neq = g != c
        bad_per_col += neq.sum(axis=0)
    n_bad_cells = int(bad_per_col.sum())
    n_bad_cols = int(np.count_nonzero(bad_per_col))
    order = np.argsort(-bad_per_col)[: a.show]

    missing_g = [g for g in g_g2 if g not in c_g2]
    bad_g = [g for g in g_g2 if g in c_g2 and c_g2[g] != g_g2[g]]
    extra_g = [g for g in c_g2 if g not in g_g2]

    print(f"golden {a.golden}: {len(g_iid)} samples x {len(g_snp)} SNPs, {len(g_g2)} genes")
    print(f"cand   {a.candidate}: {len(c_iid)} samples x {len(c_snp)} SNPs, {len(c_g2)} genes")
    print(f"compared {len(cols)} SNPs on {len(g_iid)} samples, exact int8 equality "
          f"(sample order {'identical' if same_rows else 'remapped'}, SNP order {'identical' if same_cols else 'remapped'})")
    if extra_c:
        print(f"note: {len(extra_c)} extra SNPs in candidate (not compared), e.g. {extra_c[:3]}")
    if extra_g:
        print(f"note: {len(extra_g)} extra genes in candidate (not compared), e.g. {extra_g[:3]}")
    if n_bad_cols:
        print(f"{'snp':<20}{'bad cells':>10}")
        for j in order:
            if bad_per_col[j]:
                print(f"{g_snp[gcols[j]]:<20}{bad_per_col[j]:>10}")
    if missing_c:
        print(f"MISSING: {len(missing_c)} golden SNPs absent from candidate, e.g. {missing_c[:5]}")
    if bad_a1:
        print(f"ALLELE: {len(bad_a1)} SNPs count a different allele, e.g. {bad_a1[:5]}")
    if bad_p:
        print(f"PVALUE: {bad_p} SNPs with a different P")
    if missing_g:
        print(f"GENES: {len(missing_g)} golden genes absent from candidate, e.g. {missing_g[:5]}")
    if bad_g:
        print(f"GENES: {len(bad_g)} genes with a different SNP set, e.g. {bad_g[:5]}")
    ok = not (missing_c or bad_a1 or bad_p or missing_g or bad_g or n_bad_cells)
    print(f"{'PASS' if ok else 'FAIL'}: {len(cols)} snps, {n_bad_cols} bad snps, {n_bad_cells} bad cells, "
          f"{len(missing_c)} missing, {len(bad_g) + len(missing_g)} bad genes")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
