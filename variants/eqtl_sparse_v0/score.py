#!/usr/bin/env python3
"""eqtl_sparse_v0 -- first fast implementation of eQTL-weighted gene x cell-type
features: dosage matrix (people x SNPs) times a sparse weight matrix
(SNPs x columns), in SNP blocks.

    python score.py --bed <prefix> --weights <long.tsv> --out <dir> [--block 4000] [--threads 4]

Semantics (must equal variants/ref_eqtl_plink2): dosage of the weight's a1,
missing calls imputed to the mean dosage of called people, score = sum of
beta * dosage. No averaging, no thresholds.

Intentionally plain: read a block of SNPs as int8, impute, orient, cast,
multiply by the block's slice of the sparse weight matrix, accumulate.
Everything that could be optimised later -- bit-level dosage decode, fewer
casts, column chunking for memory, multithreaded reads, keeping only SNPs
that carry weight -- is left visible so each change is a separate variant.
"""
import argparse, os, sys, time
import numpy as np
import scipy.sparse as sp
from bed_reader import open_bed

MISSING_INT8 = -127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=4000); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    bed = open_bed(a.bed + ".bed", num_threads=a.threads)
    n = bed.iid_count
    sid = {s: i for i, s in enumerate(bed.sid)}
    bim_a1 = bed.allele_1          # bed-reader: allele_1 = .bim column 5
    bim_a2 = bed.allele_2

    # --- weights -> sparse (n_snps_used x n_cols), oriented to .bim A1
    rows, cols, vals = [], [], []
    colidx, colmeta = {}, []
    n_in = n_drop = 0
    with open(a.weights) as fh:
        hdr = fh.readline().rstrip("\n").split("\t"); ix = {h: i for i, h in enumerate(hdr)}
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if len(f) < len(hdr):
                continue
            n_in += 1
            r = sid.get(f[ix["snp"]])
            if r is None:
                n_drop += 1; continue
            a1 = f[ix["a1"]]
            if a1 == bim_a1[r]:
                beta = float(f[ix["beta"]])
            elif a1 == bim_a2[r]:
                beta = -float(f[ix["beta"]])      # dosage(a2) = 2 - dosage(a1): constant handled below
            else:
                n_drop += 1; continue
            key = (f[ix["gene"]], f[ix["channel"]])
            if key not in colidx:
                colidx[key] = len(colmeta); colmeta.append([key[0], key[1], 0, 0.0])
            c = colidx[key]
            rows.append(r); cols.append(c); vals.append(beta)
            colmeta[c][2] += 1
            if a1 == bim_a2[r]:
                colmeta[c][3] += 2.0 * float(f[ix["beta"]])   # the constant from the flip
    m = len(colmeta)
    Wmat = sp.csr_matrix((vals, (rows, cols)), shape=(len(bed.sid), m))
    used = np.unique(np.asarray(rows))
    print(f"weights: {n_in} rows, {n_drop} dropped; {len(used)} SNPs x {m} columns; "
          f"nnz {Wmat.nnz}  ({time.time() - t0:.1f}s)", flush=True)

    # --- the product, in blocks over the SNPs that carry weight
    S = np.zeros((n, m), dtype=np.float64)
    for b0 in range(0, len(used), a.block):
        blk = used[b0: b0 + a.block]
        X = bed.read(index=np.s_[:, blk], dtype="float32", order="F", num_threads=a.threads)  # NaN = missing
        if np.isnan(X).any():
            mean = np.nanmean(X, axis=0, dtype=np.float64)   # double accumulation, as plink2
            mean = np.where(np.isnan(mean), 0.0, mean)
            ii, jj = np.where(np.isnan(X))
            X[ii, jj] = mean[jj]
        Wb = Wmat[blk, :]                     # (block x m), sparse
        S += X.astype(np.float64) @ Wb        # dense x sparse -> dense
    const = np.array([cm[3] for cm in colmeta])
    S += const[None, :]
    print(f"product done ({time.time() - t0:.1f}s)", flush=True)

    os.makedirs(a.out, exist_ok=True)
    out = np.memmap(os.path.join(a.out, "scores.f32"), dtype=np.float32, mode="w+", shape=(n, m))
    out[:] = S; out.flush()
    with open(os.path.join(a.out, "samples.txt"), "w") as w:
        for f, i in zip(bed.fid, bed.iid):
            w.write(f"{f}\t{i}\n")
    with open(os.path.join(a.out, "columns.tsv"), "w") as w:
        w.write("col\tgene\tchannel\tn_snps\n")
        for j, (g, c, k, _) in enumerate(colmeta):
            w.write(f"{j}\t{g}\t{c}\t{k}\n")
    print(f"eqtl_sparse_v0: {n} x {m} -> {a.out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
