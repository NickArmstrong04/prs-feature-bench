#!/usr/bin/env python3
"""sparse_score.py -- ONE kernel for both feature types.

    python sparse_score.py --bed <prefix> --weights <long.tsv> --out <dir>
                           [--mode sum|ct] [--thresholds "1e-5 ... 1"] [--block 4000] [--threads 4]

score(i, gene g, channel c) = sum_j w_{j,g,c} * dosage_i(a1_j), missing calls
mean-imputed (plink2's default rule, the one 12c uses).

  --mode sum   (eQTL)   the channel is a cell type; that sum is the feature.
  --mode ct    (C+T)    the channel is a P-bin label "P<=<t>" from --thresholds,
                        w = GWAS beta of an index SNP in its bin; the feature at
                        threshold t is the cumulative sum over bins <= t divided
                        by 2 * (cumulative number of index SNPs), 0 where that
                        count is 0 -- plink2's SCORE1_AVG, exactly as 12c.

Weight file contract (same as eqtl/weights_to_plink2.py): gene snp a1 channel
beta, one effect allele per SNP across rows, snp = .bim id. Dosages are read
once per SNP block and multiplied by the block's slice of the sparse weight
matrix; the orientation of a1 against the .bim is folded into the weights
(negated beta + 2*beta constant per column), so no per-SNP flipping of the
dosage columns is needed.

This is v0 of the shared kernel: plain and readable. Optimisations go in
copies under variants/, never here. Memory: one float64 array of
n x columns (the matmul result) is the working set; the ct post-processing is
in place and the float32 output is streamed. 381k x 11 x ~1,500 genes is
~50 GB, so a per-chromosome job at production scale needs column chunking
(README section 7, item 3) -- that is the first optimisation to make.
"""
import argparse, os, time
import numpy as np
import scipy.sparse as sp
from bed_reader import open_bed


def load_weights(path, bed, fixed_cols=None):
    """-> (W csr (n_bim_snps x m), const (m,), colmeta [[gene, channel, n_snps]], used snp rows)
    fixed_cols: optional list of (gene, channel) defining the column layout up front
    (ct mode: every gene x bin, gene-major); rows whose column is not in it are an error."""
    sid = {s: i for i, s in enumerate(bed.sid)}
    a1s, a2s = bed.allele_1, bed.allele_2
    rows, cols, vals = [], [], []
    if fixed_cols is None:
        colidx, colmeta, const = {}, [], []
    else:
        colidx = {k: j for j, k in enumerate(fixed_cols)}
        colmeta = [[g, c, 0] for g, c in fixed_cols]; const = [0.0] * len(fixed_cols)
    snp_a1 = {}
    n_in = n_nobim = n_badallele = 0
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t"); ix = {h: i for i, h in enumerate(hdr)}
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if len(f) < len(hdr):
                continue
            n_in += 1
            r = sid.get(f[ix["snp"]])
            if r is None:
                n_nobim += 1; continue
            a1 = f[ix["a1"]]
            beta = float(f[ix["beta"]])
            if a1 == a1s[r]:
                w, c0 = beta, 0.0
            elif a1 == a2s[r]:
                w, c0 = -beta, 2.0 * beta           # dosage(a2) = 2 - dosage(a1)
            else:
                n_badallele += 1; continue          # not a .bim allele: dropped, as the oracle does
            if snp_a1.setdefault(r, a1) != a1:
                raise SystemExit(f"weights: SNP {f[ix['snp']]} given on two alleles ({a1}, {snp_a1[r]}); one a1 per SNP")
            key = (f[ix["gene"]], f[ix["channel"]])
            if key not in colidx:
                if fixed_cols is not None:
                    raise SystemExit(f"weights: column {key} is not in the fixed layout (gene not in --all-genes, or bad bin label)")
                colidx[key] = len(colmeta); colmeta.append([key[0], key[1], 0]); const.append(0.0)
            c = colidx[key]
            rows.append(r); cols.append(c); vals.append(w); const[c] += c0; colmeta[c][2] += 1
    W = sp.csr_matrix((vals, (rows, cols)), shape=(len(bed.sid), len(colmeta)))
    used = np.unique(np.asarray(rows, dtype=np.int64)) if rows else np.zeros(0, dtype=np.int64)
    print(f"weights: {n_in} rows; dropped {n_nobim} not-in-bim, {n_badallele} bad-allele; "
          f"{len(used)} SNPs x {len(colmeta)} columns, nnz {W.nnz}", flush=True)
    return W, np.asarray(const), colmeta, used


def matmul_blocks(bed, W, used, block, threads):
    """S (n x m) float64 = X[:, used] @ W[used, :] with mean imputation, in SNP blocks."""
    n = bed.iid_count
    S = np.zeros((n, W.shape[1]), dtype=np.float64)
    for b0 in range(0, len(used), block):
        blk = used[b0: b0 + block]
        X = bed.read(index=np.s_[:, blk], dtype="float32", order="F", num_threads=threads)
        if np.isnan(X).any():
            mean = np.nanmean(X, axis=0, dtype=np.float64)
            mean = np.where(np.isnan(mean), 0.0, mean)      # all-missing SNP scores as 0
            ii, jj = np.where(np.isnan(X)); X[ii, jj] = mean[jj]
        S += X.astype(np.float64) @ W[blk, :]
    return S


def ct_finish_inplace(S, colmeta, T):
    """S is (n, G*T) gene-major with the T bins of each gene adjacent (the fixed
    layout). Nested-threshold averages = cumsum over bins / (2 * cumulative
    index-SNP count), 0 where the count is 0. Done IN PLACE: no second copy of
    S. Returns colmeta with n_snps replaced by the cumulative count."""
    n, m = S.shape; G = m // T
    V = S.reshape(n, G, T)                       # a view, no copy
    np.cumsum(V, axis=2, out=V)
    cnt = np.array([k for _, _, k in colmeta], dtype=np.int64).reshape(G, T)
    cc = np.cumsum(cnt, axis=1)
    inv = np.where(cc > 0, 1.0 / (2.0 * np.maximum(cc, 1)), 0.0)   # (G, T)
    V *= inv[None, :, :]
    return [[g, c, int(cc[j // T, j % T])] for j, (g, c, _) in enumerate(colmeta)]


def write_layout(out_dir, bed, S, colmeta):
    os.makedirs(out_dir, exist_ok=True)
    n, m = S.shape
    mm = np.memmap(os.path.join(out_dir, "scores.f32"), dtype=np.float32, mode="w+", shape=(n, m))
    step = max(1, (256 << 20) // (m * 4))          # ~256 MB of float32 rows at a time
    for r0 in range(0, n, step):
        mm[r0:r0 + step] = S[r0:r0 + step]
    mm.flush(); del mm
    with open(os.path.join(out_dir, "samples.txt"), "w") as w:
        for f, i in zip(bed.fid, bed.iid):
            w.write(f"{f}\t{i}\n")
    with open(os.path.join(out_dir, "columns.tsv"), "w") as w:
        w.write("col\tgene\tchannel\tn_snps\n")
        for j, (g, c, k) in enumerate(colmeta):
            w.write(f"{j}\t{g}\t{c}\t{k}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--mode", choices=["sum", "ct"], default="sum")
    ap.add_argument("--thresholds", default=""); ap.add_argument("--block", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--all-genes", default="", help="ct mode: file of gene names that must get columns even with no index SNP")
    a = ap.parse_args()
    t0 = time.time()
    bed = open_bed(a.bed + ".bed", num_threads=a.threads)
    fixed = None
    if a.mode == "ct":
        thr = a.thresholds.split()
        assert thr and a.all_genes, "--mode ct needs --thresholds and --all-genes"
        genes = []
        for l in open(a.all_genes):
            if l.strip() and l.strip() not in genes:
                genes.append(l.strip())
        fixed = [(g, f"P<={t}") for g in genes for t in thr]          # gene-major, bins adjacent
    W, const, colmeta, used = load_weights(a.weights, bed, fixed)
    S = matmul_blocks(bed, W, used, a.block, a.threads)
    S += const[None, :]
    print(f"matmul done ({time.time() - t0:.1f}s; S {S.nbytes / 2**30:.2f} GiB)", flush=True)
    if a.mode == "ct":
        colmeta = ct_finish_inplace(S, colmeta, len(thr))
    write_layout(a.out, bed, S, colmeta)
    print(f"sparse_score[{a.mode}]: {S.shape[0]} x {S.shape[1]} -> {a.out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
