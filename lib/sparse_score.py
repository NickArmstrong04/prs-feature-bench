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
copies under variants/, never here.
"""
import argparse, os, time
import numpy as np
import scipy.sparse as sp
from bed_reader import open_bed


def load_weights(path, bed, thresholds=None):
    """-> (W csr (n_bim_snps x m), const (m,), colmeta [[gene, channel, n_snps]], used snp rows)"""
    sid = {s: i for i, s in enumerate(bed.sid)}
    a1s, a2s = bed.allele_1, bed.allele_2
    rows, cols, vals = [], [], []
    colidx, colmeta, const = {}, [], []
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


def ct_finish(S, colmeta, thresholds, all_genes=()):
    """Nested-threshold averages from per-bin sums: cumsum over bins, / (2*cumcount).
    Genes in all_genes with no weight (no index SNP) get all-zero columns, as 12c."""
    labels = [f"P<={t}" for t in thresholds]
    lab_ix = {l: k for k, l in enumerate(labels)}
    genes = []
    gi = {}
    for g in all_genes:
        if g not in gi:
            gi[g] = len(genes); genes.append(g)
    for g, ch, _ in colmeta:
        if g not in gi:
            gi[g] = len(genes); genes.append(g)
    T, n = len(labels), S.shape[0]
    sums = np.zeros((len(genes), T, n)); cnt = np.zeros((len(genes), T), dtype=np.int64)
    for c, (g, ch, k) in enumerate(colmeta):
        sums[gi[g], lab_ix[ch]] = S[:, c]; cnt[gi[g], lab_ix[ch]] = k
    np.cumsum(sums, axis=1, out=sums); cc = np.cumsum(cnt, axis=1)
    inv = np.where(cc > 0, 1.0 / (2.0 * np.maximum(cc, 1)), 0.0)
    out = (sums * inv[:, :, None]).reshape(len(genes) * T, n).T      # n x (genes*T), gene-major
    meta = [[g, lab, int(cc[gi[g], t])] for g in genes for t, lab in enumerate(labels)]
    return np.ascontiguousarray(out), meta


def write_layout(out_dir, bed, S, colmeta):
    os.makedirs(out_dir, exist_ok=True)
    n, m = S.shape
    mm = np.memmap(os.path.join(out_dir, "scores.f32"), dtype=np.float32, mode="w+", shape=(n, m))
    mm[:] = S; mm.flush()
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
    W, const, colmeta, used = load_weights(a.weights, bed)
    S = matmul_blocks(bed, W, used, a.block, a.threads) + const[None, :]
    print(f"matmul done ({time.time() - t0:.1f}s)", flush=True)
    if a.mode == "ct":
        thr = a.thresholds.split()
        assert thr, "--mode ct needs --thresholds"
        allg = [l.strip() for l in open(a.all_genes) if l.strip()] if a.all_genes else []
        S, colmeta = ct_finish(S, colmeta, thr, allg)
    write_layout(a.out, bed, S, colmeta)
    print(f"sparse_score[{a.mode}]: {S.shape[0]} x {S.shape[1]} -> {a.out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
