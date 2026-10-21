#!/usr/bin/env python3
"""sparse_score.py (ct_sparse_v1) -- lib/sparse_score.py with COLUMN CHUNKING.

    python sparse_score.py --bed <prefix> --weights <long.tsv> --out <dir>
                           [--mode sum|ct] [--thresholds "1e-5 ... 1"] [--block 4000] [--threads 4]
                           [--chunk-gb 1.0 | --chunk-cols N]

Same contract and same arithmetic as lib/sparse_score.py (v0); the one change
is README section 7 "column chunking": instead of one n x m float64 result,
the columns are processed in chunks of whole genes. For each chunk only the
SNPs that carry weight in those columns are read, the chunk's n x c float64
block is accumulated, post-processed in place (ct mode) and written as
float32 into its column range of scores.f32. Peak memory is set by
--chunk-gb (the chunk block plus the matmul temporary of the same size),
not by n x m. With genes in genomic order consecutive chunks share SNPs
only at their edges, so the bed is still read about once.
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
    """S (n x m) float64 = X[:, used] @ W[used, :] with mean imputation, in SNP blocks.
    Unchanged from v0; here W is one column chunk and `used` its SNPs."""
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


def chunk_width(n, m, T, chunk_gb, chunk_cols):
    """Columns per chunk: --chunk-cols if given, else what fits --chunk-gb
    counting the float64 chunk block and the matmul temporary (2 x n x c x 8).
    In ct mode a multiple of T so a gene's bins never straddle a chunk."""
    if chunk_cols <= 0:
        chunk_cols = int(chunk_gb * 2**30) // (n * 8 * 2)
    chunk_cols = max(T, (chunk_cols // T) * T)
    return min(chunk_cols, m)


def score_chunked(bed, W, const, colmeta, mode, T, out_dir, chunk_cols, block, threads):
    """Loop over column chunks: read that chunk's SNPs, matmul, finish, write
    float32 into scores.f32[:, c0:c1]. Returns the finished colmeta."""
    os.makedirs(out_dir, exist_ok=True)
    n, m = bed.iid_count, W.shape[1]
    Wc = W.tocsc()                               # column slicing is cheap on csc
    mm = np.memmap(os.path.join(out_dir, "scores.f32"), dtype=np.float32, mode="w+", shape=(n, m))
    out_meta = []
    n_chunks = (m + chunk_cols - 1) // chunk_cols
    n_reads = 0
    for k, c0 in enumerate(range(0, m, chunk_cols)):
        t = time.time()
        c1 = min(m, c0 + chunk_cols)
        Wsub = Wc[:, c0:c1]
        used = np.unique(Wsub.indices)           # sorted: bed reads are sequential
        n_reads += len(used)
        S = matmul_blocks(bed, Wsub.tocsr(), used, block, threads)
        S += const[None, c0:c1]
        meta = colmeta[c0:c1]
        if mode == "ct":
            meta = ct_finish_inplace(S, meta, T)
        out_meta.extend(meta)
        step = max(1, (256 << 20) // ((c1 - c0) * 4))   # ~256 MB of float32 rows at a time
        for r0 in range(0, n, step):
            mm[r0:r0 + step, c0:c1] = S[r0:r0 + step]
        print(f"chunk {k + 1}/{n_chunks}: cols {c0}-{c1} ({(c1 - c0) // T if mode == 'ct' else c1 - c0} "
              f"{'genes' if mode == 'ct' else 'columns'}), {len(used)} SNPs, {time.time() - t:.1f}s", flush=True)
        del S
    mm.flush(); del mm
    return out_meta, n_reads


def write_meta(out_dir, bed, colmeta):
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
    ap.add_argument("--chunk-gb", type=float, default=1.0, help="memory budget per column chunk (chunk block + matmul temporary)")
    ap.add_argument("--chunk-cols", type=int, default=0, help="columns per chunk (overrides --chunk-gb)")
    a = ap.parse_args()
    t0 = time.time()
    bed = open_bed(a.bed + ".bed", num_threads=a.threads)
    fixed = None; T = 1
    if a.mode == "ct":
        thr = a.thresholds.split()
        assert thr and a.all_genes, "--mode ct needs --thresholds and --all-genes"
        genes = []
        for l in open(a.all_genes):
            if l.strip() and l.strip() not in genes:
                genes.append(l.strip())
        fixed = [(g, f"P<={t}") for g in genes for t in thr]          # gene-major, bins adjacent
        T = len(thr)
    W, const, colmeta, used = load_weights(a.weights, bed, fixed)
    n, m = bed.iid_count, W.shape[1]
    cw = chunk_width(n, m, T, a.chunk_gb, a.chunk_cols)
    print(f"chunking: {m} columns in chunks of {cw} ({(m + cw - 1) // cw} chunks; "
          f"chunk block {n * cw * 8 / 2**30:.2f} GiB)", flush=True)
    colmeta, n_reads = score_chunked(bed, W, const, colmeta, a.mode, T, a.out, cw, a.block, a.threads)
    write_meta(a.out, bed, colmeta)
    print(f"sparse_score[{a.mode}]: {n} x {m} -> {a.out}  ({time.time() - t0:.1f}s; "
          f"SNP reads {n_reads} for {len(used)} unique = {n_reads / max(1, len(used)):.2f}x)")


if __name__ == "__main__":
    main()
