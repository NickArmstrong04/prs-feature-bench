#!/usr/bin/env python3
"""sparse_score.py (ct_sparse_v2) -- ct_sparse_v1 with the PACKED-BED DECODE.

    python sparse_score.py --bed <prefix> --weights <long.tsv> --out <dir>
                           [--mode sum|ct] [--thresholds "1e-5 ... 1"] [--block 4000] [--threads 4]
                           [--chunk-gb 1.0 | --chunk-cols N]

Same contract and arithmetic as v1 (column chunking kept); the one change is
README section 7 item 2: dosages are decoded from the packed .bed bytes
directly instead of through bed-reader's float32 path. The .bed is memmapped
as uint8 (n_snps x ceil(n/4)); a block of sorted SNP rows is gathered and
pushed through a 256 x 4 lookup table into int8 A1 counts (2/1/0, missing
-1 -- bed-reader's count_A1 convention), giving G (k x n). The per-SNP
imputation mean comes from int8 integer sums in the same pass; G is then
cast to float64 once with the means substituted where missing, and the
block product is W_blk^T @ G (what scipy did internally for v1's X @ W, so
the float64 accumulation is the same). bed-reader is still imported, for the
.bim/.fam metadata only. The kernel logs read / decode+impute / cast /
matmul / write time separately.
"""
import argparse, os, time
import numpy as np
import scipy.sparse as sp
from bed_reader import open_bed

# 2-bit PLINK codes, low bits = first sample: 00 hom A1 -> 2, 01 missing -> -1, 10 het -> 1, 11 hom A2 -> 0
_CODE = np.array([2, -1, 1, 0], dtype=np.int8)
_LUT = np.stack([_CODE[(np.arange(256) >> (2 * s)) & 3] for s in range(4)], axis=1)   # (256, 4) int8
MISSING = -1


class PackedBed:
    """The .bed as a uint8 memmap; decode(rows) -> int8 (k x n) A1 counts, missing = MISSING."""
    def __init__(self, path, n_samples, n_snps):
        self.n = n_samples
        bpr = (n_samples + 3) // 4
        with open(path, "rb") as fh:
            magic = fh.read(3)
        if magic != b"\x6c\x1b\x01":
            raise SystemExit(f"{path}: not a SNP-major PLINK .bed (magic {magic.hex()})")
        if os.path.getsize(path) != 3 + n_snps * bpr:
            raise SystemExit(f"{path}: size {os.path.getsize(path)} != 3 + {n_snps} x {bpr}")
        self.mm = np.memmap(path, dtype=np.uint8, mode="r", offset=3, shape=(n_snps, bpr))

    def read_rows(self, rows):
        return np.asarray(self.mm[rows])                        # (k, bpr) uint8, one copy

    def decode(self, raw):
        k = raw.shape[0]
        return _LUT[raw].reshape(k, -1)[:, : self.n]           # (k, n) int8


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


TIMES = {"read": 0.0, "decode+impute": 0.0, "cast": 0.0, "matmul": 0.0, "write": 0.0}


def matmul_blocks(pbed, W, used, block, times=TIMES):
    """S (n x m) float64 = X[:, used] @ W[used, :] with mean imputation, in SNP blocks.
    W is one column chunk (csr, n_bim x c) and `used` its sorted SNP rows.
    Per block: gather packed rows, decode to int8 G (k x n), impute from the
    int8 sums, cast to float64, S += (W_blk^T @ G)^T."""
    n = pbed.n
    S = np.zeros((n, W.shape[1]), dtype=np.float64)
    for b0 in range(0, len(used), block):
        blk = used[b0: b0 + block]
        t = time.time(); raw = pbed.read_rows(blk); times["read"] += time.time() - t
        t = time.time()
        G = pbed.decode(raw)                                      # (k, n) int8
        miss = G == MISSING
        nmiss = miss.sum(axis=1)
        any_missing = bool(nmiss.any())
        if any_missing:
            cnt = n - nmiss
            tot = G.sum(axis=1, dtype=np.int64) + nmiss            # undo the -1 of each missing call
            mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)   # all-missing SNP scores as 0
        times["decode+impute"] += time.time() - t
        t = time.time()
        G64 = G.astype(np.float64)
        if any_missing:
            jj, ii = np.nonzero(miss); G64[jj, ii] = mean[jj]
        times["cast"] += time.time() - t
        t = time.time()
        WT = W[blk, :].T.tocsr()                                  # (c, k)
        S += (WT @ G64).T
        times["matmul"] += time.time() - t
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


def score_chunked(pbed, W, const, colmeta, mode, T, out_dir, chunk_cols, block):
    """Loop over column chunks: read that chunk's SNPs, matmul, finish, write
    float32 into scores.f32[:, c0:c1]. Returns the finished colmeta."""
    os.makedirs(out_dir, exist_ok=True)
    n, m = pbed.n, W.shape[1]
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
        S = matmul_blocks(pbed, Wsub.tocsr(), used, block)
        S += const[None, c0:c1]
        meta = colmeta[c0:c1]
        if mode == "ct":
            meta = ct_finish_inplace(S, meta, T)
        out_meta.extend(meta)
        tw = time.time()
        step = max(1, (256 << 20) // ((c1 - c0) * 4))   # ~256 MB of float32 rows at a time
        for r0 in range(0, n, step):
            mm[r0:r0 + step, c0:c1] = S[r0:r0 + step]
        TIMES["write"] += time.time() - tw
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
    pbed = PackedBed(a.bed + ".bed", n, bed.sid_count)
    cw = chunk_width(n, m, T, a.chunk_gb, a.chunk_cols)
    print(f"chunking: {m} columns in chunks of {cw} ({(m + cw - 1) // cw} chunks; "
          f"chunk block {n * cw * 8 / 2**30:.2f} GiB)", flush=True)
    colmeta, n_reads = score_chunked(pbed, W, const, colmeta, a.mode, T, a.out, cw, a.block)
    write_meta(a.out, bed, colmeta)
    print("kernel time: " + ", ".join(f"{k} {v:.1f}s" for k, v in TIMES.items()))
    print(f"sparse_score[{a.mode}]: {n} x {m} -> {a.out}  ({time.time() - t0:.1f}s; "
          f"SNP reads {n_reads} for {len(used)} unique = {n_reads / max(1, len(used)):.2f}x)")


if __name__ == "__main__":
    main()
