#!/usr/bin/env python3
"""sparse_score.py (eqtl_sparse_v2) -- the v3 kernel with ONE GEMM PER GENE.

    python sparse_score.py --bed <prefix> --weights <long.tsv> --out <dir>
                           [--mode sum|ct] [--thresholds "1e-5 ... 1"] [--block 4000] [--threads 4]
                           [--chunk-gb 1.0 | --chunk-cols N]

Same contract as ct_sparse_v3 / eqtl_sparse_v1 (column chunking, packed-bed
decode, np.fromfile reads); the change is the block product. The v3 kernel
did one axpy per nonzero weight, so with C cell types every SNP's dosage
vector was re-read C times from a Python loop (7.9 s of eqtl_sparse_v1's
19.3 s on synth14). An eQTL weight matrix is block-dense per gene -- a
(k_g SNPs x C channels) tile -- so here each gene is one BLAS call:
S^T[gene cols] = W_g^T (C x k_g) @ G_g (k_g x n), with G_g the gene's
dosage rows cast to float64 (imputed) just before the call. Columns are
grouped by gene (in sum mode they are permuted into gene-major order; the
output order is free under the contract), chunks end on gene boundaries,
and a SNP block is a run of whole genes. The GEMM uses the BLAS threads
(OMP/OPENBLAS_NUM_THREADS from the harness), the first part of this kernel
to use --threads at all. In ct mode a gene's tile is (k_g x T) with one
nonzero per row; the GEMM does T x the needed flops, still cheap.
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
    """The .bed read with np.fromfile; decode(rows) -> int8 (k x n) A1 counts, missing = MISSING."""
    def __init__(self, path, n_samples, n_snps):
        self.n = n_samples
        self.bpr = bpr = (n_samples + 3) // 4
        self.fh = open(path, "rb")
        magic = self.fh.read(3)
        if magic != b"\x6c\x1b\x01":
            raise SystemExit(f"{path}: not a SNP-major PLINK .bed (magic {magic.hex()})")
        if os.path.getsize(path) != 3 + n_snps * bpr:
            raise SystemExit(f"{path}: size {os.path.getsize(path)} != 3 + {n_snps} x {bpr}")

    def read_rows(self, rows):
        """Reads the span [min(rows), max(rows)] in one call (used SNPs are
        ~30% of the window, so the over-read is small and sequential) and
        picks the rows out; no memmap, so nothing file-backed stays in RSS."""
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        self.fh.seek(3 + r0 * self.bpr)
        span = np.fromfile(self.fh, dtype=np.uint8, count=(r1 - r0) * self.bpr).reshape(r1 - r0, self.bpr)
        return span[rows - r0]                                  # (k, bpr) uint8

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


def gene_major(W, const, colmeta):
    """Permute columns so each gene's columns are adjacent (genes in order of
    first appearance, channels in order of first appearance within the gene).
    No-op when already grouped (ct mode's fixed layout)."""
    first = {}
    for g, _, _ in colmeta:
        first.setdefault(g, len(first))
    order = sorted(range(len(colmeta)), key=lambda j: (first[colmeta[j][0]], j))
    if order == list(range(len(colmeta))):
        return W, const, colmeta
    order = np.asarray(order)
    return W[:, order], const[order], [colmeta[j] for j in order]


def gene_ranges(colmeta):
    """[(a, b)] column ranges of consecutive same-gene columns (gene-major input)."""
    out, a = [], 0
    for j in range(1, len(colmeta) + 1):
        if j == len(colmeta) or colmeta[j][0] != colmeta[a][0]:
            out.append((a, j)); a = j
    return out


def plan_chunks(ranges, n, chunk_gb, chunk_cols):
    """[(c0, c1)] chunks of whole genes: --chunk-cols columns if given, else
    what fits --chunk-gb for the float64 chunk block (n x c x 8; no matmul
    temporary in this kernel). At least one gene per chunk."""
    if chunk_cols <= 0:
        chunk_cols = max(1, int(chunk_gb * 2**30) // (n * 8))
    chunks, c0 = [], 0
    for i, (a, b) in enumerate(ranges):
        if b - c0 > chunk_cols and a > c0:
            chunks.append((c0, a)); c0 = a
    if ranges:
        chunks.append((c0, ranges[-1][1]))
    return chunks, chunk_cols


TIMES = {"read": 0.0, "decode+impute": 0.0, "gemm": 0.0, "write": 0.0}


def gemm_blocks(pbed, Wc, genes, block, times=TIMES):
    """S^T (c x n) float64 for one chunk. Wc: csc (n_bim x c) of the chunk;
    genes: [(a, b, rows_g)] column range within the chunk and the gene's
    sorted SNP rows. SNP blocks are runs of whole genes with ~block SNPs:
    read + decode once per block, then per gene cast its rows to float64
    (imputed), and S^T[a:b] = W_g^T @ G_g in one BLAS call."""
    n = pbed.n
    c = Wc.shape[1]
    ST = np.zeros((c, n), dtype=np.float64)
    i = 0
    while i < len(genes):
        j, cnt = i, 0                                 # genes i..j-1 form the block
        while j < len(genes) and (j == i or cnt + len(genes[j][2]) <= block):
            cnt += len(genes[j][2]); j += 1
        used = np.unique(np.concatenate([genes[q][2] for q in range(i, j)]))
        t = time.time(); raw = pbed.read_rows(used); times["read"] += time.time() - t
        t = time.time()
        G = pbed.decode(raw)                                      # (k, n) int8
        miss = G == MISSING
        nmiss = miss.sum(axis=1)
        cnt_ = n - nmiss
        tot = G.sum(axis=1, dtype=np.int64) + nmiss                # undo the -1 of each missing call
        mean = np.where(cnt_ > 0, tot / np.maximum(cnt_, 1), 0.0)   # all-missing SNP scores as 0
        times["decode+impute"] += time.time() - t
        for q in range(i, j):
            a, b, rows_g = genes[q]
            t = time.time()
            idx = np.searchsorted(used, rows_g)
            Gg = G[idx].astype(np.float64)                        # (k_g, n)
            if nmiss[idx].any():
                mg = miss[idx]
                Gg[mg] = np.broadcast_to(mean[idx][:, None], Gg.shape)[mg]
            times["decode+impute"] += time.time() - t
            t = time.time()
            Wg = Wc[:, a:b][rows_g, :].toarray()                  # (k_g, C_g) dense tile
            np.matmul(Wg.T, Gg, out=ST[a:b])                      # (C_g, n), BLAS
            times["gemm"] += time.time() - t
        i = j
    return ST


def ct_finish_inplace(ST, colmeta, T):
    """ST is (G*T, n) gene-major with the T bins of each gene adjacent (the
    fixed layout, transposed). Nested-threshold averages = cumsum over bins /
    (2 * cumulative index-SNP count), 0 where the count is 0. Done IN PLACE.
    Returns colmeta with n_snps replaced by the cumulative count."""
    m, n = ST.shape; G = m // T
    V = ST.reshape(G, T, n)                      # a view, no copy
    np.cumsum(V, axis=1, out=V)
    cnt = np.array([k for _, _, k in colmeta], dtype=np.int64).reshape(G, T)
    cc = np.cumsum(cnt, axis=1)
    inv = np.where(cc > 0, 1.0 / (2.0 * np.maximum(cc, 1)), 0.0)   # (G, T)
    V *= inv[:, :, None]
    return [[g, c, int(cc[j // T, j % T])] for j, (g, c, _) in enumerate(colmeta)]


def score_chunked(pbed, W, const, colmeta, mode, T, out_dir, chunks, block):
    """Loop over column chunks (whole genes): per gene GEMM, finish, write
    float32 into scores.f32[:, c0:c1]. Returns the finished colmeta, SNP reads."""
    os.makedirs(out_dir, exist_ok=True)
    n, m = pbed.n, W.shape[1]
    Wc = W.tocsc()                               # column slicing is cheap on csc
    mm = np.memmap(os.path.join(out_dir, "scores.f32"), dtype=np.float32, mode="w+", shape=(n, m))
    out_meta = []
    n_reads = 0
    for k, (c0, c1) in enumerate(chunks):
        t = time.time()
        Wsub = Wc[:, c0:c1]
        genes = []
        for a, b in gene_ranges(colmeta[c0:c1]):
            rows_g = np.unique(Wsub.indices[Wsub.indptr[a]:Wsub.indptr[b]])
            if len(rows_g):
                genes.append((a, b, rows_g))      # genes with no weighted SNP stay zero
        n_reads += sum(len(r) for _, _, r in genes)
        ST = gemm_blocks(pbed, Wsub, genes, block)                # (c, n)
        ST += const[c0:c1, None]
        meta = colmeta[c0:c1]
        if mode == "ct":
            meta = ct_finish_inplace(ST, meta, T)
        out_meta.extend(meta)
        tw = time.time()
        step = max(1, (256 << 20) // ((c1 - c0) * 4))   # ~256 MB of float32 rows at a time
        for r0 in range(0, n, step):
            mm[r0:r0 + step, c0:c1] = ST[:, r0:r0 + step].T
        TIMES["write"] += time.time() - tw
        print(f"chunk {k + 1}/{len(chunks)}: cols {c0}-{c1} ({len(genes)} genes with SNPs), "
              f"{n_reads} SNP rows so far, {time.time() - t:.1f}s", flush=True)
        del ST
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
    ap.add_argument("--chunk-gb", type=float, default=1.0, help="memory budget per column chunk (the float64 chunk block)")
    ap.add_argument("--chunk-cols", type=int, default=0, help="columns per chunk (overrides --chunk-gb; rounded to whole genes)")
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
    W, const, colmeta = gene_major(W, const, colmeta)
    n, m = bed.iid_count, W.shape[1]
    pbed = PackedBed(a.bed + ".bed", n, bed.sid_count)
    ranges = gene_ranges(colmeta)
    chunks, cw = plan_chunks(ranges, n, a.chunk_gb, a.chunk_cols)
    print(f"chunking: {m} columns, {len(ranges)} genes, {len(chunks)} chunks of <= {cw} columns "
          f"(largest {max((c1 - c0 for c0, c1 in chunks), default=0)} = "
          f"{max((c1 - c0 for c0, c1 in chunks), default=0) * n * 8 / 2**30:.2f} GiB)", flush=True)
    colmeta, n_reads = score_chunked(pbed, W, const, colmeta, a.mode, T, a.out, chunks, a.block)
    write_meta(a.out, bed, colmeta)
    print("kernel time: " + ", ".join(f"{k} {v:.1f}s" for k, v in TIMES.items()))
    print(f"sparse_score[{a.mode}]: {n} x {m} -> {a.out}  ({time.time() - t0:.1f}s; "
          f"SNP reads {n_reads} for {len(used)} unique = {n_reads / max(1, len(used)):.2f}x)")


if __name__ == "__main__":
    main()
