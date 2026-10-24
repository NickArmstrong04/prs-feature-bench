#!/usr/bin/env python3
"""sparse_score.py (ct_sparse_v5) -- ct_sparse_v3 writing 12c's CHUNK LAYOUT.

v5 change (ct mode): --out-layout chunks writes each column chunk as one
contiguous file, exactly what production's `12_prsnet.py score` writes and 12d
assembles -- scores_chunk<k>.npy (n, G_k, T) float32, genes_chunk<k>.txt,
scores_manifest.tsv (chunk file n_genes n_index_snps), gene_counts.tsv (gene
n_index n_P<=t ...), samples.txt. v3 wrote the (n, m) scores.f32 memmap one
column chunk at a time; at 380,962 people that is 1.4 KB into every row of a
25 GB file per chunk, 48 times over (chr19: write 822 s of a 1016 s run, the
kernel itself 141 s). One np.save per chunk writes the 25 GB once,
sequentially. compare.py reads either layout. Below is v3's description.

sparse_score.py (ct_sparse_v3) -- ct_sparse_v2 with a ROW-STREAMED SCATTER-ADD.

    python sparse_score.py --bed <prefix> --weights <long.tsv> --out <dir>
                           [--mode sum|ct] [--thresholds "1e-5 ... 1"] [--block 4000] [--threads 4]
                           [--chunk-gb 1.0 | --chunk-cols N]

Same contract as v2 (column chunking, packed-bed decode kept); the change is
the block product. v2 cast the whole int8 block to float64 (k x n) and
called scipy's sparse @ dense, which builds a (c x n) temporary -- 9.3 s of
its 16.6 s kernel. v3 keeps the chunk result transposed, S^T (c x n), and
streams the SNPs: one int8 row is cast to a cache-resident float64 vector
(imputed on the fly), and for each of that SNP's nonzero weights
S^T[col] += w * g, a contiguous axpy of length n. No float64 block, no
temporary. ct_finish and the memmap write work on the (m x n) layout.
Folded in: the block's row span is read with np.fromfile instead of a
memmap, so file-backed pages no longer inflate RSS (v2: +1 GB). The kernel
logs read / decode+impute / scatter / write time separately.
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


TIMES = {"read": 0.0, "decode+impute": 0.0, "scatter": 0.0, "write": 0.0}


def scatter_blocks(pbed, W, used, block, times=TIMES):
    """S^T (c x n) float64 = (X[:, used] @ W[used, :])^T with mean imputation,
    in SNP blocks. W is one column chunk (csr, n_bim x c), `used` its sorted
    SNP rows. Per block: read + decode to int8 G (k x n), per-SNP means from
    the int8 sums; then per SNP, cast its row to a float64 vector (imputed
    where missing) and axpy it into each weighted column's row of S^T."""
    n = pbed.n
    ST = np.zeros((W.shape[1], n), dtype=np.float64)
    g = np.empty(n, dtype=np.float64); tmp = np.empty(n, dtype=np.float64)
    for b0 in range(0, len(used), block):
        blk = used[b0: b0 + block]
        t = time.time(); raw = pbed.read_rows(blk); times["read"] += time.time() - t
        t = time.time()
        G = pbed.decode(raw)                                      # (k, n) int8
        miss = G == MISSING
        nmiss = miss.sum(axis=1)
        cnt = n - nmiss
        tot = G.sum(axis=1, dtype=np.int64) + nmiss                # undo the -1 of each missing call
        mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)   # all-missing SNP scores as 0
        Wb = W[blk, :]                                            # csr (k, c): row j = SNP blk[j]'s weights
        indptr, indices, data = Wb.indptr, Wb.indices, Wb.data
        times["decode+impute"] += time.time() - t
        t = time.time()
        for j in range(len(blk)):
            s, e = indptr[j], indptr[j + 1]
            if s == e:
                continue
            np.copyto(g, G[j], casting="unsafe")                  # int8 -> float64, in the same buffer
            if nmiss[j]:
                g[miss[j]] = mean[j]
            for col, w in zip(indices[s:e], data[s:e]):
                np.multiply(g, w, out=tmp)
                ST[col] += tmp
        times["scatter"] += time.time() - t
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


def chunk_width(n, m, T, chunk_gb, chunk_cols):
    """Columns per chunk: --chunk-cols if given, else what fits --chunk-gb
    counting the float64 chunk block and the matmul temporary (2 x n x c x 8).
    In ct mode a multiple of T so a gene's bins never straddle a chunk."""
    if chunk_cols <= 0:
        chunk_cols = int(chunk_gb * 2**30) // (n * 8 * 2)
    chunk_cols = max(T, (chunk_cols // T) * T)
    return min(chunk_cols, m)


def score_chunked(pbed, W, const, colmeta, mode, T, out_dir, chunk_cols, block, layout="flat", fixed_layout=False):
    """Loop over column chunks: read that chunk's SNPs, matmul, finish, write
    float32 into scores.f32[:, c0:c1] (flat) or scores_chunk<k>.npy (chunks,
    12c's layout, ct mode only). Returns the finished colmeta."""
    os.makedirs(out_dir, exist_ok=True)
    n, m = pbed.n, W.shape[1]
    Wc = W.tocsc()                               # column slicing is cheap on csc
    if layout == "chunks":
        assert mode == "ct" or fixed_layout, "chunks layout in sum mode needs the fixed layout (--all-genes + --channels)"
        manifest = open(os.path.join(out_dir, "scores_manifest.tsv"), "w")
        manifest.write("chunk\tfile\tn_genes\tn_index_snps\n")
    else:
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
        ST = scatter_blocks(pbed, Wsub.tocsr(), used, block)     # (c, n)
        ST += const[c0:c1, None]
        meta = colmeta[c0:c1]
        if mode == "ct":
            meta = ct_finish_inplace(ST, meta, T)
        out_meta.extend(meta)
        tw = time.time()
        if layout == "chunks":
            genes = [g for g, _, _ in meta[::T]]
            fn = f"scores_chunk{k:03d}.npy"
            np.save(os.path.join(out_dir, fn), ST.T.astype(np.float32).reshape(n, len(genes), T))
            with open(os.path.join(out_dir, f"genes_chunk{k:03d}.txt"), "w") as gw:
                gw.write("".join(g + "\n" for g in genes))
            manifest.write(f"{k}\t{fn}\t{len(genes)}\t{len(used)}\n"); manifest.flush()
        else:
            step = max(1, (256 << 20) // ((c1 - c0) * 4))   # ~256 MB of float32 rows at a time
            for r0 in range(0, n, step):
                mm[r0:r0 + step, c0:c1] = ST[:, r0:r0 + step].T
        TIMES["write"] += time.time() - tw
        print(f"chunk {k + 1}/{n_chunks}: cols {c0}-{c1} ({(c1 - c0) // T if mode == 'ct' else c1 - c0} "
              f"{'genes' if mode == 'ct' else 'columns'}), {len(used)} SNPs, {time.time() - t:.1f}s", flush=True)
        del ST
    if layout == "chunks":
        manifest.close()
    else:
        mm.flush(); del mm
    return out_meta, n_reads


def write_meta(out_dir, bed, colmeta, layout="flat", T=1):
    with open(os.path.join(out_dir, "samples.txt"), "w") as w:
        for f, i in zip(bed.fid, bed.iid):
            w.write(f"{f}\t{i}\n")
    if layout == "chunks":                       # 12c's gene_counts.tsv: cumulative index-SNP counts per bin (ct) / SNPs per channel (sum)
        with open(os.path.join(out_dir, "gene_counts.tsv"), "w") as w:
            w.write("gene\tn_index\t" + "\t".join("n_" + c for _, c, _ in colmeta[:T]) + "\n")
            for j in range(0, len(colmeta), T):
                cnts = [k for _, _, k in colmeta[j:j + T]]
                w.write(f"{colmeta[j][0]}\t{cnts[-1]}\t" + "\t".join(str(k) for k in cnts) + "\n")
        return
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
    ap.add_argument("--out-layout", choices=["flat", "chunks"], default="flat", help="chunks = 12c's per-chunk .npy layout")
    ap.add_argument("--channels", default="", help="sum mode: space-separated channel labels; with --all-genes fixes a gene-major "
                                                     "(gene x channel) column layout so every gene gets columns and chunks can be written")
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
    elif a.all_genes and a.channels:                                   # ldpred2 rows: fixed gene x channel layout, sums
        chans = a.channels.split()
        genes = []
        for l in open(a.all_genes):
            if l.strip() and l.strip() not in genes:
                genes.append(l.strip())
        fixed = [(g, c) for g in genes for c in chans]
        T = len(chans)
    W, const, colmeta, used = load_weights(a.weights, bed, fixed)
    n, m = bed.iid_count, W.shape[1]
    pbed = PackedBed(a.bed + ".bed", n, bed.sid_count)
    cw = chunk_width(n, m, T, a.chunk_gb, a.chunk_cols)
    print(f"chunking: {m} columns in chunks of {cw} ({(m + cw - 1) // cw} chunks; "
          f"chunk block {n * cw * 8 / 2**30:.2f} GiB)", flush=True)
    colmeta, n_reads = score_chunked(pbed, W, const, colmeta, a.mode, T, a.out, cw, a.block, a.out_layout, fixed is not None)
    write_meta(a.out, bed, colmeta, a.out_layout, T)
    print("kernel time: " + ", ".join(f"{k} {v:.1f}s" for k, v in TIMES.items()))
    print(f"sparse_score[{a.mode}]: {n} x {m} -> {a.out}  ({time.time() - t0:.1f}s; "
          f"SNP reads {n_reads} for {len(used)} unique = {n_reads / max(1, len(used)):.2f}x)")


if __name__ == "__main__":
    main()
