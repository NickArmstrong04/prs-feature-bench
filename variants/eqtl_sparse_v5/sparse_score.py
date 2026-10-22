#!/usr/bin/env python3
"""sparse_score.py (eqtl_sparse_v5) -- eqtl_sparse_v3 with the OUTPUT WRITTEN BY
pwrite INSTEAD OF A MEMMAP (--write pwrite), and the setup phases timed.

    python sparse_score.py --bed <prefix> --weights <long.tsv> --out <dir>
                           [--mode sum|ct] [--thresholds "1e-5 ... 1"] [--block 4000] [--threads 4]
                           [--chunk-gb 1.0 | --chunk-cols N] [--write mmap|pwrite]

v5 change: the kernel is eqtl_sparse_v3's unchanged. The write of the (c x n)
chunk result into the (n x m) float32 scores.f32 went through a numpy memmap;
eqtl_sparse_v4 showed that even a row-contiguous memmap write costs 1.1 s on
synth14 (4 column chunks: each row is four 27 KB segments, each page faulted
and dirtied through the mapping). --write pwrite converts each ~256 MB row
step to a contiguous float32 (step x c) block and issues one os.pwrite per
row at its file offset (one pwrite per step when the chunk spans all
columns); no mapping, the kernel writes to the page cache like plink does.
main() also prints a "setup time" line (weights parse, column permute, chunk
plan) and the kernel's untimed remainder as "other", because 2.1 s of v3's
7.1 s on synth14 fell outside the timed parts. v3's design, kept here:
how the gene's float64 tile is made. v2 decoded the whole SNP block to an
int8 (k x n) matrix, took per-SNP sums and a missing mask from it (three
passes over k x n bytes), then per gene did G[idx].astype(float64), which
allocates a fresh (k_g x n) array every call -- page-faulting 4.8 MB per
gene at n = 20k was the single largest cost in the measured 3.4 s
decode+impute. Here:
  * per-SNP allele totals and missing counts come from the PACKED bytes via
    two 256-entry byte tables (4 codes per byte, so a quarter of the work),
    with the pad codes of the last byte subtracted;
  * each gene decodes its own rows straight from the packed bytes into a
    preallocated int8 buffer, casts into a preallocated float64 buffer, and
    imputes with np.copyto(where=tile < 0); no (k x n) int8 block, no
    missing mask, no per-gene allocation;
  * in sum mode genes are ordered by their first SNP row, so a SNP block's
    row span stays tight whatever order the weights file lists genes in.
Bit-identical to v2 on the synthetic smoke data. In ct mode the fixed
gene-major layout is kept, so columns.tsv is unchanged.
"""
import argparse, os, time
import numpy as np
import scipy.sparse as sp
from bed_reader import open_bed

# 2-bit PLINK codes, low bits = first sample: 00 hom A1 -> 2, 01 missing -> -1, 10 het -> 1, 11 hom A2 -> 0
_CODE = np.array([2, -1, 1, 0], dtype=np.int8)
_LUT = np.stack([_CODE[(np.arange(256) >> (2 * s)) & 3] for s in range(4)], axis=1)   # (256, 4) int8
MISSING = -1
# per packed byte: number of missing codes, and the A1 total of the non-missing codes
_BMISS = (_LUT == MISSING).sum(axis=1).astype(np.int8)                                # (256,)
_BSUM = np.where(_LUT == MISSING, 0, _LUT).sum(axis=1).astype(np.int16)              # (256,)
PAD_CODE_VALUE = 2   # pad bits are 0 = code 00 = hom A1 -> counted as 2 by _BSUM; subtracted


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

    def decode_into(self, raw, buf8):
        """Decode (k, bpr) packed rows into buf8[:k] ((K, bpr, 4) int8, K >= k);
        returns the (k, n) int8 view. No allocation."""
        k = raw.shape[0]
        np.take(_LUT, raw, axis=0, out=buf8[:k])
        return buf8[:k].reshape(k, -1)[:, : self.n]

    def stats(self, raw):
        """Per-SNP (missing count, A1 total of the called genotypes) from the
        packed bytes; a quarter of the work of the same sums on int8 codes."""
        nmiss = _BMISS[raw].sum(axis=1, dtype=np.int64)
        tot = _BSUM[raw].sum(axis=1, dtype=np.int64) - PAD_CODE_VALUE * (4 * self.bpr - self.n)
        return nmiss, tot


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


def gene_major(W, const, colmeta, by_position=False):
    """Permute columns so each gene's columns are adjacent (channels in order
    of first appearance within the gene). Genes in order of first appearance,
    or with by_position by their first weighted SNP row (.bim order), so a
    run of genes reads a tight span of the .bed. No-op when nothing moves
    (ct mode's fixed layout, which is never reordered)."""
    first = {}
    for g, _, _ in colmeta:
        first.setdefault(g, len(first))
    if by_position:
        Wc = W.tocsc()
        lo = {}
        for j, (g, _, _) in enumerate(colmeta):
            s, e = Wc.indptr[j], Wc.indptr[j + 1]
            if e > s:
                lo[g] = min(lo.get(g, np.inf), int(Wc.indices[s:e].min()))
        first = {g: (lo.get(g, np.inf), k) for g, k in first.items()}   # weightless genes last, stable
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
    one read per block and per-SNP means from the packed bytes; then per
    gene decode its rows into the int8 buffer, cast into the float64 buffer
    (imputed in place), and S^T[a:b] = W_g^T @ G_g in one BLAS call. The two
    buffers are sized to the largest gene and reused for every gene."""
    n = pbed.n
    c = Wc.shape[1]
    ST = np.zeros((c, n), dtype=np.float64)
    kmax = max((len(r) for _, _, r in genes), default=0)
    buf8 = np.empty((kmax, pbed.bpr, 4), dtype=np.int8)
    buf64 = np.empty((kmax, n), dtype=np.float64)
    i = 0
    while i < len(genes):
        j, cnt = i, 0                                 # genes i..j-1 form the block
        while j < len(genes) and (j == i or cnt + len(genes[j][2]) <= block):
            cnt += len(genes[j][2]); j += 1
        used = np.unique(np.concatenate([genes[q][2] for q in range(i, j)]))
        t = time.time(); raw = pbed.read_rows(used); times["read"] += time.time() - t
        t = time.time()
        nmiss, tot = pbed.stats(raw)
        cnt_ = n - nmiss
        mean = np.where(cnt_ > 0, tot / np.maximum(cnt_, 1), 0.0)   # all-missing SNP scores as 0
        times["decode+impute"] += time.time() - t
        for q in range(i, j):
            a, b, rows_g = genes[q]
            t = time.time()
            idx = np.searchsorted(used, rows_g)
            k_g = len(idx)
            Gg = buf64[:k_g]                                      # (k_g, n) float64, reused
            np.copyto(Gg, pbed.decode_into(raw[idx], buf8), casting="unsafe")
            if nmiss[idx].any():
                np.copyto(Gg, np.broadcast_to(mean[idx][:, None], Gg.shape), where=Gg < 0)
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


class RowWriter:
    """scores.f32 (n x m float32) written with os.pwrite: put(r0, c0, blk)
    stores blk (rows x cols float32, C-contiguous) at rows r0.., cols c0..;
    one call when the block spans all m columns, else one per row."""
    def __init__(self, path, n, m):
        self.m = m
        flags = os.O_RDWR | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
        self.fd = os.open(path, flags, 0o644)
        os.ftruncate(self.fd, n * m * 4)
        self._pwrite = getattr(os, "pwrite", None)

    def _w(self, buf, off):
        if self._pwrite is not None:
            k = 0
            while k < len(buf):
                k += self._pwrite(self.fd, buf[k:], off + k)
        else:                                    # no pwrite (Windows smoke test): seek + write
            os.lseek(self.fd, off, os.SEEK_SET)
            os.write(self.fd, buf)

    def put(self, r0, c0, blk):
        rows, cols = blk.shape
        mv = memoryview(blk).cast("B")
        if cols == self.m:
            self._w(mv, r0 * self.m * 4)
        else:
            for r in range(rows):
                self._w(mv[r * cols * 4:(r + 1) * cols * 4], ((r0 + r) * self.m + c0) * 4)

    def close(self):
        os.close(self.fd)


def score_chunked(pbed, W, const, colmeta, mode, T, out_dir, chunks, block, write="mmap"):
    """Loop over column chunks (whole genes): per gene GEMM, finish, write
    float32 into scores.f32[:, c0:c1] through a memmap or pwrite (--write).
    Returns the finished colmeta, SNP reads."""
    os.makedirs(out_dir, exist_ok=True)
    n, m = pbed.n, W.shape[1]
    Wc = W.tocsc()                               # column slicing is cheap on csc
    path = os.path.join(out_dir, "scores.f32")
    if write == "mmap":
        mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(n, m))
    else:
        rw = RowWriter(path, n, m)
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
            if write == "mmap":
                mm[r0:r0 + step, c0:c1] = ST[:, r0:r0 + step].T
            else:
                rw.put(r0, c0, np.ascontiguousarray(ST[:, r0:r0 + step].T, dtype=np.float32))
        TIMES["write"] += time.time() - tw
        print(f"chunk {k + 1}/{len(chunks)}: cols {c0}-{c1} ({len(genes)} genes with SNPs), "
              f"{n_reads} SNP rows so far, {time.time() - t:.1f}s", flush=True)
        del ST
    if write == "mmap":
        mm.flush(); del mm
    else:
        rw.close()
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
    ap.add_argument("--write", choices=["mmap", "pwrite"], default="mmap", help="how scores.f32 is written")
    a = ap.parse_args()
    t0 = time.time()
    setup = {}
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
    setup["open"] = time.time() - t0
    t = time.time(); W, const, colmeta, used = load_weights(a.weights, bed, fixed); setup["weights"] = time.time() - t
    t = time.time(); W, const, colmeta = gene_major(W, const, colmeta, by_position=(a.mode == "sum")); setup["permute"] = time.time() - t
    n, m = bed.iid_count, W.shape[1]
    pbed = PackedBed(a.bed + ".bed", n, bed.sid_count)
    t = time.time()
    ranges = gene_ranges(colmeta)
    chunks, cw = plan_chunks(ranges, n, a.chunk_gb, a.chunk_cols)
    setup["plan"] = time.time() - t
    print("setup time: " + ", ".join(f"{k} {v:.1f}s" for k, v in setup.items()), flush=True)
    print(f"chunking: {m} columns, {len(ranges)} genes, {len(chunks)} chunks of <= {cw} columns "
          f"(largest {max((c1 - c0 for c0, c1 in chunks), default=0)} = "
          f"{max((c1 - c0 for c0, c1 in chunks), default=0) * n * 8 / 2**30:.2f} GiB)", flush=True)
    tk = time.time()
    colmeta, n_reads = score_chunked(pbed, W, const, colmeta, a.mode, T, a.out, chunks, a.block, a.write)
    TIMES["other"] = time.time() - tk - sum(TIMES.values())
    write_meta(a.out, bed, colmeta)
    print("kernel time: " + ", ".join(f"{k} {v:.1f}s" for k, v in TIMES.items()))
    print(f"sparse_score[{a.mode}]: {n} x {m} -> {a.out}  ({time.time() - t0:.1f}s; "
          f"SNP reads {n_reads} for {len(used)} unique = {n_reads / max(1, len(used)):.2f}x)")


if __name__ == "__main__":
    main()
