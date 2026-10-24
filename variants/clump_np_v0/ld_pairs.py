#!/usr/bin/env python3
"""ld_pairs.py (clump_np_v0) -- every pair of variants within --kb with r^2 >= --r2, once per chromosome.

    python ld_pairs.py --bed <prefix> --out <pairs.npz> [--kb 250] [--r2 0.5] [--block 512] [--threads 4]

12b runs plink --clump once per gene; every gene on a chromosome clumps a
subset of the same window SNPs under the same rule (250 kb, r^2 >= 0.5,
P-ordered greedy), so the giant genes (30 genome-wide, 300k-470k SNPs each,
10-30 min of plink alone) recompute the chromosome's LD over and over. This
pass computes it once: a sparse graph of the pairs that can ever clump
anything, from which clump_genes.py derives any gene's clumps in milliseconds.

r^2 as plink 1.9 computes it for --clump: the squared Pearson correlation of
the two allele-count vectors over the people with BOTH calls present
(pairwise deletion). With missing coded 0, for SNPs x, y over n people:
    Sxy = x.y          (missing contributes 0 either way)
    Sx  = sum(x) - x.miss_y      Sxx = sum(x^2) - x^2.miss_y      (same for y)
    nb  = n - |miss_x| - |miss_y| + |miss_x & miss_y|
    r^2 = (nb*Sxy - Sx*Sy)^2 / ((nb*Sxx - Sx^2) * (nb*Syy - Sy^2))
Every sum is an integer below 2^24 (4 * 380,962 = 1.5e6), so the float32
GEMM x.y is EXACT; the miss corrections are sparse products (missing calls
are rare); r^2 is then formed in float64. Pairs with a zero denominator
(monomorphic in the shared people) have no r^2 and never clump.

Blocks of --block SNPs in position order; each block is compared with itself
(upper triangle) and with the previous blocks that still reach within --kb.
Memory: ~2 x block x n float32 per resident block (512 x 380,962 = 0.8 GB).
Output: npz with i, j (int32 .bim row indices, i < j), r2 (float32), bp, sid, kb, r2min.
"""
import argparse, os, time
import numpy as np
import scipy.sparse as sp

_LUT = np.stack([(np.arange(256, dtype=np.uint8) >> (2 * s)) & 3 for s in range(4)], axis=1)   # bed codes
_DOSE = np.array([2, 0, 1, 0], dtype=np.float32)      # bed code -> allele count; missing (01) -> 0
_MISS = np.array([0, 1, 0, 0], dtype=np.uint8)        # bed code -> missing flag


class Block:
    __slots__ = ("r0", "r1", "X", "M", "s1", "s2", "nm", "bp")

    def __init__(self, r0, r1, X, M, bp):
        self.r0, self.r1, self.X, self.M, self.bp = r0, r1, X, M, bp
        self.s1 = X.sum(axis=1, dtype=np.float64)                 # sum x over non-missing
        self.s2 = (X * X).sum(axis=1, dtype=np.float64)           # sum x^2
        self.nm = np.asarray(M.sum(axis=1)).ravel().astype(np.float64)   # missing count


def load_block(fh, bpr, n, r0, r1, bp):
    fh.seek(3 + r0 * bpr)
    raw = np.fromfile(fh, dtype=np.uint8, count=(r1 - r0) * bpr).reshape(r1 - r0, bpr)
    codes = _LUT[raw].reshape(r1 - r0, -1)[:, :n]              # (k, n) uint8 bed codes
    X = _DOSE[codes]                                            # float32, missing -> 0
    M = sp.csr_matrix(_MISS[codes])                             # sparse missing indicator (k, n)
    return Block(r0, r1, X, M, bp[r0:r1])


def pairs_between(A, B, kb_bp, r2min, same):
    """r^2 >= r2min pairs between blocks A and B (A == B when same: upper triangle only).
    Returns (ia, ib, r2) with .bim row indices."""
    n = A.X.shape[1]
    Sxy = A.X @ B.X.T                                           # exact integer sums in float32
    Sxy = Sxy.astype(np.float64)
    # corrections through the sparse missing indicators
    x_my = np.asarray((B.M @ A.X.T).T)                          # (ka, kb): sum of x over people where y missing
    y_mx = np.asarray(A.M @ B.X.T)                              # (ka, kb): sum of y over people where x missing
    x2_my = np.asarray((B.M @ (A.X * A.X).T).T)
    y2_mx = np.asarray(A.M @ (B.X * B.X).T)
    both = np.asarray((A.M @ B.M.T).todense()) if A.M.nnz and B.M.nnz else np.zeros((A.X.shape[0], B.X.shape[0]))
    nb = n - A.nm[:, None] - B.nm[None, :] + both
    Sx = A.s1[:, None] - x_my; Sy = B.s1[None, :] - y_mx
    Sxx = A.s2[:, None] - x2_my; Syy = B.s2[None, :] - y2_mx
    num = (nb * Sxy - Sx * Sy) ** 2
    den = (nb * Sxx - Sx * Sx) * (nb * Syy - Sy * Sy)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(den > 0, num / den, 0.0)
    near = np.abs(A.bp[:, None].astype(np.int64) - B.bp[None, :].astype(np.int64)) <= kb_bp
    hit = (r2 >= r2min) & near
    if same:
        hit &= np.triu(np.ones_like(hit), k=1)
    ia, ib = np.nonzero(hit)
    return ia + A.r0, ib + B.r0, r2[ia, ib].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bed", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--kb", type=float, default=250); ap.add_argument("--r2", type=float, default=0.5)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("BENCH_THREADS", "4")))
    a = ap.parse_args()
    t0 = time.time()
    bim = [l.split() for l in open(a.bed + ".bim") if l.strip()]
    sid = np.array([f[1] for f in bim]); bp = np.array([int(f[3]) for f in bim], dtype=np.int64)
    if not np.all(np.diff(bp) >= 0):
        raise SystemExit("variants must be in position order (one chromosome); sort the .bim first")
    n = sum(1 for l in open(a.bed + ".fam") if l.strip()); m = len(bim); bpr = (n + 3) // 4
    fh = open(a.bed + ".bed", "rb")
    if fh.read(3) != b"\x6c\x1b\x01" or os.path.getsize(a.bed + ".bed") != 3 + m * bpr:
        raise SystemExit(f"{a.bed}.bed: not a SNP-major bed of {m} x {n}")
    kb_bp = int(round(a.kb * 1000))
    print(f"ld_pairs: {m} variants x {n} people, window {a.kb:g} kb, r2 >= {a.r2:g}, blocks of {a.block}", flush=True)
    I, J, R = [], [], []
    resident = []                                               # previous blocks still within reach
    T = {"read": 0.0, "gemm": 0.0}
    n_pairs_eval = 0
    for k, r0 in enumerate(range(0, m, a.block)):
        r1 = min(m, r0 + a.block)
        t = time.time(); B = load_block(fh, bpr, n, r0, r1, bp); T["read"] += time.time() - t
        t = time.time()
        resident = [A for A in resident if bp[r0] - A.bp[-1] <= kb_bp]   # drop blocks out of reach
        for A in resident:
            ia, ib, r2 = pairs_between(A, B, kb_bp, a.r2, False); n_pairs_eval += A.X.shape[0] * B.X.shape[0]
            I.append(ia); J.append(ib); R.append(r2)
        ia, ib, r2 = pairs_between(B, B, kb_bp, a.r2, True); n_pairs_eval += B.X.shape[0] * (B.X.shape[0] - 1) // 2
        I.append(ia); J.append(ib); R.append(r2)
        resident.append(B)
        T["gemm"] += time.time() - t
        if k % 20 == 0:
            print(f"  block {k + 1}/{(m + a.block - 1) // a.block}: {sum(len(x) for x in I)} pairs so far, "
                  f"{len(resident)} resident, {time.time() - t0:.0f}s", flush=True)
    i = np.concatenate(I).astype(np.int32); j = np.concatenate(J).astype(np.int32); r2 = np.concatenate(R)
    np.savez(a.out, i=i, j=j, r2=r2, bp=bp, sid=sid, kb=a.kb, r2min=a.r2, n=n)
    print(f"ld_pairs: {len(i)} pairs with r2 >= {a.r2:g} within {a.kb:g} kb out of {n_pairs_eval} evaluated "
          f"({len(i) / m:.1f} per variant) -> {a.out}; read {T['read']:.1f}s, gemm {T['gemm']:.1f}s, total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
