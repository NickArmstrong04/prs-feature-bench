#!/usr/bin/env python3
"""dosage_pack.py (dosage_pack_v0) -- dosage_raw_v0 writing the 2-BIT SAMPLE-MAJOR
STORE X2.npy (lib/x2bit.py) instead of the int8 X.npy.

v0 change: each decoded (K x n) block is transposed and packed 4 SNPs per
byte into X2.npy, a uint8 (n, ceil(M/4)) memmap -- a quarter of the bytes of
X.npy, each person's row contiguous, read back by lib/x2bit.unpack (the
trainer's loader). Blocks start on a multiple of 4 columns so each block is a
byte slice. compare_dosage.py unpacks it, so the golden stays int8. Below is
dosage_raw_v0's description.


    python dosage_raw.py --raw <raw prefix> --bed <fixture prefix> --pvalue <SNP.pvalue>
                         --genes-dir <dir> --manifest <gene_manifest.tsv> --out <dir>
                         [--pthr 0.01] [--block-mb 256] [--sub 64] [--max-gap 2]

12a streams a whole raw chromosome (chr1 = 258 GB) through plink to make a
cohort bed of window SNPs; this path needs only the P < pthr SNPs inside gene
windows (~3% of the window SNPs), so it seeks to exactly those rows of the raw
.bed and reads nothing else. The fixture supplies what 12a's flags supplied:
  --keep       the fixture's .fam = the people, in its order (mapped to raw rows)
  --extract    the selection (same rule as dosage_v0/v1, on the fixture's .bim ids)
  --a1-allele  the fixture's .bim A1 = the counted allele; where the raw .bim has
               it as A2 the decoded codes are flipped per SNP (2-x, 3 stays 3)
The output is the dosage layout of README s2 and must equal golden/dosage made
from the cohort bed, cell for cell.

Per sub-block of --sub raw rows: one fromfile per run of near-adjacent rows
(--max-gap), 2-bit -> int8 through the 256 x 4 table for all raw people
(488k columns), column-gather of the kept people, flip of the flipped rows,
into an int8 (K x n) block buffer sized by --block-mb; the block is written
transposed into the memmapped X.npy.
"""
import argparse, os, pickle, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("BENCH_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib")))
from x2bit import pack
from dosage_select import read_bim, read_fam, read_pvalues, select   # shared with dosage_v1

MISSING = 3
_CODE = np.array([2, MISSING, 1, 0], dtype=np.int8)
_LUT = np.stack([_CODE[(np.arange(256) >> (2 * s)) & 3] for s in range(4)], axis=1)   # (256, 4) int8
_FLIP = np.array([2, 1, 0, MISSING], dtype=np.int8)                                  # value -> flipped value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw release plink prefix (one chromosome)")
    ap.add_argument("--bed", required=True, help="plink prefix whose .bim gives the selection ids, positions and A1 (fixture geno, or production ${TARGET}.PH)")
    ap.add_argument("--keep", default=None, help="people as 'FID IID' lines (production KEEP_LIST); default: the --bed .fam")
    ap.add_argument("--a1", default=None, help="'SNP allele' lines overriding the .bim A1 as the counted allele (production target_data.a1)")
    ap.add_argument("--pvalue", required=True); ap.add_argument("--genes-dir", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--pthr", type=float, default=0.01)
    ap.add_argument("--block-mb", type=int, default=256, help="int8 (K x n) block buffer size -> K")
    ap.add_argument("--sub", type=int, default=64, help="raw rows decoded per sub-block")
    ap.add_argument("--max-gap", type=int, default=2, help="raw rows further apart than this start a new read")
    a = ap.parse_args()
    t0 = time.time(); T = {}

    # selection on the fixture's ids, exactly as dosage_v1
    sid, a1, bp = read_bim(a.bed)
    if a.a1:
        over = {}
        with open(a.a1) as fh:
            for l in fh:
                f = l.split()
                if len(f) >= 2:
                    over[f[0]] = f[1]
        a1 = [over.get(s, x) for s, x in zip(sid, a1)]
        print(f"a1: {sum(1 for s in sid if s in over)} of {len(sid)} .bim SNPs take their counted allele from {a.a1}", flush=True)
    fam = read_fam(a.bed) if a.keep is None else [(f[0], f[1]) for f in (l.split() for l in open(a.keep)) if len(f) >= 2]
    pvals = read_pvalues(a.pvalue)
    rows_fix, gene2rows, cnt = select(sid, pvals, a.genes_dir, a.manifest, a.pthr)
    M = len(rows_fix)
    print(f"select: P < {a.pthr:g}: {cnt['sig_in_bim']} of {cnt['p_rows']} SNPs in the fixture .bim; {cnt['selected']} linked to "
          f"{cnt['genes_with_snps']} of {cnt['genes_total']} genes ({cnt['sig_unlinked']} significant SNPs in no gene, dropped)", flush=True)
    if M == 0:
        raise SystemExit("no SNPs selected")

    # raw release: row of each selected id, allele orientation, column of each kept person
    raw_row, raw_al = {}, {}
    want = {sid[int(r)] for r in rows_fix}
    with open(a.raw + ".bim") as fh:
        for i, l in enumerate(fh):
            f = l.split()
            if f[1] in want:
                raw_row[f[1]] = i; raw_al[f[1]] = (f[4], f[5])
        m_raw = i + 1
    raw_fam = read_fam(a.raw)
    n_raw = len(raw_fam)
    missing = [sid[int(r)] for r in rows_fix if sid[int(r)] not in raw_row]
    if missing:
        raise SystemExit(f"{len(missing)} selected SNPs not in the raw .bim, e.g. {missing[:5]}")
    # output columns stay in fixture .bim order; the raw file is read in ITS row order
    # (the same order when 12a kept it, which is the production case) and each
    # decoded row lands in its fixture column
    col_of_fixrow = {int(r): j for j, r in enumerate(rows_fix)}
    rr_fix = np.asarray([raw_row[sid[int(r)]] for r in rows_fix], dtype=np.int64)
    order = np.argsort(rr_fix, kind="stable")           # raw-row order of the selected SNPs
    rrows = rr_fix[order]                                 # raw rows, ascending
    outcol = order                                        # fixture column of each raw-ordered row
    same_order = bool(np.all(np.diff(outcol) == 1))
    # orientation per raw-ordered row. The counted allele (target A1) is matched to
    # the raw alleles directly, else on the other strand (step 03's mismatch.R
    # complements strand-flipped target SNPs; the raw release still holds the
    # original strand): a1 == comp(raw A1) counts raw A1, a1 == comp(raw A2) counts
    # raw A2. Anything else is an error (mismatch.R should have excluded it).
    COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    flip = np.zeros(M, dtype=bool)
    n_strand = 0
    for j, c in enumerate(outcol):
        r = int(rows_fix[c]); s = sid[r]; ra1, ra2 = raw_al[s]; x = a1[r]
        if x == ra1:
            pass
        elif x == ra2:
            flip[j] = True
        elif COMP.get(x) == ra1:
            n_strand += 1
        elif COMP.get(x) == ra2:
            flip[j] = True; n_strand += 1
        else:
            raise SystemExit(f"{s}: counted allele {x} is neither raw allele ({ra1}, {ra2}) nor their complement")
    pos = {p: i for i, p in enumerate(raw_fam)}
    try:
        keep = np.asarray([pos[p] for p in fam], dtype=np.int64)
    except KeyError as e:
        raise SystemExit(f"fixture person {e} not in the raw .fam")
    n = len(keep)
    T["select"] = time.time() - t0
    print(f"raw: {m_raw} variants x {n_raw} people; {M} rows wanted ({int(flip.sum())} flipped to the counted allele, {n_strand} matched on the other strand), "
          f"{n} people kept ({'same order' if bool(np.all(np.diff(keep) > 0)) else 'reordered'}); "
          f"raw SNP order {'= fixture order' if same_order else 'differs from fixture order (columns scattered on write)'}", flush=True)

    bpr = (n_raw + 3) // 4
    path = a.raw + ".bed"
    if os.path.getsize(path) != 3 + m_raw * bpr:
        raise SystemExit(f"{path}: size {os.path.getsize(path)} != 3 + {m_raw} x {bpr}")
    fh = open(path, "rb")
    if fh.read(3) != b"\x6c\x1b\x01":
        raise SystemExit(f"{path}: not a SNP-major PLINK .bed")

    os.makedirs(a.out, exist_ok=True)
    B = (M + 3) // 4
    X2 = np.lib.format.open_memmap(os.path.join(a.out, "X2.npy"), mode="w+", dtype=np.uint8, shape=(n, B))
    K = max(a.sub, min(M, (a.block_mb << 20) // max(1, n)))
    K -= K % 4                                             # blocks start on a byte boundary
    if not same_order:
        raise SystemExit("raw SNP order differs from the fixture order: the packed store needs contiguous column blocks "
                         "(write the int8 layout with dosage_raw_v0 instead)")
    S = min(a.sub, K)
    blk = np.empty((K, n), dtype=np.int8)
    sub8 = np.empty((S, bpr, 4), dtype=np.int8)
    T["read"] = T["decode"] = T["gather"] = T["write"] = 0.0
    n_read = n_runs = 0
    for c0 in range(0, M, K):
        k = min(K, M - c0)
        for s0 in range(0, k, S):
            s1 = min(s0 + S, k); rr = rrows[c0 + s0:c0 + s1]; ks = s1 - s0
            cuts = np.flatnonzero(np.diff(rr) > a.max_gap) + 1
            t = time.time()
            raw = np.empty((ks, bpr), dtype=np.uint8)
            for u0, u1 in zip(np.r_[0, cuts], np.r_[cuts, ks]):
                r0, r1 = int(rr[u0]), int(rr[u1 - 1]) + 1
                fh.seek(3 + r0 * bpr)
                span = np.fromfile(fh, dtype=np.uint8, count=(r1 - r0) * bpr).reshape(r1 - r0, bpr)
                raw[u0:u1] = span[rr[u0:u1] - r0]
                n_read += r1 - r0; n_runs += 1
            T["read"] += time.time() - t; t = time.time()
            np.take(_LUT, raw, axis=0, out=sub8[:ks])
            G = sub8[:ks].reshape(ks, -1)                        # (ks, 4*bpr) int8, all raw people
            T["decode"] += time.time() - t; t = time.time()
            np.take(G, keep, axis=1, out=blk[s0:s1])             # kept people only
            fl = np.flatnonzero(flip[c0 + s0:c0 + s1])
            if len(fl):
                blk[s0 + fl] = _FLIP[blk[s0 + fl]]
            T["gather"] += time.time() - t
        t = time.time()
        X2[:, c0 // 4:(c0 + k + 3) // 4] = pack(blk[:k].T)
        T["write"] += time.time() - t
    t = time.time(); X2.flush(); del X2; T["write"] += time.time() - t

    with open(os.path.join(a.out, "samples.txt"), "w") as w:
        for fid, iid in fam:
            w.write(f"{fid}\t{iid}\n")
    pv = np.asarray([pvals[sid[int(r)]] for r in rows_fix], dtype=np.float64)
    np.save(os.path.join(a.out, "pvalues.npy"), pv)
    with open(os.path.join(a.out, "snps.tsv"), "w") as w:
        w.write("col\tsnp\ta1\tbp\tp\n")
        for j, r in enumerate(rows_fix):
            r = int(r); w.write(f"{j}\t{sid[r]}\t{a1[r]}\t{bp[r]}\t{float(pv[j])!r}\n")
    g2c = {g: sorted(col_of_fixrow[r] for r in rs) for g, rs in gene2rows.items()}
    with open(os.path.join(a.out, "gene2snps.pkl"), "wb") as w:
        pickle.dump(g2c, w)
    with open(os.path.join(a.out, "gene2snps.tsv"), "w") as w:
        w.write("gene\tn_snps\tsnps\n")
        for g in sorted(g2c):
            w.write(f"{g}\t{len(g2c[g])}\t" + ",".join(sid[int(rows_fix[c])] for c in g2c[g]) + "\n")
    print(f"dosage[pack_v0]: {n} x {M} 2-bit ({n * B / 1e6:.0f} MB), {len(g2c)} genes, {sum(len(v) for v in g2c.values())} links; "
          f"raw rows read {n_read} ({n_read / M:.2f}x wanted, {n_read * bpr / 1e6:.0f} MB) in {n_runs} reads", flush=True)
    print("kernel time: " + ", ".join(f"{k} {v:.1f}s" for k, v in T.items()) + f"; total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
