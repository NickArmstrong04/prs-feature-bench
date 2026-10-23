#!/usr/bin/env python3
"""bed_seek.py (bed_seek_v1) -- bed_seek_v0 with a FUSED GATHER+PACK KERNEL.

v1 changes (same rules, same files, same reads as v0):
  * decode through a 256-entry uint32 table (one take, viewed back as 4 bytes)
    instead of a (256, 4) uint8 table taken row-wise;
  * the kept people are gathered in four strided groups keep[s::4], each
    shifted into its 2-bit slot and OR-ed straight into the output bytes --
    no (k, n) code matrix, no separate pack pass;
  * allele swaps are applied to the PACKED bytes of the swapped rows through
    a 256-entry table (00 <-> 11 in each field), padding bits re-cleared.
Below is v0's description.

bed_seek.py -- 12a's cohort .bed without plink: seek the window
SNPs out of the raw release, keep the cohort, orient A1, write a PLINK .bed.

    python bed_seek.py --raw <raw prefix> --extract <snplist> --keep <FID IID file>
                       [--a1 <SNP allele file> | --a1-from-bim <prefix>] --out <prefix>
                       [--block 256] [--max-gap 2] [--threads 4] [--unassignable minor|complement|error]

Replicates   plink --bfile RAW --keep KEEP --extract EXTRACT --a1-allele A1 --make-bed
byte for byte, which is what 12a ran per chromosome (4 h 10 min of plink
streaming the 3.65 TB release). Rules copied from plink 1.9:
  * variants come out in the raw .bim order, people in the raw .fam order
    (--extract / --keep are sets, not orders);
  * --a1-allele: allele == raw A1 -> unchanged; == raw A2 -> alleles swapped in
    the .bim and homozygous codes swapped (00 <-> 11; 10 het and 01 missing stay);
  * an allele that is neither ("Impossible A1 allele assignment" in plink's log)
    leaves plink's default in force, which for 1.9 is A1 = the MINOR allele among
    the kept people (A1 frequency > 0.5 -> swap; exactly 0.5 -> keep). That is
    --unassignable minor (default, = production 12a). --unassignable complement
    matches through the strand complement instead (what step 03 intended;
    dosage_pack_v0 does this); error stops.
  * .fam lines are the raw ones; .bim lines are the raw ones with A1/A2 swapped
    where the genotype codes were.

Per block of --block consecutive wanted rows: one np.fromfile per run of raw
rows no further apart than --max-gap, byte -> 4 two-bit codes through a
256 x 4 table (all raw people), column gather of the kept people, code swap on
the flipped rows, repack 4 people per byte, one sequential write at the row's
offset. --threads workers take contiguous row ranges and write to their own
offsets of the pre-sized .bed (fork; sequential where fork is unavailable).
"""
import argparse, os, sys, time
import numpy as np


_LUT = np.stack([(np.arange(256, dtype=np.uint8) >> (2 * s)) & 3 for s in range(4)], axis=1)   # (256, 4) raw bed codes
_LUT32 = np.ascontiguousarray(_LUT).view(np.uint32).reshape(256)                              # the same 4 bytes as one word
_SWAP = np.array([3, 1, 2, 0], dtype=np.uint8)     # hom A1 <-> hom A2; missing (1) and het (2) stay
_SWAP_BYTE = np.zeros(256, dtype=np.uint8)         # the swap applied to all four fields of a packed byte
for _b in range(256):
    _SWAP_BYTE[_b] = sum(int(_SWAP[(_b >> (2 * s)) & 3]) << (2 * s) for s in range(4))
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
MAGIC = b"\x6c\x1b\x01"


def read_lines(path):
    return [l.rstrip("\n") for l in open(path) if l.strip()]


def plan(a):
    """Everything that is not genotype bytes: rows wanted, people kept, orientation per row."""
    want = set(l.split()[0] for l in read_lines(a.extract))
    bim = [l.split() for l in read_lines(a.raw + ".bim")]
    rows = [i for i, f in enumerate(bim) if f[1] in want]
    found = {bim[i][1] for i in rows}
    miss = want - found
    if miss:
        raise SystemExit(f"{len(miss)} --extract ids not in {a.raw}.bim, e.g. {sorted(miss)[:5]}")
    dup = len(rows) - len(found)
    if dup:
        raise SystemExit(f"{dup} duplicated ids among the wanted rows of {a.raw}.bim")
    fam = [l.split() for l in read_lines(a.raw + ".fam")]
    keep_ids = set()
    for l in read_lines(a.keep):
        f = l.split()
        if len(f) >= 2:
            keep_ids.add((f[0], f[1]))
    keep = np.asarray([i for i, f in enumerate(fam) if (f[0], f[1]) in keep_ids], dtype=np.int64)
    if len(keep) != len(keep_ids):
        raise SystemExit(f"{len(keep_ids) - len(keep)} --keep people are not in {a.raw}.fam")
    # counted allele per wanted row
    a1 = {}
    if a.a1:
        for l in read_lines(a.a1):
            f = l.split()
            if len(f) >= 2:
                a1[f[0]] = f[1]
    elif a.a1_from_bim:
        for l in read_lines(a.a1_from_bim + ".bim"):
            f = l.split(); a1[f[1]] = f[4]
    # state per row: 0 keep, 1 swap, 2 undecided (minor rule, needs the genotypes)
    state = np.zeros(len(rows), dtype=np.uint8)
    n_set = n_swap = n_strand = n_undec = 0
    for j, r in enumerate(rows):
        f = bim[r]; s = f[1]; ra1, ra2 = f[4], f[5]
        x = a1.get(s)
        if x is None:
            state[j] = 2; n_undec += 1; continue
        n_set += 1
        if x == ra1:
            pass
        elif x == ra2:
            state[j] = 1; n_swap += 1
        elif a.unassignable == "complement" and COMP.get(x) == ra1:
            n_strand += 1
        elif a.unassignable == "complement" and COMP.get(x) == ra2:
            state[j] = 1; n_swap += 1; n_strand += 1
        elif a.unassignable == "error":
            raise SystemExit(f"{s}: counted allele {x} is neither raw allele ({ra1}, {ra2})")
        else:
            state[j] = 2; n_undec += 1          # plink: impossible assignment -> default (minor) rule
    return bim, fam, np.asarray(rows, dtype=np.int64), keep, state, (n_set, n_swap, n_strand, n_undec)


def work(args):
    """Rows [j0, j1) of the wanted list: read, gather, orient, write. Returns (timings, swapped-by-minor-rule, reads, rows read)."""
    a, rows, keep, state, j0, j1, n_raw = args
    bpr_in = (n_raw + 3) // 4; n = len(keep); bpr_out = (n + 3) // 4
    T = {"read": 0.0, "decode": 0.0, "gather": 0.0, "pack": 0.0, "write": 0.0}
    fin = open(a.raw + ".bed", "rb"); fout = open(a.out + ".bed", "r+b")
    minor_swapped = []
    n_reads = n_rows_read = 0
    K = a.block
    dec32 = np.empty((K, bpr_in), dtype=np.uint32)
    groups = [keep[s::4] for s in range(4)]                  # output slot s takes every 4th kept person
    out = np.empty((K, bpr_out), dtype=np.uint8)
    tmps = [np.empty((K, len(g)), dtype=np.uint8) for g in groups]   # contiguous, exact widths
    pad_mask = np.uint8((1 << (2 * (n % 4))) - 1) if n % 4 else None
    for b0 in range(j0, j1, K):
        b1 = min(b0 + K, j1); k = b1 - b0; rr = rows[b0:b1]
        t = time.time()
        raw = np.empty((k, bpr_in), dtype=np.uint8)
        cuts = np.flatnonzero(np.diff(rr) > a.max_gap) + 1
        for u0, u1 in zip(np.r_[0, cuts], np.r_[cuts, k]):
            r0, r1 = int(rr[u0]), int(rr[u1 - 1]) + 1
            fin.seek(3 + r0 * bpr_in)
            span = np.fromfile(fin, dtype=np.uint8, count=(r1 - r0) * bpr_in).reshape(r1 - r0, bpr_in)
            raw[u0:u1] = span[rr[u0:u1] - r0]
            n_reads += 1; n_rows_read += r1 - r0
        T["read"] += time.time() - t; t = time.time()
        np.take(_LUT32, raw, out=dec32[:k])              # one word per raw byte = its 4 codes
        G = dec32[:k].view(np.uint8)                     # (k, 4*bpr_in) codes of all raw people
        T["decode"] += time.time() - t; t = time.time()
        st = state[b0:b1]
        und = np.flatnonzero(st == 2)
        if len(und):                                     # plink's default: A1 = minor allele among kept people
            sub = np.take(G[und], keep, axis=1)
            n_hom1 = np.count_nonzero(sub == 0, axis=1); n_het = np.count_nonzero(sub == 2, axis=1)
            n_hom2 = np.count_nonzero(sub == 3, axis=1)
            cnt1 = 2 * n_hom1 + n_het; tot = 2 * (n_hom1 + n_het + n_hom2)
            sw = und[cnt1 * 2 > tot]                     # A1 frequency > 0.5 -> A1 becomes the other allele
            minor_swapped += [b0 + int(i) for i in sw]
            st = st.copy(); st[und] = 0; st[sw] = 1
        # fused gather + pack: slot s of each output byte is kept person 4*byte + s
        o = out[:k]
        np.take(G, groups[0], axis=1, out=o[:, :len(groups[0])])
        if len(groups[0]) < bpr_out:
            o[:, len(groups[0]):] = 0
        for s in (1, 2, 3):
            g = groups[s]
            if len(g) == 0:
                continue
            tg = tmps[s][:k]
            np.take(G, g, axis=1, out=tg)
            tg <<= 2 * s
            o[:, :len(g)] |= tg
        T["gather"] += time.time() - t; t = time.time()
        fl = np.flatnonzero(st == 1)
        if len(fl):                                      # swap hom codes in the packed bytes, keep padding at 0
            o[fl] = _SWAP_BYTE[o[fl]]
            if pad_mask is not None:
                o[fl, -1] &= pad_mask
        T["pack"] += time.time() - t; t = time.time()
        fout.seek(3 + b0 * bpr_out); o.tofile(fout)
        T["write"] += time.time() - t
    fout.close(); fin.close()
    return T, minor_swapped, n_reads, n_rows_read


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True); ap.add_argument("--extract", required=True); ap.add_argument("--keep", required=True)
    ap.add_argument("--a1", default=None, help="'SNP allele' lines (production target_data.a1)")
    ap.add_argument("--a1-from-bim", default=None, help="take the counted allele from this prefix's .bim A1 (the fixture)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=256, help="wanted rows per read/decode/write block")
    ap.add_argument("--max-gap", type=int, default=2, help="raw rows further apart than this start a new read")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("BENCH_THREADS", "1")))
    ap.add_argument("--unassignable", choices=["minor", "complement", "error"], default="minor")
    a = ap.parse_args()
    t0 = time.time()
    bim, fam, rows, keep, state, (n_set, n_swap, n_strand, n_undec) = plan(a)
    M, n, n_raw = len(rows), len(keep), len(fam)
    bpr_in, bpr_out = (n_raw + 3) // 4, (n + 3) // 4
    path = a.raw + ".bed"
    if os.path.getsize(path) != 3 + len(bim) * bpr_in:
        raise SystemExit(f"{path}: size {os.path.getsize(path)} != 3 + {len(bim)} x {bpr_in}")
    if open(path, "rb").read(3) != MAGIC:
        raise SystemExit(f"{path}: not a SNP-major PLINK .bed")
    t_plan = time.time() - t0
    print(f"raw: {len(bim)} variants x {n_raw} people; wanted {M} rows ({n_set} with a counted allele: {n_swap} swapped"
          f"{f', {n_strand} matched on the other strand' if n_strand else ''}; {n_undec} left to the minor-allele rule); "
          f"kept {n} people; plan {t_plan:.1f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".bed", "wb") as f:
        f.write(MAGIC); f.truncate(3 + M * bpr_out)
    # split the wanted rows into contiguous ranges, one per worker
    nt = max(1, min(a.threads, M))
    edges = np.linspace(0, M, nt + 1).astype(int)
    jobs = [(a, rows, keep, state, int(edges[i]), int(edges[i + 1]), n_raw) for i in range(nt) if edges[i + 1] > edges[i]]
    if len(jobs) > 1 and hasattr(os, "fork"):
        import multiprocessing as mp
        with mp.get_context("fork").Pool(len(jobs)) as pool:
            res = pool.map(work, jobs)
    else:
        res = [work(j) for j in jobs]
    T = {k: sum(r[0][k] for r in res) for k in res[0][0]}
    minor_sw = sorted(j for r in res for j in r[1])
    n_reads = sum(r[2] for r in res); n_rows_read = sum(r[3] for r in res)
    # .bim / .fam
    swap = state == 1; swap[minor_sw] = True
    with open(a.out + ".bim", "w") as w:
        for j, r in enumerate(rows):
            f = bim[r]
            a1, a2 = (f[5], f[4]) if swap[j] else (f[4], f[5])
            w.write(f"{f[0]}\t{f[1]}\t{f[2]}\t{f[3]}\t{a1}\t{a2}\n")
    with open(a.out + ".fam", "w") as w:
        for i in keep:
            w.write(" ".join(fam[i][:6]) + "\n")
    wall = time.time() - t0
    gb_in = n_rows_read * bpr_in / 1e9; gb_out = M * bpr_out / 1e9
    print(f"bed[seek_v1]: {M} variants x {n} people -> {a.out}.bed ({gb_out:.2f} GB); raw rows read {n_rows_read} "
          f"({n_rows_read / M:.2f}x wanted, {gb_in:.2f} GB) in {n_reads} reads; minor-allele rule swapped {len(minor_sw)} of {n_undec}", flush=True)
    print(f"kernel time ({nt} workers, summed): " + ", ".join(f"{k} {v:.1f}s" for k, v in T.items())
          + f"; wall {wall:.1f}s ({gb_in / max(wall, 1e-9):.2f} GB/s in)", flush=True)


if __name__ == "__main__":
    main()
