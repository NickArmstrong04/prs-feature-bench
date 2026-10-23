#!/usr/bin/env python3
"""compare_bed.py -- exact comparison of two PLINK filesets (the bed layout, README s11).

    python compare_bed.py <golden_dir> <candidate_dir>      (each holds geno.bed/.bim/.fam)

PASS (exit 0) when the .fam rows match field for field in order, the .bim rows
match (chr, id, cM, bp, A1, A2) in order, and every genotype byte of the .bed
matches. FAIL (exit 2) otherwise, worst first: a different variant or sample
list, then the number of variants whose bytes differ (first few named, with
the number of differing people after decoding, and whether their .bim alleles
were swapped -- the signature of an orientation bug rather than a data bug).
"""
import os, sys
import numpy as np

_LUT = np.stack([(np.arange(256, dtype=np.uint8) >> (2 * s)) & 3 for s in range(4)], axis=1)


def rows(path):
    return [l.split() for l in open(path) if l.strip()]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    g, c = sys.argv[1], sys.argv[2]
    gf, cf = rows(f"{g}/geno.fam"), rows(f"{c}/geno.fam")
    gb, cb = rows(f"{g}/geno.bim"), rows(f"{c}/geno.bim")
    print(f"golden {g}: {len(gb)} variants x {len(gf)} people;  cand {c}: {len(cb)} x {len(cf)}")
    bad = []
    if [f[:6] for f in gf] != [f[:6] for f in cf]:
        gi, ci = [tuple(f[:2]) for f in gf], [tuple(f[:2]) for f in cf]
        if set(gi) != set(ci):
            bad.append(f"FAM: {len(set(gi) - set(ci))} golden people missing from cand, {len(set(ci) - set(gi))} extra")
        elif gi != ci:
            bad.append("FAM: same people, different order")
        else:
            d = sum(1 for x, y in zip(gf, cf) if x[:6] != y[:6])
            bad.append(f"FAM: {d} rows differ in PID/MID/SEX/PHENO")
    if [f[1] for f in gb] != [f[1] for f in cb]:
        gi, ci = [f[1] for f in gb], [f[1] for f in cb]
        if set(gi) != set(ci):
            bad.append(f"BIM: {len(set(gi) - set(ci))} golden variants missing from cand, {len(set(ci) - set(gi))} extra")
        else:
            bad.append("BIM: same variants, different order")
    swapped = []
    if not bad:
        other = 0
        for j, (x, y) in enumerate(zip(gb, cb)):
            if x[4:6] != y[4:6]:
                if x[4:6] == y[5:3:-1]:
                    swapped.append(j)
                else:
                    other += 1
            elif x[0] != y[0] or x[3] != y[3]:
                other += 1
        if swapped:
            bad.append(f"BIM: {len(swapped)} variants with A1/A2 swapped, e.g. {[gb[j][1] for j in swapped[:5]]}")
        if other:
            bad.append(f"BIM: {other} variants differ in chr/bp/alleles (not a plain swap)")
    # genotype bytes, only meaningful when the frames agree
    if [f[1] for f in gb] == [f[1] for f in cb] and len(gf) == len(cf):
        n, M = len(gf), len(gb); bpr = (n + 3) // 4
        for p in (f"{g}/geno.bed", f"{c}/geno.bed"):
            sz = os.path.getsize(p)
            if sz != 3 + M * bpr:
                bad.append(f"BED: {p} is {sz} bytes, expected 3 + {M} x {bpr}")
            if open(p, "rb").read(3) != b"\x6c\x1b\x01":
                bad.append(f"BED: {p} has no SNP-major magic")
        if not any(b.startswith("BED") for b in bad):
            G = np.memmap(f"{g}/geno.bed", dtype=np.uint8, mode="r", offset=3, shape=(M, bpr))
            C = np.memmap(f"{c}/geno.bed", dtype=np.uint8, mode="r", offset=3, shape=(M, bpr))
            step = max(1, (256 << 20) // bpr)
            badv = []
            for r0 in range(0, M, step):
                r1 = min(M, r0 + step)
                d = np.flatnonzero(np.any(G[r0:r1] != C[r0:r1], axis=1))
                badv += [r0 + int(i) for i in d]
            if badv:
                sw = set(swapped)
                names = []
                for j in badv[:5]:
                    dg = _LUT[G[j]].reshape(-1)[:n]; dc = _LUT[C[j]].reshape(-1)[:n]
                    names.append(f"{gb[j][1]}({int((dg != dc).sum())} people{', bim swapped' if j in sw else ''})")
                bad.insert(0, f"BED: {len(badv)} of {M} variants differ, {sum(1 for j in badv if j in sw)} of them with swapped .bim alleles; "
                              f"e.g. {', '.join(names)}")
            else:
                print(f"BED: all {M} variants x {n} people identical ({M * bpr / 1e6:.1f} MB)")
    if bad:
        for b in bad:
            print(b)
        print(f"FAIL: {len(bad)} problems, worst first above")
        sys.exit(2)
    print(f"PASS: {len(gb)} variants, {len(gf)} people, .bed/.bim/.fam identical")


if __name__ == "__main__":
    main()
