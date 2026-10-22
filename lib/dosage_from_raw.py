#!/usr/bin/env python3
"""dosage_from_raw.py -- plink `--recode A` output -> the harness's dosage layout.

    python dosage_from_raw.py --raw <out.raw> --bim <geno.bim> --pvalue <SNP.pvalue>
                              --pairs <gene_snp.tsv> --out <dir>

Used by variants/ref_dosage. The .raw has 6 leading columns (FID IID PAT MAT
SEX PHENOTYPE) then one column per variant named <snp>_<A1>, values 0/1/2 =
copies of A1, NA = missing. Writes samples.txt, snps.tsv, X.npy (int8, 3 =
missing), pvalues.npy, gene2snps.{tsv,pkl}; columns in the .raw's order (= .bim
order). Checks every counted allele against the .bim's A1.
--pairs: two columns, gene<TAB>snp, one row per gene-SNP link.
"""
import argparse, os, pickle
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True); ap.add_argument("--bim", required=True)
    ap.add_argument("--pvalue", required=True); ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    bim_a1, bim_bp = {}, {}
    with open(a.bim) as fh:
        for l in fh:
            f = l.split(); bim_a1[f[1]] = f[4]; bim_bp[f[1]] = int(f[3])
    pv = {}
    with open(a.pvalue) as fh:
        for l in fh:
            f = l.split()
            if len(f) >= 2:
                try:
                    pv[f[0]] = float(f[1])
                except ValueError:
                    pass

    with open(a.raw, "rb") as fh:
        hdr = fh.readline().replace(b"\t", b" ").split()
        if hdr[:6] != [b"FID", b"IID", b"PAT", b"MAT", b"SEX", b"PHENOTYPE"]:
            raise SystemExit(f"{a.raw}: unexpected header {hdr[:6]}")
        snps, a1s = [], []
        for h in hdr[6:]:
            s, al = h.decode().rsplit("_", 1)
            if s not in bim_a1:
                raise SystemExit(f"{a.raw}: column {s} not in .bim")
            if al != bim_a1[s]:
                raise SystemExit(f"{a.raw}: column {s} counts {al}, .bim A1 is {bim_a1[s]}")
            snps.append(s); a1s.append(al)
        M = len(snps)
        lines = [l for l in fh if l.strip()]
    n = len(lines)
    X = np.lib.format.open_memmap(os.path.join(a.out, "X.npy"), mode="w+", dtype=np.int8, shape=(n, M))
    fam = []
    for i, l in enumerate(lines):
        l = l.replace(b"\t", b" ").rstrip(b"\r\n")
        # skip 6 fields
        pos = 0
        for _ in range(6):
            pos = l.index(b" ", pos) + 1
        fam.append(tuple(x.decode() for x in l[:pos - 1].split()[:2]))
        rest = l[pos:].replace(b"NA", b"3")
        if len(rest) != 2 * M - 1:
            raise SystemExit(f"{a.raw}: row {i} has {len(rest.split())} genotype tokens, expected {M} single-character ones")
        v = np.frombuffer(rest, dtype=np.uint8)[::2].astype(np.int8) - 48
        if v.min() < 0 or v.max() > 3:
            raise SystemExit(f"{a.raw}: row {i} has a genotype outside 0/1/2/NA")
        X[i, :] = v
    X.flush(); del X

    col = {s: j for j, s in enumerate(snps)}
    with open(os.path.join(a.out, "samples.txt"), "w") as w:
        for fid, iid in fam:
            w.write(f"{fid}\t{iid}\n")
    p = np.asarray([pv[s] for s in snps], dtype=np.float64)
    np.save(os.path.join(a.out, "pvalues.npy"), p)
    with open(os.path.join(a.out, "snps.tsv"), "w") as w:
        w.write("col\tsnp\ta1\tbp\tp\n")
        for j, s in enumerate(snps):
            w.write(f"{j}\t{s}\t{a1s[j]}\t{bim_bp[s]}\t{p[j]!r}\n")
    g2 = {}
    with open(a.pairs) as fh:
        for l in fh:
            f = l.split()
            if len(f) >= 2 and f[1] in col:
                g2.setdefault(f[0], set()).add(col[f[1]])
    g2c = {g: sorted(v) for g, v in g2.items()}
    with open(os.path.join(a.out, "gene2snps.pkl"), "wb") as w:
        pickle.dump(g2c, w)
    with open(os.path.join(a.out, "gene2snps.tsv"), "w") as w:
        w.write("gene\tn_snps\tsnps\n")
        for g in sorted(g2c):
            w.write(f"{g}\t{len(g2c[g])}\t" + ",".join(snps[j] for j in g2c[g]) + "\n")
    print(f"reference: {n} x {M} int8 from {a.raw}, {len(g2c)} genes, {sum(len(v) for v in g2c.values())} links -> {a.out}")


if __name__ == "__main__":
    main()
