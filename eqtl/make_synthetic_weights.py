#!/usr/bin/env python3
"""make_synthetic_weights.py -- a placeholder eQTL weight set drawn from the fixture.

    python make_synthetic_weights.py <fixture_dir> <out.tsv> [--snps-per-gene 10] [--channels Tcell,Bcell,mono] [--seed 1]

Until the real sc-eQTL weights exist this gives the eQTL variants something
real-shaped to run on: for every gene with window SNPs, up to --snps-per-gene
SNPs are drawn from its window, each channel gets its own random betas, and
roughly half the rows are given on the .bim's second allele so the allele
orientation code paths are exercised. Output follows the weight contract
(gene snp a1 channel beta). The numbers mean nothing biologically.
"""
import argparse, glob, os, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture"); ap.add_argument("out")
    ap.add_argument("--snps-per-gene", type=int, default=10)
    ap.add_argument("--channels", default="Tcell,Bcell,mono")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    alleles = {}
    with open(os.path.join(a.fixture, "geno.bim")) as fh:
        for l in fh:
            f = l.split(); alleles[f[1]] = (f[4], f[5])
    chans = a.channels.split(",")
    orient = {}                      # one effect allele per SNP, as in a real eQTL release
    n_rows = n_genes = 0
    with open(a.out, "w") as w:
        w.write("gene\tsnp\ta1\tchannel\tbeta\n")
        for fn in sorted(glob.glob(os.path.join(a.fixture, "genes", "*.snplist"))):
            gene = os.path.basename(fn)[:-8]
            snps = [l.strip() for l in open(fn) if l.strip() in alleles]
            if not snps:
                continue
            n_genes += 1
            pick = rng.sample(snps, min(a.snps_per_gene, len(snps)))
            for ch in chans:
                for s in pick:
                    a1 = orient.setdefault(s, alleles[s][1] if rng.random() < 0.5 else alleles[s][0])
                    w.write(f"{gene}\t{s}\t{a1}\t{ch}\t{rng.gauss(0, 0.3):.8g}\n")
                    n_rows += 1
    print(f"synthetic weights: {n_genes} genes x {len(chans)} channels, {n_rows} rows -> {a.out}")


if __name__ == "__main__":
    main()
