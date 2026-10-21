#!/usr/bin/env python3
"""weights_to_plink2.py -- long-format eQTL weights -> one wide plink2 score file.

    python weights_to_plink2.py <weights.tsv> <geno.bim> <out_wide.tsv> [--min-snps 1]

Input (the harness's weight contract, fixture/weights/<set>.tsv, tab-separated,
header):
    gene    snp    a1    channel    beta
one row per (gene, SNP, channel): beta is the effect on the gene's expression in
that channel (cell type / tissue) per copy of allele a1. SNP ids must be the
fixture's .bim ids (chr:pos).

Output: a plink2 --score file with header
    SNP  A1  <gene>|<channel>  <gene>|<channel> ...
rows = SNPs that (1) exist in the .bim and (2) have a1 equal to one of the two
.bim alleles; zero where a SNP carries no weight for a column. Columns with
fewer than --min-snps usable SNPs are dropped. A sidecar <out>.columns.tsv
lists col gene channel n_snps in the same order, for the harness layout.

Rows that are dropped are counted and reported, never silently lost.
"""
import argparse, sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("bim"); ap.add_argument("out")
    ap.add_argument("--min-snps", type=int, default=1)
    a = ap.parse_args()

    alleles = {}
    with open(a.bim) as fh:
        for l in fh:
            f = l.split()
            alleles[f[1]] = (f[4], f[5])

    cols = {}                   # (gene, channel) -> col index
    w = defaultdict(dict)       # snp -> {col: beta}
    snp_a1 = {}
    n_in = n_nobim = n_badallele = n_a1conflict = 0
    with open(a.weights) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {h: i for i, h in enumerate(hdr)}
        for need in ("gene", "snp", "a1", "channel", "beta"):
            if need not in ix:
                sys.exit(f"weights file lacks column '{need}' (has {hdr})")
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if len(f) < len(hdr):
                continue
            n_in += 1
            snp, a1 = f[ix["snp"]], f[ix["a1"]]
            if snp not in alleles:
                n_nobim += 1; continue
            if a1 not in alleles[snp]:
                n_badallele += 1; continue
            beta = float(f[ix["beta"]])
            key = (f[ix["gene"]], f[ix["channel"]])
            if key not in cols:
                cols[key] = len(cols)
            c = cols[key]
            # contract: one effect allele per SNP across every gene/channel row
            # (true of any eQTL release). A plink2 score file has one allele per
            # SNP row and no place for the 2*beta constant a flip would need.
            if snp_a1.setdefault(snp, a1) != a1:
                sys.exit(f"weights: SNP {snp} is given on allele {a1} in one row and {snp_a1[snp]} in another; "
                         "the weight contract requires one a1 per SNP -- fix the source file")
            if c in w[snp]:
                n_a1conflict += 1          # duplicate (gene, snp, channel) row
            w[snp][c] = beta

    # drop thin columns
    nsnp = defaultdict(int)
    for snp, d in w.items():
        for c in d:
            nsnp[c] += 1
    keep = [c for (k, c) in sorted(cols.items(), key=lambda kv: kv[1]) if nsnp[c] >= a.min_snps]
    remap = {c: j for j, c in enumerate(keep)}
    inv = {c: k for k, c in cols.items()}

    with open(a.out, "w") as fo, open(a.out + ".columns.tsv", "w") as fc:
        fo.write("SNP\tA1\t" + "\t".join(f"{inv[c][0]}|{inv[c][1]}" for c in keep) + "\n")
        fc.write("col\tgene\tchannel\tn_snps\n")
        for j, c in enumerate(keep):
            fc.write(f"{j}\t{inv[c][0]}\t{inv[c][1]}\t{nsnp[c]}\n")
        row = [0.0] * len(keep)
        for snp in sorted(w):                       # deterministic order
            for j in range(len(keep)):
                row[j] = 0.0
            for c, b in w[snp].items():
                if c in remap:
                    row[remap[c]] = b
            fo.write(snp + "\t" + snp_a1[snp] + "\t" + "\t".join(repr(x) if x else "0" for x in row) + "\n")

    print(f"weights_to_plink2: {n_in} rows in; dropped {n_nobim} (SNP not in bim), "
          f"{n_badallele} (a1 not a bim allele), {n_a1conflict} duplicates overwritten; "
          f"{len(w)} SNPs x {len(keep)} columns written ({len(cols) - len(keep)} columns < {a.min_snps} SNPs dropped) -> {a.out}")


if __name__ == "__main__":
    main()
