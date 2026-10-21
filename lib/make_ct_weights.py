#!/usr/bin/env python3
"""make_ct_weights.py -- turn per-gene PLINK .clumped files into a weight file
for sparse_score.py --mode ct.

    python make_ct_weights.py --clumps <dir> --manifest <clump_manifest.tsv> \
        --gwas <gwas.QC.txt> --pvalue <SNP.pvalue> --thresholds "1e-5 ... 1" \
        --out <weights.tsv> --genes-out <genes.txt>

One row per (gene, index SNP): a1 = the GWAS effect allele (gwas.QC.txt col 4),
beta = gwas.QC.txt col 12 (the columns `12_prsnet.sh check` hands plink2),
channel = "P<=<t>" for the smallest threshold t with P <= t (the SNP's bin;
the kernel accumulates bins into the nested thresholds). SNPs with P above the
largest threshold get no row. genes-out lists every gene in the manifest with
status ok or nosnp, so genes without index SNPs still get zero columns.
"""
import argparse, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clumps", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--gwas", required=True); ap.add_argument("--pvalue", required=True)
    ap.add_argument("--thresholds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--genes-out", required=True)
    a = ap.parse_args()
    thr = [(float(t), t) for t in a.thresholds.split()]
    thr.sort()
    eff, beta = {}, {}
    with open(a.gwas) as fh:
        fh.readline()
        for l in fh:
            f = l.split()
            eff[f[2]] = f[3]; beta[f[2]] = f[11]
    pval = {}
    with open(a.pvalue) as fh:
        for l in fh:
            f = l.split()
            if len(f) >= 2:
                try:
                    pval[f[0]] = float(f[1])
                except ValueError:
                    pass                                      # header line
    n_rows = n_genes = n_over = n_nogwas = 0
    with open(a.out, "w") as w, open(a.genes_out, "w") as gw, open(a.manifest) as fh:
        w.write("gene\tsnp\ta1\tchannel\tbeta\n")
        fh.readline()
        for l in fh:
            f = l.rstrip("\n").split("\t")
            g, status = f[0], f[-1]
            if status not in ("ok", "nosnp"):
                raise SystemExit(f"gene {g} has status {status}; refuse to build weights from a partial clump set")
            gw.write(g + "\n"); n_genes += 1
            if status != "ok":
                continue
            with open(os.path.join(a.clumps, g + ".clumped")) as ch:
                ch.readline()
                for cl in ch:
                    c = cl.split()
                    if len(c) < 3:
                        continue
                    s = c[2]
                    if s not in eff or s not in pval:
                        n_nogwas += 1; continue
                    p = pval[s]
                    lab = next((t for v, t in thr if p <= v), None)
                    if lab is None:
                        n_over += 1; continue
                    w.write(f"{g}\t{s}\t{eff[s]}\tP<={lab}\t{beta[s]}\n"); n_rows += 1
    print(f"ct weights: {n_genes} genes, {n_rows} (gene, index SNP) rows; {n_over} above the top threshold, "
          f"{n_nogwas} index SNPs absent from gwas/pvalue (must be 0) -> {a.out}")
    if n_nogwas:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
