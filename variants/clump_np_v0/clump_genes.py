#!/usr/bin/env python3
"""clump_genes.py (clump_np_v0) -- plink --clump's index-SNP selection for every gene from the LD pair graph.

    python clump_genes.py --pairs <pairs.npz> --genes-dir <dir> --manifest <gene_manifest.tsv>
                          --pvalue <SNP.pvalue> --out <dir> [--clumps-tsv <file>]

For gene g with SNP set S_g (its .snplist, restricted to the .bim and to SNPs
with a P), plink's greedy with --clump-p1 1 --clump-index-first visits S_g in
ascending P; a SNP not yet clumped becomes an index and every not-yet-clumped
SNP of S_g within --clump-kb with r^2 >= --clump-r2 joins its clump. The index
set therefore depends only on the P order and on the pairs (r^2 >= threshold,
within kb) among S_g -- exactly what ld_pairs.py stored for the chromosome.
Ties in P are visited in .bim (position) order (plink sorts the loaded
association records by P with a stable sort over its variant order).

Writes, so lib/make_ct_weights.py and the rest of the chain run unchanged:
    <out>/<gene>.clumped       header + one row per index SNP: CHR F SNP BP P TOTAL (SNP in column 3,
                               TOTAL = SNPs of S_g clumped under it; the S05..SP2 columns are not reproduced)
    <out>/clump_manifest.tsv   gene_manifest columns + n_index + status (ok / nosnp), as 12b's write_12b_results
    --clumps-tsv               gene  n_index  index SNPs comma-joined in P order   (the bench's clumps layout)
"""
import argparse, os, time
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True); ap.add_argument("--genes-dir", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--pvalue", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--clumps-tsv", default=None)
    ap.add_argument("--chr", default=None, help="CHR column of the .clumped rows (default: from the first .snplist's gene window)")
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.pairs, allow_pickle=False)
    sid = z["sid"]; bp = z["bp"]; i = z["i"]; j = z["j"]
    m = len(sid)
    row = {s: k for k, s in enumerate(sid)}
    # adjacency as CSR over .bim rows (both directions)
    src = np.concatenate([i, j]); dst = np.concatenate([j, i])
    order = np.argsort(src, kind="stable")
    src = src[order]; dst = dst[order]
    indptr = np.searchsorted(src, np.arange(m + 1))
    pval = {}
    for l in open(a.pvalue):
        f = l.split()
        if len(f) >= 2:
            try:
                pval[f[0]] = float(f[1])
            except ValueError:
                pass
    P = np.full(m, np.inf)
    for s, k in row.items():
        p = pval.get(s)
        if p is not None:
            P[k] = p
    os.makedirs(a.out, exist_ok=True)
    hdr = [l.rstrip("\n").split("\t") for l in open(a.manifest)]
    cols = hdr[0]; ig = cols.index("gene"); ins = cols.index("n_snps") if "n_snps" in cols else 3
    chrom = a.chr
    n_genes = n_ok = n_index_total = 0
    tsv = open(a.clumps_tsv, "w") if a.clumps_tsv else None
    if tsv:
        tsv.write("gene\tn_index\tindex_snps\n")
    with open(os.path.join(a.out, "clump_manifest.tsv"), "w") as mw:
        mw.write("\t".join(cols) + "\tn_index\tstatus\n")
        for f in hdr[1:]:
            if not f or not f[0]:
                continue
            g = f[ig]; n_genes += 1
            if int(f[ins]) == 0:
                mw.write("\t".join(f) + "\t0\tnosnp\n")
                if tsv:
                    tsv.write(f"{g}\t0\t\n")
                continue
            snps = [l.split()[0] for l in open(os.path.join(a.genes_dir, g + ".snplist")) if l.strip()]
            rows = np.array([row[s] for s in snps if s in row and np.isfinite(P[row[s]])], dtype=np.int64)
            if len(rows) == 0:
                mw.write("\t".join(f) + "\t0\tnosnp\n")
                if tsv:
                    tsv.write(f"{g}\t0\t\n")
                continue
            rows = np.unique(rows)                                  # .bim order
            order = rows[np.argsort(P[rows], kind="stable")]        # P ascending, ties by position
            inset = np.zeros(m, dtype=bool); inset[rows] = True
            clumped = np.zeros(m, dtype=bool)
            index = []; totals = []
            for r in order:
                if clumped[r]:
                    continue
                nb = dst[indptr[r]:indptr[r + 1]]
                nb = nb[inset[nb] & ~clumped[nb]]
                nb = nb[nb != r]
                clumped[nb] = True; clumped[r] = True
                index.append(int(r)); totals.append(len(nb))
            if chrom is None:
                chrom = "?"
            with open(os.path.join(a.out, g + ".clumped"), "w") as w:
                w.write(" CHR    F             SNP         BP        P    TOTAL\n")
                for r, tot in zip(index, totals):
                    w.write(f"{chrom:>4} {1:>4} {sid[r]:>15} {bp[r]:>10} {P[r]:>10.4g} {tot:>8}\n")
            mw.write("\t".join(f) + f"\t{len(index)}\tok\n")
            if tsv:
                tsv.write(f"{g}\t{len(index)}\t" + ",".join(sid[r] for r in index) + "\n")
            n_ok += 1; n_index_total += len(index)
    if tsv:
        tsv.close()
    print(f"clump_genes: {n_genes} genes ({n_ok} clumped, {n_genes - n_ok} without SNPs), {n_index_total} index SNPs "
          f"from {len(i)} LD pairs -> {a.out}  ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
