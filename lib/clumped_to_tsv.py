#!/usr/bin/env python3
"""clumped_to_tsv.py -- plink --clump outputs for a gene set -> the bench's clumps layout.

    python clumped_to_tsv.py <clumps_dir> <clump_manifest.tsv> <out clumps.tsv>

clumps_dir holds <gene>.clumped (plink: SNP in column 3, one row per index
SNP, in P order); the manifest's last column is the status (ok / nosnp /
failed, as 12b's write_12b_results writes it). Output: gene, n_index, index
SNPs comma-joined in file order. A 'failed' gene is an error.
"""
import os, sys


def main(cdir, manifest, out):
    n_ok = n_idx = 0
    with open(out, "w") as w, open(manifest) as fh:
        hdr = fh.readline().rstrip("\n").split("\t"); ig = hdr.index("gene")
        w.write("gene\tn_index\tindex_snps\n")
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if not f or not f[0]:
                continue
            g, st = f[ig], f[-1]
            if st == "nosnp":
                w.write(f"{g}\t0\t\n"); continue
            if st != "ok":
                sys.exit(f"gene {g}: status {st}")
            idx = []
            with open(os.path.join(cdir, g + ".clumped")) as ch:
                ch.readline()
                for cl in ch:
                    c = cl.split()
                    if len(c) >= 3:
                        idx.append(c[2])
            w.write(f"{g}\t{len(idx)}\t" + ",".join(idx) + "\n"); n_ok += 1; n_idx += len(idx)
    print(f"clumped_to_tsv: {n_ok} clumped genes, {n_idx} index SNPs -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
