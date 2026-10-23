"""dosage_select.py -- selection helpers of dosage_v1/dosage.py, copied verbatim so
dosage_raw_v0 selects exactly the SNPs and gene links dosage_v1 does
(read_bim, read_fam, read_pvalues, select). Keep in sync by copying, not editing."""
import os
import numpy as np


def read_bim(prefix):
    sid, a1, bp = [], [], []
    with open(prefix + ".bim") as fh:
        for l in fh:
            f = l.split()
            sid.append(f[1]); bp.append(int(f[3])); a1.append(f[4])
    return sid, a1, np.asarray(bp, dtype=np.int64)


def read_fam(prefix):
    with open(prefix + ".fam") as fh:
        return [(f[0], f[1]) for f in (l.split() for l in fh) if f]


def read_pvalues(path):
    """SNP.pvalue: 'SNP P' per line, header optional (non-numeric P skipped)."""
    p = {}
    with open(path) as fh:
        for l in fh:
            f = l.split()
            if len(f) < 2:
                continue
            try:
                p[f[0]] = float(f[1])
            except ValueError:
                pass
    return p


def select(prefix_sid, pvals, genes_dir, manifest, pthr):
    """-> rows (bim row indices, ascending), col_of_row, gene2rows {gene: [rows]}, counts"""
    row_of = {s: i for i, s in enumerate(prefix_sid)}
    sig = {s for s, p in pvals.items() if p < pthr and s in row_of}
    n_sig_nobim = sum(1 for s, p in pvals.items() if p < pthr and s not in row_of)
    genes = []
    with open(manifest) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ic = hdr.index("n_snps") if "n_snps" in hdr else 3      # production's awk uses $4 when unnamed
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if len(f) > ic and f[ic].isdigit() and int(f[ic]) > 0:
                genes.append(f[0])
    gene2rows, used = {}, set()
    for g in genes:
        rows_g = set()
        with open(os.path.join(genes_dir, g + ".snplist")) as fh:
            for l in fh:
                s = l.strip()
                if s in sig:
                    rows_g.add(row_of[s])
        if rows_g:
            gene2rows[g] = sorted(rows_g); used |= rows_g
    rows = np.asarray(sorted(used), dtype=np.int64)
    counts = dict(p_rows=len(pvals), sig_in_bim=len(sig), sig_not_in_bim=n_sig_nobim,
                  genes_with_snps=len(gene2rows), genes_total=len(genes),
                  selected=len(rows), sig_unlinked=len(sig) - len(rows))
    return rows, gene2rows, counts
