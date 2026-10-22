#!/usr/bin/env python3
"""dosage.py (dosage_v0) -- PRSNet-2 input features straight from the packed .bed.

    python dosage.py --bed <prefix> --pvalue <SNP.pvalue> --genes-dir <dir of <gene>.snplist>
                     --manifest <gene_manifest.tsv> --out <dir> [--pthr 0.01] [--block 4000]

PRSNet-2 (Li et al., bioRxiv 10.1101/2025.11.22.689899; github lihan97/PRSNet-2)
takes no per-gene PRS. Its inputs are
    X.npy          int8 (n_samples x n_snps): copies of the counted allele, 0/1/2,
                   3 = missing (the released loader sets 3 -> 0)
    pvalues.npy    float64 (n_snps,): GWAS P per column
    gene2snps.pkl  dict gene -> list of column indices (a SNP may sit in several genes)
Selection: SNPs with GWAS P < --pthr (paper default 0.01, strict) that lie in at
least one gene's window. Here a gene's window is the fixture's
genes/<gene>.snplist (production 12a's windows); the paper's own rule is the
gene body +-10 kb. Columns are in .bim order, so every gene's SNPs are a
near-contiguous run and the .bed is read in sequential spans. The counted
allele is .bim A1 (= the GWAS effect allele in this fixture), the same allele
plink --recode A counts, which is how the reference variant makes the golden.

Harness layout written (see README s2): samples.txt, snps.tsv (col snp a1 bp p),
X.npy, pvalues.npy, gene2snps.tsv (gene n_snps snps), gene2snps.pkl.

Kernel (what carried over from eqtl_sparse_v3): 2-bit codes -> int8 through a
256 x 4 lookup table, taken straight into a preallocated buffer, SNP blocks
read as one span each, written transposed into a memmapped .npy. No GEMM,
no imputation, no float at all.
"""
import argparse, os, pickle, time
import numpy as np

# 2-bit PLINK codes, low bits = first sample: 00 hom A1 -> 2, 01 missing -> 3, 10 het -> 1, 11 hom A2 -> 0
MISSING = 3
_CODE = np.array([2, MISSING, 1, 0], dtype=np.int8)
_LUT = np.stack([_CODE[(np.arange(256) >> (2 * s)) & 3] for s in range(4)], axis=1)   # (256, 4) int8


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", required=True, help="plink prefix")
    ap.add_argument("--pvalue", required=True, help="SNP.pvalue (SNP P)")
    ap.add_argument("--genes-dir", required=True, help="directory of <gene>.snplist")
    ap.add_argument("--manifest", required=True, help="gene_manifest.tsv (gene ... n_snps in col 4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pthr", type=float, default=0.01, help="keep SNPs with P < pthr (strict)")
    ap.add_argument("--block", type=int, default=4000, help="selected SNPs decoded per block")
    a = ap.parse_args()
    t0 = time.time()
    T = {}

    sid, a1, bp = read_bim(a.bed)
    fam = read_fam(a.bed)
    n, m_bim = len(fam), len(sid)
    pvals = read_pvalues(a.pvalue)
    rows, gene2rows, cnt = select(sid, pvals, a.genes_dir, a.manifest, a.pthr)
    M = len(rows)
    col_of_row = {int(r): j for j, r in enumerate(rows)}
    T["select"] = time.time() - t0
    print(f"select: P < {a.pthr:g}: {cnt['sig_in_bim']} of {cnt['p_rows']} SNPs in the .bim "
          f"({cnt['sig_not_in_bim']} significant not in .bim); {cnt['selected']} linked to "
          f"{cnt['genes_with_snps']} of {cnt['genes_total']} genes ({cnt['sig_unlinked']} significant SNPs in no gene, dropped)", flush=True)
    if M == 0:
        raise SystemExit("no SNPs selected")

    os.makedirs(a.out, exist_ok=True)
    bpr = (n + 3) // 4
    path = a.bed + ".bed"
    if os.path.getsize(path) != 3 + m_bim * bpr:
        raise SystemExit(f"{path}: size {os.path.getsize(path)} != 3 + {m_bim} x {bpr}")
    fh = open(path, "rb")
    if fh.read(3) != b"\x6c\x1b\x01":
        raise SystemExit(f"{path}: not a SNP-major PLINK .bed")

    X = np.lib.format.open_memmap(os.path.join(a.out, "X.npy"), mode="w+", dtype=np.int8, shape=(n, M))
    K = min(a.block, M)
    buf8 = np.empty((K, bpr, 4), dtype=np.int8)
    T["read"] = T["decode"] = T["write"] = 0.0
    n_span = 0
    for c0 in range(0, M, K):
        blk = rows[c0:c0 + K]; k = len(blk)
        t = time.time()
        r0, r1 = int(blk[0]), int(blk[-1]) + 1
        fh.seek(3 + r0 * bpr)
        span = np.fromfile(fh, dtype=np.uint8, count=(r1 - r0) * bpr).reshape(r1 - r0, bpr)
        raw = span[blk - r0]                                   # (k, bpr)
        n_span += r1 - r0
        T["read"] += time.time() - t; t = time.time()
        np.take(_LUT, raw, axis=0, out=buf8[:k])
        G = buf8[:k].reshape(k, -1)[:, :n]                      # (k, n) int8 view
        T["decode"] += time.time() - t; t = time.time()
        X[:, c0:c0 + k] = G.T
        T["write"] += time.time() - t
    t = time.time()
    X.flush(); del X
    T["write"] += time.time() - t

    # sidecars
    with open(os.path.join(a.out, "samples.txt"), "w") as w:
        for fid, iid in fam:
            w.write(f"{fid}\t{iid}\n")
    pv = np.asarray([pvals[sid[int(r)]] for r in rows], dtype=np.float64)
    np.save(os.path.join(a.out, "pvalues.npy"), pv)
    with open(os.path.join(a.out, "snps.tsv"), "w") as w:
        w.write("col\tsnp\ta1\tbp\tp\n")
        for j, r in enumerate(rows):
            r = int(r); w.write(f"{j}\t{sid[r]}\t{a1[r]}\t{bp[r]}\t{pv[j]!r}\n")
    g2c = {g: [col_of_row[r] for r in rs] for g, rs in gene2rows.items()}
    with open(os.path.join(a.out, "gene2snps.pkl"), "wb") as w:
        pickle.dump(g2c, w)
    with open(os.path.join(a.out, "gene2snps.tsv"), "w") as w:
        w.write("gene\tn_snps\tsnps\n")
        for g in sorted(g2c):
            w.write(f"{g}\t{len(g2c[g])}\t" + ",".join(sid[int(r)] for r in gene2rows[g]) + "\n")
    links = sum(len(v) for v in g2c.values())
    print(f"dosage[v0]: {n} x {M} int8 ({n * M / 1e6:.0f} MB), {len(g2c)} genes, {links} gene-SNP links, "
          f"span rows read {n_span} ({n_span / M:.2f}x selected)", flush=True)
    print("kernel time: " + ", ".join(f"{k} {v:.1f}s" for k, v in T.items()) + f"; total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
