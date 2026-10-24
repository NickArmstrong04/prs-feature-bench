#!/usr/bin/env python3
"""make_ld_weights.py -- weight files for the LD-ablation rows (README §12).

Every row of the ablation is the SAME bed, the SAME per-gene clumps (12b) and
the SAME kernel (sparse_score.py); only the (gene, snp, a1, channel, beta)
rows differ. This script writes them, in the weight contract of
lib/make_ct_weights.py, for four feature constructions:

  --mode ct               the authors' features: per-gene clump index SNPs,
                          GWAS marginal beta, 11 nested P-bins. Must reproduce
                          make_ct_weights.py row for row (the control).
  --mode ct-unique        same clumps and betas, but an index SNP that several
                          genes share is kept in ONE gene only -- the gene whose
                          window centre is nearest (ties: the narrower window).
                          Removes the shared-SNP duplication (ld_experiments.md
                          §2A item 3) and nothing else.
  --mode ldpred2          no clumping, no thresholds: every window SNP that has an
                          LDpred2 posterior beta, with that beta, ONE channel
                          (--beta-col, e.g. the chosen grid model g011). Feature =
                          sum_{j in W_g} beta_hat_j x_ij  (kernel --mode sum).
  --mode ldpred2-unique   ldpred2 with the same unique assignment.

Inputs
  --windows gene_windows.tsv      chr start end gene (fixture or $PRSNET_WINDOWS)
  --genes-dir <dir>               <gene>.snplist per gene (12a `snps` output)
  --manifest clump_manifest.tsv   gene start end n_snps big n_index status  (ct modes)
  --clumps <dir>                  <gene>.clumped                               (ct modes)
  --gwas gwas.QC.txt  --pvalue SNP.pvalue  --thresholds "..."                (ct modes)
  --bim <prefix>.bim              SNP positions (any mode; for the distance rule)
  --betas betas.txt --beta-col g011   LDpred2 10b `write` output: SNP CHR A1 cols... (ldpred2 modes)
  --out weights.tsv  --genes-out genes.txt  [--report assign_report.tsv]

Unique-assignment rule (recorded as a deviation): for a SNP at position p in the
windows of genes g_1..g_k, keep it in argmin_g |p - (start_g + end_g)/2|, ties
by smaller (end - start), then by gene name. Applied to the SNP set the mode
scores (index SNPs for ct-unique, window SNPs with a beta for ldpred2-unique),
so a gene that loses a SNP loses it from its score; its clump is unchanged.
genes-out lists every gene in --windows for this chromosome so the kernel
makes zero columns for genes left with nothing.
"""
import argparse, os, sys
from collections import defaultdict


def read_windows(path, chrom=None):
    out = {}
    with open(path) as fh:
        hdr = fh.readline().split()
        if hdr[:4] != ["chr", "start", "end", "gene"]:
            sys.exit(f"{path}: unexpected header {hdr}")
        for l in fh:
            c, s, e, g = l.rstrip("\n").split("\t")[:4]
            c = c.replace("chr", "")
            if chrom is not None and c != str(chrom):
                continue
            out[g] = (int(s), int(e))
    return out


def read_bim_pos(path):
    pos = {}
    with open(path) as fh:
        for l in fh:
            f = l.split()
            pos[f[1]] = int(f[3])
    return pos


def gene_snps_from_lists(genes_dir, genes):
    out = {}
    for g in genes:
        p = os.path.join(genes_dir, g + ".snplist")
        if os.path.isfile(p):
            out[g] = [l.strip() for l in open(p) if l.strip()]
    return out


def read_clumps(clumps_dir, manifest):
    """gene -> list of index SNPs (status ok); every gene in the manifest in order."""
    genes, idx = [], {}
    with open(manifest) as fh:
        fh.readline()
        for l in fh:
            f = l.rstrip("\n").split("\t")
            g, status = f[0], f[-1]
            if status not in ("ok", "nosnp"):
                sys.exit(f"gene {g} has status {status}; refuse to build weights from a partial clump set")
            genes.append(g)
            if status != "ok":
                continue
            s = []
            with open(os.path.join(clumps_dir, g + ".clumped")) as ch:
                ch.readline()
                for cl in ch:
                    c = cl.split()
                    if len(c) >= 3:
                        s.append(c[2])
            idx[g] = s
    return genes, idx


def read_gwas(path):
    eff, beta = {}, {}
    with open(path) as fh:
        fh.readline()
        for l in fh:
            f = l.split()
            eff[f[2]] = f[3]; beta[f[2]] = f[11]
    return eff, beta


def read_pvalue(path):
    pval = {}
    with open(path) as fh:
        for l in fh:
            f = l.split()
            if len(f) >= 2:
                try:
                    pval[f[0]] = float(f[1])
                except ValueError:
                    pass
    return pval


def read_betas(path, col):
    """betas.txt: SNP CHR A1 <model columns...>; -> snp -> (a1, beta) for one column."""
    out = {}
    with open(path) as fh:
        hdr = fh.readline().split()
        if hdr[:3] != ["SNP", "CHR", "A1"]:
            sys.exit(f"{path}: expected header SNP CHR A1 ..., got {hdr[:5]}")
        if col not in hdr:
            sys.exit(f"{path}: no column {col}; have {hdr[3:8]} ... ({len(hdr) - 3} models)")
        j = hdr.index(col)
        for l in fh:
            f = l.split()
            b = float(f[j])
            if b != 0.0:
                out[f[0]] = (f[2], b)
    return out


def unique_assign(gene_snps, windows, pos):
    """gene -> set of SNPs it KEEPS under the nearest-window-centre rule.
    Returns (kept, report rows [(snp, n_genes, winner, losers)])."""
    owners = defaultdict(list)
    for g, snps in gene_snps.items():
        for s in snps:
            owners[s].append(g)
    kept = {g: set() for g in gene_snps}
    report = []
    for s, gs in owners.items():
        if len(gs) == 1:
            kept[gs[0]].add(s); continue
        p = pos.get(s)
        if p is None:
            sys.exit(f"SNP {s} not in the .bim; cannot apply the distance rule")
        def key(g):
            st, en = windows[g]
            return (abs(p - (st + en) / 2.0), en - st, g)
        win = min(gs, key=key)
        kept[win].add(s)
        report.append((s, len(gs), win, ",".join(sorted(x for x in gs if x != win))))
    return kept, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["ct", "ct-unique", "ldpred2", "ldpred2-unique"])
    ap.add_argument("--windows", required=True); ap.add_argument("--chr", default=None)
    ap.add_argument("--genes-dir", default=None, help="<gene>.snplist dir (ldpred2 modes; ct modes use the clumps)")
    ap.add_argument("--manifest", default=None); ap.add_argument("--clumps", default=None)
    ap.add_argument("--gwas", default=None); ap.add_argument("--pvalue", default=None); ap.add_argument("--thresholds", default=None)
    ap.add_argument("--bim", default=None, help="<prefix>.bim for SNP positions (needed by the unique modes)")
    ap.add_argument("--betas", default=None); ap.add_argument("--beta-col", default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--genes-out", required=True)
    ap.add_argument("--report", default=None, help="unique modes: one row per reassigned SNP")
    a = ap.parse_args()
    ct = a.mode.startswith("ct"); unique = a.mode.endswith("unique")

    windows = read_windows(a.windows, a.chr)
    if ct:
        for k in ("manifest", "clumps", "gwas", "pvalue", "thresholds"):
            getattr(a, k) or sys.exit(f"--mode {a.mode} needs --{k}")
        genes, gene_snps = read_clumps(a.clumps, a.manifest)
        eff, beta = read_gwas(a.gwas); pval = read_pvalue(a.pvalue)
        thr = sorted((float(t), t) for t in a.thresholds.split())
    else:
        for k in ("genes_dir", "betas", "beta_col"):
            getattr(a, k) or sys.exit(f"--mode {a.mode} needs --{k.replace('_', '-')}")
        genes = sorted(windows)
        bet = read_betas(a.betas, a.beta_col)
        lists = gene_snps_from_lists(a.genes_dir, genes)
        gene_snps = {g: [s for s in snps if s in bet] for g, snps in lists.items()}
    missing_w = [g for g in gene_snps if g not in windows]
    if missing_w:
        sys.exit(f"{len(missing_w)} genes have SNPs but no window row (e.g. {missing_w[:3]}); pass the right --windows/--chr")

    if unique:
        pos = read_bim_pos(a.bim) if a.bim else sys.exit("unique modes need --bim")
        kept, report = unique_assign(gene_snps, windows, pos)
        if a.report:
            with open(a.report, "w") as w:
                w.write("snp\tn_genes\tkept_in\tremoved_from\n")
                for r in sorted(report):
                    w.write("\t".join(str(x) for x in r) + "\n")
    else:
        kept = {g: set(s) for g, s in gene_snps.items()}; report = []

    n_rows = n_over = n_nogwas = 0
    n_genes_scored = 0
    with open(a.out, "w") as w, open(a.genes_out, "w") as gw:
        w.write("gene\tsnp\ta1\tchannel\tbeta\n")
        for g in genes:
            gw.write(g + "\n")
            snps = gene_snps.get(g, [])
            wrote = False
            for s in snps:                                  # file order of the clump / snplist
                if s not in kept.get(g, ()):
                    continue
                if ct:
                    if s not in eff or s not in pval:
                        n_nogwas += 1; continue
                    lab = next((t for v, t in thr if pval[s] <= v), None)
                    if lab is None:
                        n_over += 1; continue
                    w.write(f"{g}\t{s}\t{eff[s]}\tP<={lab}\t{beta[s]}\n")
                else:
                    a1, b = bet[s]
                    w.write(f"{g}\t{s}\t{a1}\t{a.beta_col}\t{b:.6e}\n")
                n_rows += 1; wrote = True
            n_genes_scored += wrote
    n_in = sum(len(v) for v in gene_snps.values())
    n_removed = n_in - sum(len(v) for v in kept.values())
    print(f"ld weights [{a.mode}]: {len(genes)} genes ({n_genes_scored} with >=1 row), {n_in} (gene, SNP) links in, "
          f"{n_removed} removed by unique assignment ({len(report)} shared SNPs), {n_rows} rows written"
          + (f"; {n_over} above the top threshold, {n_nogwas} absent from gwas/pvalue (must be 0)" if ct else
             f"; channel {a.beta_col}, {len(bet)} SNPs with a non-zero beta on this file") + f" -> {a.out}")
    if ct and n_nogwas:
        sys.exit(2)


if __name__ == "__main__":
    main()
