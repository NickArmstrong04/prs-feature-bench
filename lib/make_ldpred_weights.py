#!/usr/bin/env python3
"""make_ldpred_weights.py -- LDpred2 posterior effects -> the long weight file
sparse_score.py --mode sum scores (gene, snp, a1, channel, beta), one row per
(gene, HapMap3+ SNP in the gene's window, channel). No clumping, no P
thresholds: 12b disappears, 12c sums posterior effects per gene.

    python make_ldpred_weights.py --betas $LDP_GEN/betas.txt --grid-params $LDP_GEN/grid_params.tsv \
        --channels auto | inf | g011 | g011-g021 | auto,inf,g035   (comma list; g###-g### is a range) \
        --bim <target_chr<k>.PH.bim> --windows <gene_windows.tsv> --chr <k> \
        [--manifest <gene_manifest.tsv>] [--assign shared|unique] \
        [--id-map rsid<TAB>target_id] [--pos-map rsid<TAB>chr<TAB>pos] \
        --out <weights.tsv> --genes-out <genes.txt> --channels-out <channels.txt> [--report <report.tsv>]

Inputs (10_ldpred2.R cmd_write, verified on Bouchet 2026-10-25):
  betas.txt        tab-separated, header SNP CHR A1 inf auto g001 ... g168; SNP = rsID,
                   A1 = the TARGET's own A1 (sign already reconciled), so betas are used
                   as-is; diverged grid models are zero-filled columns; absent models are
                   absent columns -> every column is found by header name, never position.
  grid_params.tsv  p h2 sparse label diverged n_nonzero, one row per g###. A requested
                   g### with diverged=TRUE is refused (it would silently score zeros).
  .bim             the step-04 aligned target for this chromosome (all QC'd SNPs, not the
                   window-only 12a bed) so the intergenic report means something.
  windows          chr start end gene (PRSNET_WINDOWS); membership is start < pos <= end,
                   the rule 12_prsnet.py snps uses.

rsID -> target ID: a betas SNP that is itself a .bim ID matches directly; otherwise
--id-map (two columns) or --pos-map (rsID chr pos, e.g. exported from map_hm3_plus.rds)
resolves it through the .bim's chr:pos. Unmatched rows are counted, never guessed.

--assign shared  (default)  a SNP in k overlapping windows gets k rows (the C+T geometry)
--assign unique             the SNP goes to the gene whose window centre is nearest
                            (ties: narrower window, then name), so summing the gene
                            scores over all genes reproduces the genome-wide LDpred2 PRS
                            restricted to windowed SNPs -- the identity-graph floor.

Zero betas (sparse grid models) write no row. The report gives, per channel, the
share of this chromosome's sum|beta| that falls outside every window (intergenic
mass the gene features cannot see) and in how many windows the SNPs sit.
"""
import argparse, bisect, os, re, sys
from collections import defaultdict


def expand_channels(spec):
    out = []
    for tok in spec.replace(" ", "").split(","):
        if not tok:
            continue
        m = re.fullmatch(r"g(\d{3})-g(\d{3})", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b < a:
                sys.exit(f"--channels: bad range {tok}")
            out += [f"g{k:03d}" for k in range(a, b + 1)]
        else:
            out.append(tok)
    if len(set(out)) != len(out):
        sys.exit(f"--channels: duplicate channel in {spec}")
    return out


def read_grid(path):
    grid = {}
    if not path or not os.path.isfile(path):
        return grid
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t"); ix = {h: i for i, h in enumerate(hdr)}
        for need in ("p", "h2", "sparse", "label", "diverged"):
            if need not in ix:
                sys.exit(f"{path}: lacks column {need} (has {hdr})")
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if len(f) < len(hdr):
                continue
            grid[f[ix["label"]]] = {"p": f[ix["p"]], "h2": f[ix["h2"]], "sparse": f[ix["sparse"]],
                                    "diverged": f[ix["diverged"]].upper() == "TRUE",
                                    "n_nonzero": f[ix["n_nonzero"]] if "n_nonzero" in ix else "?"}
    return grid


def read_windows(path, chrom):
    """[(start, end, gene)] on this chromosome, as 12_prsnet.py read_windows keys them."""
    out = []
    with open(path) as fh:
        hdr = fh.readline().split()
        if hdr[:4] != ["chr", "start", "end", "gene"]:
            sys.exit(f"{path}: expected header chr start end gene, got {hdr}")
        for l in fh:
            c, s, e, g = l.rstrip("\n").split("\t")[:4]
            c = c[3:] if c.startswith("chr") else c
            if c == str(chrom):
                out.append((int(s), int(e), g))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", required=True); ap.add_argument("--grid-params", default="")
    ap.add_argument("--channels", required=True)
    ap.add_argument("--bim", required=True); ap.add_argument("--windows", required=True)
    ap.add_argument("--chr", required=True)
    ap.add_argument("--manifest", default="", help="gene_manifest.tsv: every gene in it gets a column (zero if no weight)")
    ap.add_argument("--assign", choices=["shared", "unique"], default="shared")
    ap.add_argument("--id-map", default=""); ap.add_argument("--pos-map", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--genes-out", required=True)
    ap.add_argument("--channels-out", required=True); ap.add_argument("--report", default="")
    a = ap.parse_args()
    chrom = a.chr[3:] if a.chr.startswith("chr") else a.chr

    # --- channels: exist in betas.txt, not diverged -------------------------------
    channels = expand_channels(a.channels)
    grid = read_grid(a.grid_params)
    for c in channels:
        if c.startswith("g") and c[1:].isdigit():
            if not grid:
                sys.exit(f"channel {c} needs --grid-params to check divergence")
            if c not in grid:
                sys.exit(f"channel {c} is not in {a.grid_params}")
            if grid[c]["diverged"]:
                sys.exit(f"channel {c} DIVERGED (p {grid[c]['p']}, h2 {grid[c]['h2']}, sparse {grid[c]['sparse']}): "
                         f"betas.txt holds zeros for it; pick a surviving model")
    with open(a.betas) as fh:
        bhdr = fh.readline().rstrip("\n").split("\t")
    bix = {h: i for i, h in enumerate(bhdr)}
    for need in ("SNP", "CHR", "A1"):
        if need not in bix:
            sys.exit(f"{a.betas}: lacks column {need} (has {bhdr[:6]} ...)")
    missing = [c for c in channels if c not in bix]
    if missing:
        sys.exit(f"{a.betas}: no column(s) {missing}; it has {[h for h in bhdr[3:]][:8]} ...")
    cix = [bix[c] for c in channels]
    print("channels: " + ", ".join(
        f"{c} (p {grid[c]['p']}, h2 {grid[c]['h2']}, sparse {grid[c]['sparse']}, nonzero {grid[c]['n_nonzero']})"
        if c in grid else c for c in channels), flush=True)

    # --- target .bim for this chromosome ------------------------------------------
    bim_pos, bim_alleles, by_pos = {}, {}, {}
    n_bim = 0
    with open(a.bim) as fh:
        for l in fh:
            f = l.split()
            if not f:
                continue
            c = f[0][3:] if f[0].startswith("chr") else f[0]
            if c != chrom:
                continue
            n_bim += 1
            bim_pos[f[1]] = int(f[3]); bim_alleles[f[1]] = (f[4], f[5])
            by_pos.setdefault(int(f[3]), []).append(f[1])
    if n_bim == 0:
        sys.exit(f"{a.bim}: no variants on chromosome {chrom}")
    print(f"bim: {n_bim:,} variants on chr{chrom}", flush=True)
    idmap = {}
    if a.id_map:
        with open(a.id_map) as fh:
            for l in fh:
                f = l.split()
                if len(f) >= 2 and not f[0].lower().startswith(("rsid", "snp", "#")):
                    idmap[f[0]] = f[1]
        print(f"id-map: {len(idmap):,} rsID -> target ID pairs", flush=True)
    posmap = {}
    if a.pos_map:
        with open(a.pos_map) as fh:
            for l in fh:
                f = l.split()
                if len(f) >= 3 and f[2].isdigit():
                    c = f[1][3:] if f[1].startswith("chr") else f[1]
                    if c == chrom:
                        posmap[f[0]] = int(f[2])
        print(f"pos-map: {len(posmap):,} rsIDs with a chr{chrom} position", flush=True)

    # --- betas on this chromosome, resolved to target IDs -------------------------
    T = len(channels)
    snps = []                                   # (pos, target_id, a1, [beta_c])
    n_chr = n_direct = n_viamap = n_viapos = n_unmatched = n_badallele = n_allzero = 0
    mass_all = [0.0] * T                        # sum|beta| over every resolved SNP on the chromosome
    with open(a.betas) as fh:
        fh.readline()
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if len(f) < len(bhdr):
                continue
            c = f[bix["CHR"]]; c = c[3:] if c.startswith("chr") else c
            if c != chrom:
                continue
            n_chr += 1
            rs = f[bix["SNP"]]
            tid = None
            if rs in bim_pos:
                tid = rs; n_direct += 1
            elif rs in idmap and idmap[rs] in bim_pos:
                tid = idmap[rs]; n_viamap += 1
            elif rs in posmap and posmap[rs] in by_pos:
                cands = by_pos[posmap[rs]]
                a1 = f[bix["A1"]]
                cands = [t for t in cands if a1 in bim_alleles[t]] or cands
                if len(cands) == 1:
                    tid = cands[0]; n_viapos += 1
            if tid is None:
                n_unmatched += 1; continue
            a1 = f[bix["A1"]]
            if a1 not in bim_alleles[tid]:
                n_badallele += 1; continue
            betas = [float(f[i]) for i in cix]
            for k, b in enumerate(betas):
                mass_all[k] += abs(b)
            if not any(betas):
                n_allzero += 1; continue
            snps.append((bim_pos[tid], tid, a1, betas))
    snps.sort()
    print(f"betas: {n_chr:,} rows on chr{chrom}; matched {n_direct:,} by ID, {n_viamap:,} via --id-map, "
          f"{n_viapos:,} via --pos-map; {n_unmatched:,} unmatched (dropped), {n_badallele:,} A1 not a .bim allele "
          f"(dropped), {n_allzero:,} zero in every channel (no rows); {len(snps):,} SNPs carry weight", flush=True)
    if n_chr and n_unmatched > 0.2 * n_chr:
        print(f"WARNING: {100.0 * n_unmatched / n_chr:.0f}% of chr{chrom} betas did not resolve to a target ID -- "
              "if the .bim uses chr:pos IDs pass --id-map or --pos-map", flush=True)

    # --- windows -> rows -------------------------------------------------------------
    windows = read_windows(a.windows, chrom)
    genes_all = [g for _, _, g in windows]
    if a.manifest:
        genes_all = []
        with open(a.manifest) as fh:
            fh.readline()
            for l in fh:
                g = l.split("\t")[0].strip()
                if g and g not in genes_all:
                    genes_all.append(g)
    pos_list = [s[0] for s in snps]
    member = defaultdict(list)                  # snp index -> [(gene, centre, width)]
    for s, e, g in windows:
        lo = bisect.bisect_right(pos_list, s)   # start < pos
        hi = bisect.bisect_right(pos_list, e)   # pos <= end
        for i in range(lo, hi):
            member[i].append((g, (s + e) / 2.0, e - s))
    n_rows = 0; n_in = 0
    mass_in = [0.0] * T
    hist = defaultdict(int)                     # windows per SNP (0, 1, 2, 3-5, 6+)
    per_gene = defaultdict(int)
    with open(a.out, "w") as w:
        w.write("gene\tsnp\ta1\tchannel\tbeta\n")
        for i, (pos, tid, a1, betas) in enumerate(snps):
            ms = member.get(i, [])
            k = len(ms)
            hist[0 if k == 0 else 1 if k == 1 else 2 if k == 2 else 3 if k <= 5 else 6] += 1
            if k == 0:
                continue
            n_in += 1
            for c in range(T):
                mass_in[c] += abs(betas[c])
            if a.assign == "unique" and k > 1:
                ms = [min(ms, key=lambda m: (abs(pos - m[1]), m[2], m[0]))]
            for g, _, _ in ms:
                per_gene[g] += 1
                for c, b in zip(channels, betas):
                    if b != 0.0:
                        w.write(f"{g}\t{tid}\t{a1}\t{c}\t{b:.6e}\n"); n_rows += 1
    with open(a.genes_out, "w") as gw:
        gw.write("".join(g + "\n" for g in genes_all))
    with open(a.channels_out, "w") as cw:
        cw.write("".join(c + "\n" for c in channels))
    genes_with = sum(1 for g in genes_all if per_gene.get(g, 0) > 0)
    print(f"windows: {len(windows)} on chr{chrom}; {genes_with} of {len(genes_all)} genes get >= 1 weighted SNP "
          f"({a.assign} assignment); {n_in:,} SNPs in >= 1 window; windows per SNP: 0:{hist[0]:,} 1:{hist[1]:,} "
          f"2:{hist[2]:,} 3-5:{hist[3]:,} 6+:{hist[6]:,}", flush=True)
    print(f"rows: {n_rows:,} (gene, snp, channel) -> {a.out}", flush=True)
    for c, mi, ma in zip(channels, mass_in, mass_all):
        out = 1.0 - mi / ma if ma > 0 else float("nan")
        print(f"intergenic mass {c}: {100.0 * out:.1f}% of chr{chrom} sum|beta| lies outside every window "
              f"(sum|beta| in windows {mi:.4g} of {ma:.4g})", flush=True)
    if a.report:
        with open(a.report, "w") as r:
            r.write("chr\tchannel\tassign\tn_betas_chr\tn_matched\tn_unmatched\tn_snps_weighted\tn_snps_in_window\t"
                    "n_rows\tgenes_with_weight\tgenes_all\tmass_in_windows\tmass_total\tfrac_outside\n")
            for c, mi, ma in zip(channels, mass_in, mass_all):
                r.write(f"{chrom}\t{c}\t{a.assign}\t{n_chr}\t{n_direct + n_viamap + n_viapos}\t{n_unmatched}\t"
                        f"{len(snps)}\t{n_in}\t{n_rows}\t{genes_with}\t{len(genes_all)}\t{mi:.6g}\t{ma:.6g}\t"
                        f"{(1.0 - mi / ma) if ma > 0 else float('nan'):.4f}\n")


if __name__ == "__main__":
    main()
