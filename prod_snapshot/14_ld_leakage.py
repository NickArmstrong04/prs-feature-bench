#!/usr/bin/env python3
# =============================================================================
#  14_ld_leakage.py -- how much of the PRS-Net node-feature correlation is
#  genomic adjacency (LD / shared SNPs) rather than PPI?
#
#  Diagnostic A from the LD future-direction (future_directions.md §2):
#
#    1. sample N people (default 20,000) from the 12d feature store feats.f32
#    2. for a few P-thresholds, Pearson-correlate every gene's feature across
#       those people for four classes of gene pairs:
#         ppi_far      STRING PPI neighbours NOT within 500 kb (the signal the
#                      GNN is meant to use)
#         ppi_near     STRING PPI neighbours that are ALSO within 500 kb
#         adjacent     same chromosome, windows within 500 kb, NOT a PPI pair
#                      -- split further by whether the two genes share any
#                      index SNP, and by gap distance
#         random       random pairs on different chromosomes (null)
#       Giant windows (> --giant-mb, 30 annotation artefacts) are reported
#       as their own class and excluded from the others.
#    3. tabulates index-SNP sharing from the 12b .clumped files: how many
#       index SNPs appear in 1, 2, 3-5, 6+ genes, with and without giants.
#
#  Reading: 20,000 random rows of a 381k x 218k float32 memmap = 17 GB off
#  NFS, done in row blocks; only the chosen threshold columns are kept
#  (3 thresholds -> ~4.8 GB). Peak RAM ~8 GB; 16 GB / 4 CPU / 2 h is plenty.
#
#  Outputs (--out-dir):
#    summary.tsv      one row per (threshold, class): n, mean r, mean|r|,
#                     median|r|, p90|r|, frac|r|>0.1, frac|r|>0.3
#    pairs.tsv.gz     per pair: class, genes, gap_bp, n_shared_index, r per thr
#    index_snp_sharing.tsv   genes-per-index-SNP histogram
#    sampled_rows.txt        the row indices used (reproducible: --seed)
# =============================================================================
import argparse, glob, gzip, json, os, sys, time
import numpy as np

def log(*a): print(*a, flush=True)
def die(m): print("ERROR: " + m, file=sys.stderr, flush=True); sys.exit(1)

# ----------------------------------------------------------------------------
def read_windows(path, order):
    """chr/start/end per gene in node order; genes without a window -> chr 0."""
    idx = {g: i for i, g in enumerate(order)}
    chrom = np.zeros(len(order), dtype=np.int64)
    start = np.zeros(len(order), dtype=np.int64)
    end = np.zeros(len(order), dtype=np.int64)
    n = 0
    with open(path) as fh:
        hdr = fh.readline().split()
        if hdr[:4] != ["chr", "start", "end", "gene"]:
            die(f"{path}: unexpected header {hdr}")
        for line in fh:
            c, s, e, g = line.rstrip("\n").split("\t")
            if g in idx:
                i = idx[g]; chrom[i] = int(c); start[i] = int(s); end[i] = int(e); n += 1
    log(f"  windows: {n:,} of {len(order):,} nodes have a window")
    return chrom, start, end

def read_ppi(path, order):
    idx = {g: i for i, g in enumerate(order)}
    pairs = set()
    with open(path) as fh:
        hdr = fh.readline().split()
        if hdr[:2] != ["gene_a", "gene_b"]:
            die(f"{path}: expected ggi_edges.tsv (gene_a gene_b score), got {hdr}")
        for line in fh:
            a, b = line.split()[:2]
            if a in idx and b in idx and a != b:
                i, j = idx[a], idx[b]
                pairs.add((i, j) if i < j else (j, i))
    log(f"  PPI pairs (undirected, both genes in node order): {len(pairs):,}")
    return pairs

def adjacent_pairs(chrom, start, end, giant, max_gap):
    """(i, j, gap) for same-chromosome non-giant windows with gap <= max_gap."""
    out = []
    for c in np.unique(chrom):
        if c == 0:
            continue
        ids = np.where((chrom == c) & ~giant)[0]
        ids = ids[np.argsort(start[ids], kind="stable")]
        st, en = start[ids], end[ids]
        for a in range(len(ids)):
            lim = en[a] + max_gap
            b = a + 1
            while b < len(ids) and st[b] <= lim:
                gap = max(0, int(st[b] - en[a]))
                i, j = int(ids[a]), int(ids[b])
                out.append((i, j, gap) if i < j else (j, i, gap))
                b += 1
    return out

def read_index_snps(clumps_root, order):
    """gene -> set(index SNP id) from <clumps_root>/chr*/clumps/<gene>.clumped."""
    files = glob.glob(os.path.join(clumps_root, "chr*", "clumps", "*.clumped"))
    if not files:
        return None
    oset = set(order)
    idx = {}
    t0 = time.time()
    for k, f in enumerate(files):
        g = os.path.basename(f)[:-len(".clumped")]
        if g not in oset:
            continue
        s = set()
        with open(f) as fh:
            fh.readline()
            for line in fh:
                p = line.split()
                if len(p) >= 3:
                    s.add(p[2])
        idx[g] = s
        if (k + 1) % 5000 == 0:
            log(f"    {k + 1:,} / {len(files):,} .clumped files read ({(time.time() - t0) / 60:.1f} min)")
    log(f"  index-SNP sets: {len(idx):,} genes, {sum(len(s) for s in idx.values()):,} entries, "
        f"{len(set().union(*idx.values())) if idx else 0:,} unique SNPs")
    return idx

def sharing_table(index, order, giant, path):
    """Genes-per-index-SNP histogram, all genes and non-giant genes only."""
    gset = {order[i] for i in np.where(giant)[0]}
    rows = []
    for label, keep in (("all_genes", lambda g: True), ("excluding_giants", lambda g: g not in gset)):
        cnt = {}
        n_entries = 0
        for g, s in index.items():
            if not keep(g):
                continue
            n_entries += len(s)
            for snp in s:
                cnt[snp] = cnt.get(snp, 0) + 1
        v = np.array(list(cnt.values()), dtype=np.int64) if cnt else np.zeros(0, dtype=np.int64)
        bins = {"1": int((v == 1).sum()), "2": int((v == 2).sum()),
                "3-5": int(((v >= 3) & (v <= 5)).sum()), "6+": int((v >= 6).sum())}
        rows.append((label, n_entries, len(v), int((v > 1).sum()), bins))
    with open(path, "w") as w:
        w.write("gene_set\tindex_entries\tunique_index_snps\tsnps_in_gt1_gene\tfrac_in_gt1_gene\tin_1\tin_2\tin_3_5\tin_6plus\n")
        for label, ne, nu, ngt, b in rows:
            w.write(f"{label}\t{ne}\t{nu}\t{ngt}\t{ngt / max(nu, 1):.4f}\t{b['1']}\t{b['2']}\t{b['3-5']}\t{b['6+']}\n")
    log("\n  index-SNP sharing (genes per index SNP):")
    for label, ne, nu, ngt, b in rows:
        log(f"    {label:18s} entries {ne:,}  unique {nu:,}  in >1 gene {ngt:,} ({100 * ngt / max(nu, 1):.1f}%)"
            f"  [1: {b['1']:,}  2: {b['2']:,}  3-5: {b['3-5']:,}  6+: {b['6+']:,}]")

# ----------------------------------------------------------------------------
def load_features(feats_dir, n_sample, seed, thr_idx, block):
    meta = json.load(open(os.path.join(feats_dir, "feats.meta.json")))
    n, G, T = meta["n_samples"], meta["n_genes"], meta["n_thresholds"]
    F = np.memmap(os.path.join(feats_dir, "feats.f32"), dtype=np.float32, mode="r", shape=(n, G * T))
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(n, size=min(n_sample, n), replace=False))
    X = [np.empty((len(rows), G), dtype=np.float32) for _ in thr_idx]
    t0 = time.time()
    for b0 in range(0, len(rows), block):
        r = rows[b0:b0 + block]
        blk = np.asarray(F[r])                       # (B, G*T) -- full rows, one read each
        blk = blk.reshape(len(r), G, T)
        for k, t in enumerate(thr_idx):
            X[k][b0:b0 + len(r)] = blk[:, :, t]
        if (b0 // block) % 10 == 0:
            log(f"    rows {b0 + len(r):,} / {len(rows):,} read ({(time.time() - t0) / 60:.1f} min)")
    del F
    return meta, rows, X

def standardize_T(Xt):
    """(n, G) -> (G, n) z-scored rows; mask of genes with zero variance."""
    mu = Xt.mean(axis=0, dtype=np.float64)
    sd = Xt.std(axis=0, dtype=np.float64)
    ok = sd > 0
    Z = np.empty((Xt.shape[1], Xt.shape[0]), dtype=np.float32)
    Z[:] = Xt.T
    Z -= mu[:, None].astype(np.float32)
    Z[ok] /= sd[ok][:, None].astype(np.float32)
    Z[~ok] = 0.0
    return Z, ok

def pair_corr(Z, A, B, batch=4096):
    n = Z.shape[1]
    r = np.empty(len(A), dtype=np.float32)
    for b0 in range(0, len(A), batch):
        a, b = A[b0:b0 + batch], B[b0:b0 + batch]
        r[b0:b0 + len(a)] = np.einsum("ij,ij->i", Z[a], Z[b]) / n
    return r

def describe(r):
    ar = np.abs(r)
    return dict(n=len(r), mean_r=float(r.mean()), mean_abs=float(ar.mean()),
                median_abs=float(np.median(ar)), p90_abs=float(np.percentile(ar, 90)),
                frac_gt_0_1=float((ar > 0.1).mean()), frac_gt_0_3=float((ar > 0.3).mean()))

# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats-dir", required=True, help="12d output: feats.f32, feats.meta.json, info.csv")
    ap.add_argument("--gene-order", required=True)
    ap.add_argument("--edges", required=True, help="ggi_edges.tsv (gene_a gene_b score)")
    ap.add_argument("--windows", required=True, help="gene_windows.tsv (chr start end gene)")
    ap.add_argument("--clumps-root", default="", help="$PNET: holds chr<N>/clumps/<gene>.clumped (optional)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thresholds", default="1e-3 0.05 1", help="labels from feats.meta.json to analyse")
    ap.add_argument("--max-gap-kb", type=int, default=500)
    ap.add_argument("--giant-mb", type=float, default=10)
    ap.add_argument("--n-random", type=int, default=200000)
    ap.add_argument("--block", type=int, default=256, help="rows per memmap read")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()

    order = [l.strip() for l in open(a.gene_order) if l.strip()]
    meta = json.load(open(os.path.join(a.feats_dir, "feats.meta.json")))
    if meta["n_genes"] != len(order):
        die(f"gene order has {len(order)} genes, feats.meta.json says {meta['n_genes']}")
    labels = meta["thresholds"]
    want = a.thresholds.split()
    thr_idx = []
    for w in want:
        if w not in labels:
            die(f"threshold {w} not in feats.meta.json thresholds {labels}")
        thr_idx.append(labels.index(w))
    log(f"  thresholds: {want} -> column offsets {thr_idx} of {len(labels)}")

    chrom, start, end = read_windows(a.windows, order)
    giant = (end - start) > a.giant_mb * 1e6
    log(f"  giant windows (> {a.giant_mb:g} Mb, excluded from ppi/adjacent/random): {int(giant.sum())}")
    ppi = read_ppi(a.edges, order)

    # ---- index SNPs (part 3) and the per-pair sharing count ----------------
    index = read_index_snps(a.clumps_root, order) if a.clumps_root else None
    if index is None:
        log("  no .clumped files found -> skipping the index-SNP sharing table and n_shared_index")
    else:
        sharing_table(index, order, giant, os.path.join(a.out_dir, "index_snp_sharing.tsv"))

    # ---- pair classes -------------------------------------------------------
    adj = adjacent_pairs(chrom, start, end, giant, a.max_gap_kb * 1000)
    adj_set = {(i, j) for i, j, _ in adj}
    gap_of = {(i, j): g for i, j, g in adj}
    cls, A, B, GAP, SH = [], [], [], [], []

    def shared(i, j):
        if index is None:
            return -1
        return len(index.get(order[i], set()) & index.get(order[j], set()))

    for i, j in sorted(ppi):
        if giant[i] or giant[j]:
            c = "ppi_with_giant"
        elif (i, j) in adj_set:
            c = "ppi_near"
        else:
            c = "ppi_far"
        cls.append(c); A.append(i); B.append(j)
        GAP.append(gap_of.get((i, j), -1 if chrom[i] != chrom[j] else -2)); SH.append(shared(i, j))
    for i, j, g in adj:
        if (i, j) in ppi:
            continue
        s = shared(i, j)
        if index is None:
            c = "adjacent"
        else:
            c = "adjacent_shared_snp" if s > 0 else "adjacent_no_shared_snp"
        cls.append(c); A.append(i); B.append(j); GAP.append(g); SH.append(s)
    rng = np.random.default_rng(a.seed + 1)
    ok_nodes = np.where((chrom > 0) & ~giant)[0]
    while sum(c == "random_diff_chr" for c in cls) < a.n_random:
        i, j = rng.choice(ok_nodes, size=2, replace=False)
        if chrom[i] == chrom[j]:
            continue
        i, j = (int(i), int(j)) if i < j else (int(j), int(i))
        cls.append("random_diff_chr"); A.append(i); B.append(j); GAP.append(-1); SH.append(0)
    # giant windows vs everything else on their chromosome
    for gi in np.where(giant)[0]:
        same = np.where((chrom == chrom[gi]) & ~giant)[0]
        for j in rng.choice(same, size=min(200, len(same)), replace=False):
            i, j = (int(gi), int(j)) if gi < j else (int(j), int(gi))
            cls.append("giant_vs_same_chr"); A.append(i); B.append(j); GAP.append(-3); SH.append(shared(i, j))
    cls = np.array(cls); A = np.array(A); B = np.array(B); GAP = np.array(GAP); SH = np.array(SH)
    log(f"  pairs: " + ", ".join(f"{c} {int((cls == c).sum()):,}" for c in np.unique(cls)))

    # ---- features -----------------------------------------------------------
    log(f"  reading {a.n_samples:,} random rows of feats.f32 ...")
    meta, rows, X = load_features(a.feats_dir, a.n_samples, a.seed, thr_idx, a.block)
    np.savetxt(os.path.join(a.out_dir, "sampled_rows.txt"), rows, fmt="%d")
    log(f"  features in memory: {len(rows):,} people x {meta['n_genes']:,} genes x {len(thr_idx)} thresholds "
        f"({(time.time() - t0) / 60:.1f} min)")

    R = np.full((len(cls), len(thr_idx)), np.nan, dtype=np.float32)
    valid_pair = np.ones((len(cls), len(thr_idx)), dtype=bool)
    for k, (t, lab) in enumerate(zip(thr_idx, want)):
        Z, ok = standardize_T(X[k])
        log(f"  P<={lab}: {int(ok.sum()):,} genes with non-zero variance in the sample")
        v = ok[A] & ok[B]
        valid_pair[:, k] = v
        R[v, k] = pair_corr(Z, A[v], B[v])
        del Z

    # ---- summary --------------------------------------------------------------
    adj_classes = [c for c in np.unique(cls) if c.startswith("adjacent")]
    gap_bins = [(0, 0, "overlap"), (1, 50_000, "1-50kb"), (50_001, 250_000, "50-250kb"), (250_001, 10 ** 9, "250-500kb")]
    with open(os.path.join(a.out_dir, "summary.tsv"), "w") as w:
        w.write("threshold\tclass\tn\tmean_r\tmean_abs_r\tmedian_abs_r\tp90_abs_r\tfrac_abs_gt_0.1\tfrac_abs_gt_0.3\n")
        for k, lab in enumerate(want):
            log(f"\n  === P<={lab} ===")
            log(f"  {'class':34s}{'n':>9}{'mean r':>9}{'mean|r|':>9}{'med|r|':>9}{'p90|r|':>9}{'>0.1':>8}{'>0.3':>8}")
            rows_out = [(c, (cls == c) & valid_pair[:, k]) for c in np.unique(cls)]
            if adj_classes:
                m_all = np.isin(cls, adj_classes) & valid_pair[:, k]
                rows_out.append(("adjacent_all", m_all))
                for lo, hi, name in gap_bins:
                    rows_out.append((f"adjacent_all_gap_{name}", m_all & (GAP >= lo) & (GAP <= hi)))
            for c, m in rows_out:
                if m.sum() == 0:
                    continue
                d = describe(R[m, k])
                w.write(f"{lab}\t{c}\t{d['n']}\t{d['mean_r']:.4f}\t{d['mean_abs']:.4f}\t{d['median_abs']:.4f}\t"
                        f"{d['p90_abs']:.4f}\t{d['frac_gt_0_1']:.4f}\t{d['frac_gt_0_3']:.4f}\n")
                log(f"  {c:34s}{d['n']:>9,}{d['mean_r']:>9.3f}{d['mean_abs']:>9.3f}{d['median_abs']:>9.3f}"
                    f"{d['p90_abs']:>9.3f}{d['frac_gt_0_1']:>8.3f}{d['frac_gt_0_3']:>8.3f}")

    with gzip.open(os.path.join(a.out_dir, "pairs.tsv.gz"), "wt") as w:
        w.write("class\tgene_a\tgene_b\tgap_bp\tn_shared_index\t" + "\t".join("r_P<=" + x for x in want) + "\n")
        for p in range(len(cls)):
            w.write(f"{cls[p]}\t{order[A[p]]}\t{order[B[p]]}\t{GAP[p]}\t{SH[p]}\t"
                    + "\t".join("nan" if np.isnan(x) else f"{x:.4f}" for x in R[p]) + "\n")
    log(f"\n  done in {(time.time() - t0) / 60:.1f} min -> {a.out_dir}")

if __name__ == "__main__":
    main()
