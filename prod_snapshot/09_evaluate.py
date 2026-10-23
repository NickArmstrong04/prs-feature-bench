#!/usr/bin/env python3
"""
09_evaluate.py -- AUROC evaluation of a genome-wide PRS, the way the paper does it.

Standard library only (no numpy/sklearn needed on the cluster).

Protocol (paper l. 165-172):
  * split the cohort 8:1:1 into train / validation / test, STRATIFIED by
    case status so every split has the same case fraction;
  * repeat for three random seeds;
  * for each seed, compute AUROC of every score column on the validation
    split, CHOOSE the column with the highest validation AUROC, and REPORT
    that column's AUROC on the test split.
  Choosing and reporting on the same samples inflates AUROC; the "oracle"
  column (best-on-test) is printed only to show how large that inflation
  would have been. It is not a result.

The score columns are whatever the matrix carries: the ten p-value
thresholds for C+T, the 168 (p, h2, sparse) grid points for LDpred2-grid, a
single column for LDpred2-inf / -auto (then "choosing" is a no-op and the
validation split is simply unused), or PRSice-2's ~1,000 thresholds.

The train split is not used by any of these methods (nothing is fitted on the
target). It is written out so a later model that DOES train something
(e.g. PRS + covariates logistic regression, or PRS-Net) can use exactly the
same partition.

Inputs
  --scores  prs.<...>.txt              (FID IID PRS_<label> ..., header)
  --pheno   <disease>_pheno.tab        (FID IID PHENO, 1=control 2=case, no header)
  --seeds   e.g. 0 1 2
  --method  label for the README (default "PLINK C+T")
  --std-all-columns   write every column to scores_std/ however wide the
                      matrix is (default: above STD_ALL_MAX columns, only the
                      column chosen on validation -- see below)
Outputs (in --out-dir)
  auroc_grid.tsv       seed, split, threshold, n, n_case, auroc, se
  summary.tsv          seed, chosen_threshold, val_auroc, test_auroc, test_se,
                       oracle_threshold, oracle_test_auroc
  splits/seed<k>.tsv   IID, split   (train/val/test) -- reuse for other methods
  scores_std/seed<k>.tsv  IID, PHENO, split, one standardised score column per
                       label (z-scored with the TRAIN split's mean/sd)
  README.md            protocol + the summary table
("threshold" in the file headers is kept for compatibility with 09c; for
LDpred2 it holds the column label.)

Wide matrices (2026-09-21): a full standardised copy of PRSice-2's ~1,000
columns is ~3.5 GB per seed of numbers nobody reads, so above STD_ALL_MAX
columns scores_std/ holds only the column chosen on validation for that
seed. auroc_grid.tsv still has every column. C+T (10) and LDpred2-grid
(168) are below the limit and unchanged.
"""
import argparse, os, random, math, statistics, datetime, collections

STD_ALL_MAX = 200

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--pheno", required=True)
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--out-dir", required=True)
ap.add_argument("--method", default="PLINK C+T")
ap.add_argument("--train-frac", type=float, default=0.8)
ap.add_argument("--val-frac", type=float, default=0.1)
ap.add_argument("--std-all-columns", action="store_true")
ap.add_argument("--splits-dir", default=None,
                help="read train/val/test assignment from DIR/seed<k>.tsv (IID, split) instead of drawing it; "
                     "seeds are then taken from the file names and --seeds is ignored")
a = ap.parse_args()
if a.splits_dir:
    import glob, re
    files = sorted(glob.glob(os.path.join(a.splits_dir, "seed*.tsv")), key=lambda p: int(re.findall(r"seed(\d+)", p)[-1]))
    if not files:
        raise SystemExit(f"ERROR: no seed*.tsv in {a.splits_dir}")
    a.seeds = [int(re.findall(r"seed(\d+)", p)[-1]) for p in files]
    print(f"splits: read from {a.splits_dir} (seeds {a.seeds})")
os.makedirs(os.path.join(a.out_dir, "splits"), exist_ok=True)
os.makedirs(os.path.join(a.out_dir, "scores_std"), exist_ok=True)

# ------------------------------------------------------------------ inputs
pheno = {}
with open(a.pheno) as f:
    for line in f:
        p = line.split()
        if len(p) < 3:
            continue
        pheno[p[1]] = 1 if p[2] == "2" else 0 if p[2] == "1" else None
pheno = {k: v for k, v in pheno.items() if v is not None}

with open(a.scores) as f:
    header = f.readline().split()
    thr_cols = [h for h in header if h.startswith("PRS_")]
    thresholds = [h[4:] for h in thr_cols]
    idx = {h: i for i, h in enumerate(header)}
    col_idx = [idx[c] for c in thr_cols]
    iid_idx = idx["IID"]
    iids, y, X = [], [], []
    n_neg = n_na = n_nopheno = 0
    for line in f:
        p = line.split()
        iid = p[iid_idx]
        if iid.startswith("-"):
            n_neg += 1
            continue
        if iid not in pheno:
            n_nopheno += 1
            continue
        try:
            row = [float(p[i]) for i in col_idx]
        except ValueError:
            n_na += 1
            continue
        iids.append(iid); y.append(pheno[iid]); X.append(row)

scored = set(iids)
missing_scores = [k for k in pheno if k not in scored]
N, K = len(iids), len(thresholds)
n_case = sum(y)
std_all = a.std_all_columns or K <= STD_ALL_MAX
print(f"method: {a.method}")
print(f"scores rows used: {N:,}   cases: {n_case:,} ({100*n_case/N:.2f}%)   score columns: {K}")
print(f"dropped: withdrawn (negative ID) {n_neg:,}; not in phenotype {n_nopheno:,}; NA score {n_na:,}")
print(f"phenotype IDs with no score row: {len(missing_scores):,}")
if not std_all:
    print(f"scores_std/: {K} columns > {STD_ALL_MAX}, so only the chosen column per seed is written (--std-all-columns to override)")
if len(missing_scores) > 0:
    raise SystemExit("ERROR: every phenotyped sample should have been scored (scoring used --keep). Check RUN.")

# ------------------------------------------------------------------ AUROC
def auroc(scores, labels):
    """Mann-Whitney AUROC with midranks for ties; Hanley-McNeil SE."""
    n1 = sum(labels); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2 + 1          # midrank, 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    rsum = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    auc = (rsum - n1 * (n1 + 1) / 2) / (n1 * n0)
    q1 = auc / (2 - auc); q2 = 2 * auc * auc / (1 + auc)
    se = math.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc) + (n0 - 1) * (q2 - auc * auc)) / (n1 * n0))
    return auc, se

# ------------------------------------------------------------------ per seed
grid_rows, summary_rows = [], []
for seed in a.seeds:
    if a.splits_dir:
        assign = {}
        with open(os.path.join(a.splits_dir, f"seed{seed}.tsv")) as fh:
            fh.readline()
            for line in fh:
                p = line.split()
                assign[p[0]] = p[1]
        split = [assign.get(iid) for iid in iids]
        missing = sum(1 for s in split if s is None)
        if missing:
            raise SystemExit(f"ERROR: {missing:,} scored samples are not in {a.splits_dir}/seed{seed}.tsv "
                             "-- the split files and the score matrix must describe the same cohort")
        extra = len(assign) - N
        if extra:
            print(f"  note: {extra:,} samples in the split file have no score row and are ignored")
    else:
        rng = random.Random(seed)
        cases = [i for i in range(N) if y[i] == 1]
        ctrls = [i for i in range(N) if y[i] == 0]
        rng.shuffle(cases); rng.shuffle(ctrls)
        split = [None] * N
        for grp in (cases, ctrls):
            n_tr = int(round(a.train_frac * len(grp)))
            n_va = int(round(a.val_frac * len(grp)))
            for k, i in enumerate(grp):
                split[i] = "train" if k < n_tr else "val" if k < n_tr + n_va else "test"
    parts = {s: [i for i in range(N) if split[i] == s] for s in ("train", "val", "test")}

    res = {}
    for s, ids in parts.items():
        lab = [y[i] for i in ids]
        for k, t in enumerate(thresholds):
            auc, se = auroc([X[i][k] for i in ids], lab)
            res[(s, t)] = (auc, se)
            grid_rows.append((seed, s, t, len(ids), sum(lab), auc, se))

    chosen = max(thresholds, key=lambda t: res[("val", t)][0])
    oracle = max(thresholds, key=lambda t: res[("test", t)][0])
    summary_rows.append((seed, chosen, res[("val", chosen)][0], res[("test", chosen)][0],
                         res[("test", chosen)][1], oracle, res[("test", oracle)][0]))

    # standardise with train mean/sd (does not change AUROC; kept for downstream
    # models). Every column for a small matrix, the chosen one for a wide one.
    std_cols = list(range(K)) if std_all else [thresholds.index(chosen)]
    mu, sd = {}, {}
    for k in std_cols:
        col = [X[i][k] for i in parts["train"]]
        m = statistics.fmean(col); s = statistics.pstdev(col) or 1.0
        mu[k] = m; sd[k] = s

    with open(os.path.join(a.out_dir, "splits", f"seed{seed}.tsv"), "w") as f:
        f.write("IID\tsplit\n")
        for i in range(N):
            f.write(f"{iids[i]}\t{split[i]}\n")
    with open(os.path.join(a.out_dir, "scores_std", f"seed{seed}.tsv"), "w") as f:
        f.write("IID\tPHENO\tsplit\t" + "\t".join("PRS_" + thresholds[k] for k in std_cols) + "\n")
        for i in range(N):
            f.write(f"{iids[i]}\t{y[i]}\t{split[i]}\t" +
                    "\t".join(f"{(X[i][k]-mu[k])/sd[k]:.5f}" for k in std_cols) + "\n")

    print(f"\nseed {seed}: train {len(parts['train']):,} / val {len(parts['val']):,} / test {len(parts['test']):,}")
    # A 168-column grid would print 168 lines per seed; show the top ten by
    # validation AUROC instead, and every column when there are few.
    show = thresholds if K <= 12 else sorted(thresholds, key=lambda t: -res[("val", t)][0])[:10]
    print(f"  {'column':>10} {'val AUROC':>10} {'test AUROC':>11}" + ("   (top 10 of %d by validation)" % K if K > 12 else ""))
    for t in show:
        flag = "  <- chosen on validation" if t == chosen else ""
        print(f"  {t:>10} {res[('val', t)][0]:10.4f} {res[('test', t)][0]:11.4f}{flag}")

# ------------------------------------------------------------------ outputs
with open(os.path.join(a.out_dir, "auroc_grid.tsv"), "w") as f:
    f.write("seed\tsplit\tthreshold\tn\tn_case\tauroc\tse\n")
    for r in grid_rows:
        f.write("\t".join(str(x) if not isinstance(x, float) else f"{x:.6f}" for x in r) + "\n")
with open(os.path.join(a.out_dir, "summary.tsv"), "w") as f:
    f.write("seed\tchosen_threshold\tval_auroc\ttest_auroc\ttest_se\toracle_threshold\toracle_test_auroc\n")
    for r in summary_rows:
        f.write("\t".join(str(x) if not isinstance(x, float) else f"{x:.6f}" for x in r) + "\n")

tests = [r[3] for r in summary_rows]
mean_t = statistics.fmean(tests)
sd_t = statistics.stdev(tests) if len(tests) > 1 else 0.0
chosen_counts = collections.Counter(r[1] for r in summary_rows)

print("\n== RESULT (column chosen on validation, AUROC reported on test)")
print(f"  {'seed':>4} {'chosen':>8} {'val':>8} {'test':>8} {'se':>7}   {'oracle(test-best)':>18}")
for r in summary_rows:
    print(f"  {r[0]:>4} {r[1]:>8} {r[2]:8.4f} {r[3]:8.4f} {r[4]:7.4f}   {r[5]:>8} {r[6]:8.4f}")
print(f"\n  test AUROC over {len(tests)} seeds: mean {mean_t:.4f}  sd {sd_t:.4f}")
print(f"  chosen columns: {dict(chosen_counts)}")

std_note = ("" if std_all else
            f"\nscores_std/ holds only the column chosen on validation for each seed\n"
            f"({K} columns exceed the {STD_ALL_MAX}-column limit for a full copy).\n")
protocol_line = (f"Train/validation/test folds read from {a.splits_dir} (the PRS-Net authors' generate_splits partition, seeds {a.seeds})"
                 if a.splits_dir else
                 f"Stratified {a.train_frac:.0%}/{a.val_frac:.0%}/{1-a.train_frac-a.val_frac:.0%} train/validation/test split,\nseeds {a.seeds}")
readme = f"""# {a.method} PRS evaluation

Generated {datetime.date.today()} by 09_evaluate.py.
Method: {a.method}
Scores: {a.scores}
Phenotype: {a.pheno}
Cohort scored: {N:,} ({n_case:,} cases, {100*n_case/N:.2f}%); withdrawn IDs dropped: {n_neg:,}

## Protocol
{protocol_line}. For each seed the score column ({K} available) is chosen by
validation AUROC and its test AUROC is reported. Nothing is fitted on the
target, so the train split is unused here; it is saved in splits/ so other
methods can share it. AUROC = Mann-Whitney U with midranks; SE = Hanley &
McNeil (1982).

## Result
| seed | chosen column | val AUROC | test AUROC | SE |
|---|---|---|---|---|
""" + "".join(f"| {r[0]} | {r[1]} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} |\n" for r in summary_rows) + f"""
**Test AUROC: {mean_t:.4f} +/- {sd_t:.4f} (mean +/- sd over {len(tests)} seeds)**

"Oracle" (column chosen on test itself, for reference only, NOT a result):
""" + "".join(f"seed {r[0]}: {r[5]} -> {r[6]:.4f}\n" for r in summary_rows) + """
Full grid in auroc_grid.tsv. No covariates (age, sex, PCs) are included;
this is the raw PRS AUROC.
""" + std_note
with open(os.path.join(a.out_dir, "README.md"), "w") as f:
    f.write(readme)
