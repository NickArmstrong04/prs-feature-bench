#!/usr/bin/env python3
"""
prsnet_to_eval.py -- present a PRS-Net training run (train_prsnet.py output)
in the shape 09c_publish.sh indexes, so PRS-Net sits in benchmark_summary
next to C+T / PRSice / LDpred2 scored on the same authors' folds.

  python3 prsnet_to_eval.py --train-dir outputs/prsnet/ad/eur/train \
      --splits-dir data/processed/ad/eur/author_splits --info <12d info.csv> \
      --out-dir data/processed/ad/eur/eval_prsnet_author

Writes
  summary.tsv     seed chosen_threshold val_auroc test_auroc test_se oracle_threshold oracle_test_auroc
                  (chosen = "step<best_step>"; oracle = same -- nothing to choose post hoc)
  auroc_grid.tsv  seed split threshold n n_case auroc se   (one "threshold" row,
                  "prsnet"; train/val rows carry n and n_case with the AUROC
                  as NA except val, which is the best validation AUROC)
  method.txt      prsnet_author<TAB>PRS-Net (authors' splits)
The test AUROC is recomputed from split<k>_test_preds.tsv (must equal
results.tsv to 1e-4) and its Hanley-McNeil SE is derived there.
"""
import argparse, math, os
import numpy as np

def auroc_se(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    n1 = int(labels.sum()); n0 = len(labels) - n1
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    auc = (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    q1 = auc / (2 - auc); q2 = 2 * auc * auc / (1 + auc)
    se = math.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc) + (n0 - 1) * (q2 - auc * auc)) / (n1 * n0))
    return auc, se

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True); ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--info", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    label = {}
    with open(a.info) as fh:
        fh.readline()
        for line in fh:
            f = line.strip().split(","); label[f[0]] = int(f[1])
    rows = []
    with open(os.path.join(a.train_dir, "results.tsv")) as fh:
        hdr = fh.readline().split()
        for line in fh:
            rows.append(dict(zip(hdr, line.split())))
    rows.sort(key=lambda r: int(r["split"]))
    os.makedirs(a.out_dir, exist_ok=True)
    S = open(os.path.join(a.out_dir, "summary.tsv"), "w")
    S.write("seed\tchosen_threshold\tval_auroc\ttest_auroc\ttest_se\toracle_threshold\toracle_test_auroc\n")
    G = open(os.path.join(a.out_dir, "auroc_grid.tsv"), "w")
    G.write("seed\tsplit\tthreshold\tn\tn_case\tauroc\tse\n")
    for r in rows:
        k, seed = int(r["split"]), int(r["seed"])
        counts = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
        with open(os.path.join(a.splits_dir, f"seed{seed}.tsv")) as fh:
            fh.readline()
            for line in fh:
                iid, sp = line.split()
                counts[sp][0] += 1; counts[sp][1] += label[iid]
        preds, labs = [], []
        with open(os.path.join(a.train_dir, f"split{k}_test_preds.tsv")) as fh:
            fh.readline()
            for line in fh:
                sid, y, logit = line.split()
                preds.append(float(logit)); labs.append(int(y))
        assert len(preds) == counts["test"][0], f"split {k}: {len(preds)} test preds vs {counts['test'][0]} in the split file"
        auc, se = auroc_se(preds, labs)
        assert abs(auc - float(r["test_auroc"])) < 1e-4, f"split {k}: recomputed test AUROC {auc:.6f} vs results.tsv {r['test_auroc']}"
        chosen = f"step{r['best_step']}"
        S.write(f"{seed}\t{chosen}\t{float(r['best_val_auroc']):.6f}\t{auc:.6f}\t{se:.6f}\t{chosen}\t{auc:.6f}\n")
        for sp in ("train", "val", "test"):
            n, nc = counts[sp]
            v = {"train": "NA", "val": f"{float(r['best_val_auroc']):.6f}", "test": f"{auc:.6f}"}[sp]
            s_ = {"train": "NA", "val": "NA", "test": f"{se:.6f}"}[sp]
            G.write(f"{seed}\t{sp}\tprsnet\t{n}\t{nc}\t{v}\t{s_}\n")
        print(f"seed {seed} (split {k}): test AUROC {auc:.4f} +/- {se:.4f}  (best step {r['best_step']})")
    S.close(); G.close()
    with open(os.path.join(a.out_dir, "method.txt"), "w") as w:
        w.write("prsnet_author\tPRS-Net (authors' splits)\n")
    t = [float(r["test_auroc"]) for r in rows]
    print(f"PRS-Net: {len(t)} splits, mean {np.mean(t):.4f} sd {np.std(t, ddof=1):.4f} -> {a.out_dir}")

if __name__ == "__main__":
    main()
