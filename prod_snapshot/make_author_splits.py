#!/usr/bin/env python3
"""
make_author_splits.py -- write the PRS-Net authors' six train/val/test
partitions (generate_splits, seeds 42-47) as split files 09_evaluate.py can
read, so every method is scored on the SAME folds PRS-Net was.

  python3 make_author_splits.py --info <12d info.csv> --out-dir <dir>
      -> <dir>/seed42.tsv ... seed47.tsv   (IID<TAB>split, split in train|val|test)
         <dir>/README.txt

The partition is a deterministic function of the label vector IN info.csv
ORDER (the .fam order of the 12a cohort bed, restricted to phenotyped
samples). train_prsnet.py computes exactly this from the same file, so the
folds written here are the folds PRS-Net trained/validated/tested on. The
code below is the authors' src/utils.py generate_splits, unchanged
(including the in-place shuffle of pos_indexs that carries over between
seeds -- do not "fix" it, it is part of the protocol).
"""
import argparse, os
import numpy as np

SEEDS = [42, 43, 44, 45, 46, 47]

def generate_splits(labels, seeds=SEEDS, train_ratio=0.8, val_ratio=0.1):
    splits = []
    pos_indexs = np.where(labels == 1)[0]
    neg_indexs = np.where(labels == 0)[0]
    for seed in seeds:
        np.random.seed(seed)
        np.random.shuffle(pos_indexs)
        shuffled_indices = np.random.permutation(len(pos_indexs))
        total_length = len(pos_indexs)
        train_size = int(total_length * train_ratio)
        val_size = int(total_length * val_ratio)
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:train_size + val_size]
        test_indices = shuffled_indices[train_size + val_size:]
        train_pos_data = pos_indexs[train_indices]
        val_pos_data = pos_indexs[val_indices]
        test_pos_data = pos_indexs[test_indices]
        train_neg_indices = np.random.choice(neg_indexs, size=int(len(neg_indexs) * 0.8), replace=False)
        val_neg_indices = np.random.choice(np.setdiff1d(neg_indexs, train_neg_indices), size=int(len(neg_indexs) * 0.1), replace=False)
        test_neg_indices = np.setdiff1d(np.setdiff1d(neg_indexs, train_neg_indices), val_neg_indices)
        train_indices = np.concatenate((train_pos_data, train_neg_indices))
        val_indices = np.concatenate((val_pos_data, val_neg_indices))
        test_indices = np.concatenate((test_pos_data, test_neg_indices))
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        splits.append((train_indices, val_indices, test_indices))
    return splits

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--info", required=True, help="12d info.csv: sample_id,label,ancestry")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    ids, labels = [], []
    with open(a.info) as fh:
        hdr = fh.readline().strip().split(",")
        assert hdr[:2] == ["sample_id", "label"], f"unexpected header {hdr}"
        for line in fh:
            f = line.strip().split(",")
            ids.append(f[0]); labels.append(int(f[1]))
    labels = np.array(labels)
    os.makedirs(a.out_dir, exist_ok=True)
    lines = [f"authors' splits from {os.path.abspath(a.info)}  ({len(ids):,} samples, {int(labels.sum()):,} cases)"]
    for seed, (tr, va, te) in zip(SEEDS, generate_splits(labels)):
        assert len(set(tr) | set(va) | set(te)) == len(ids) and len(tr) + len(va) + len(te) == len(ids), "folds do not partition the cohort"
        split = np.empty(len(ids), dtype=object)
        split[tr] = "train"; split[va] = "val"; split[te] = "test"
        with open(os.path.join(a.out_dir, f"seed{seed}.tsv"), "w") as w:
            w.write("IID\tsplit\n")
            for i, s in zip(ids, split):
                w.write(f"{i}\t{s}\n")
        msg = (f"seed {seed}: train {len(tr):,} ({int(labels[tr].sum()):,} cases)  val {len(va):,} ({int(labels[va].sum()):,})  "
               f"test {len(te):,} ({int(labels[te].sum()):,})")
        print(msg); lines.append(msg)
    with open(os.path.join(a.out_dir, "README.txt"), "w") as w:
        w.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
