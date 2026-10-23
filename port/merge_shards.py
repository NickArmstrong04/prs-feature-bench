#!/usr/bin/env python3
"""merge_shards.py -- genome-wide index over per-chromosome dosage shards.

    python merge_shards.py <out_dir> [--chrs 1-22]

Reads <out_dir>/chr<k>/{samples.txt,snps.tsv,pvalues.npy,gene2snps.tsv} written
by dosage_pack.py (or dosage_raw.py) and writes in <out_dir>:
    shards.tsv       chr  path_to_X2.npy  n_snps  col_offset     (for lib/x2bit.Sharded)
    samples.txt      the common sample list (every shard must match exactly)
    snps.tsv         col snp a1 bp p chr, genome-wide column numbering
    pvalues.npy      float64 (total_snps,)
    gene2snps.tsv    gene n_snps snps   and  gene2snps.pkl  gene -> genome-wide column indices
A gene on two chromosomes' lists (should not happen) is an error.
The shard files themselves are left where they are; the trainer opens them with
    X = x2bit.Sharded([(path, n) for path, n in shards])
"""
import argparse, os, pickle, sys
import numpy as np


def parse_chrs(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--chrs", default="1-22")
    a = ap.parse_args()
    chrs = parse_chrs(a.chrs)
    samples = None; shards = []; snps = []; pv = []; g2 = {}; off = 0
    for c in chrs:
        d = os.path.join(a.out_dir, f"chr{c}")
        for f in ("samples.txt", "snps.tsv", "pvalues.npy", "gene2snps.tsv"):
            if not os.path.exists(os.path.join(d, f)):
                sys.exit(f"{d}/{f} missing -- shard chr{c} incomplete")
        x2 = os.path.join(d, "X2.npy")
        if not os.path.exists(x2):
            sys.exit(f"{x2} missing (merge_shards expects the 2-bit store)")
        s = open(os.path.join(d, "samples.txt")).read()
        if samples is None:
            samples = s
        elif s != samples:
            sys.exit(f"chr{c}: samples.txt differs from chr{chrs[0]}'s -- shards must share people and order")
        rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(d, "snps.tsv")).read().splitlines()[1:]]
        p = np.load(os.path.join(d, "pvalues.npy"))
        if len(rows) != len(p):
            sys.exit(f"chr{c}: snps.tsv has {len(rows)} rows, pvalues.npy {len(p)}")
        for r in rows:
            snps.append((r[1], r[2], r[3], r[4], str(c)))
        pv.append(p)
        for l in open(os.path.join(d, "gene2snps.tsv")).read().splitlines()[1:]:
            g, k, ids = l.rstrip("\n").split("\t")
            if g in g2:
                sys.exit(f"gene {g} appears on chr{c} and chr{g2[g][0]}")
            col = {r[1]: j for j, r in enumerate(rows)}
            g2[g] = (c, [off + col[s] for s in ids.split(",")] if ids else [])
        shards.append((c, x2, len(rows), off)); off += len(rows)
    with open(os.path.join(a.out_dir, "shards.tsv"), "w") as w:
        w.write("chr\tx2\tn_snps\tcol_offset\n")
        for c, x2, n, o in shards:
            w.write(f"{c}\t{x2}\t{n}\t{o}\n")
    open(os.path.join(a.out_dir, "samples.txt"), "w").write(samples)
    with open(os.path.join(a.out_dir, "snps.tsv"), "w") as w:
        w.write("col\tsnp\ta1\tbp\tp\tchr\n")
        for j, r in enumerate(snps):
            w.write(f"{j}\t" + "\t".join(r) + "\n")
    np.save(os.path.join(a.out_dir, "pvalues.npy"), np.concatenate(pv))
    g2c = {g: v for g, (_, v) in g2.items()}
    with open(os.path.join(a.out_dir, "gene2snps.pkl"), "wb") as w:
        pickle.dump(g2c, w)
    with open(os.path.join(a.out_dir, "gene2snps.tsv"), "w") as w:
        w.write("gene\tn_snps\tsnps\n")
        for g in sorted(g2c):
            w.write(f"{g}\t{len(g2c[g])}\t" + ",".join(snps[j][0] for j in g2c[g]) + "\n")
    n = len(samples.splitlines())
    print(f"merged {len(shards)} shards: {n} people x {off} SNPs, {len(g2c)} genes, "
          f"{sum(len(v) for v in g2c.values())} links; packed {n * sum((m + 3) // 4 for _, _, m, _ in shards) / 1e9:.1f} GB -> {a.out_dir}")


if __name__ == "__main__":
    main()
