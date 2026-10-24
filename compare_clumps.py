#!/usr/bin/env python3
"""compare_clumps.py -- exact comparison of two clumps.tsv files (the clumps layout, README s11).

    python compare_clumps.py <golden_dir_or_file> <candidate_dir_or_file>

Each side is clumps.tsv (gene, n_index, index SNPs comma-joined) or a directory
holding one. PASS (exit 0) when every golden gene is present with the same SET
of index SNPs (order is not compared; plink lists them in P order and so does
clump_genes.py, but the features depend on the set). FAIL (exit 2) otherwise:
missing genes first, then genes whose index sets differ, worst (most
symmetric-difference SNPs) first, with the first few SNPs each way.
"""
import os, sys


def load(p):
    if os.path.isdir(p):
        p = os.path.join(p, "clumps.tsv")
    out = {}
    with open(p) as fh:
        fh.readline()
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if len(f) >= 2 and f[0]:
                out[f[0]] = set(f[2].split(",")) - {""} if len(f) > 2 else set()
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    g, c = load(sys.argv[1]), load(sys.argv[2])
    print(f"golden {sys.argv[1]}: {len(g)} genes, {sum(len(v) for v in g.values())} index SNPs;  "
          f"cand {sys.argv[2]}: {len(c)} genes, {sum(len(v) for v in c.values())} index SNPs")
    missing = [x for x in g if x not in c]
    diffs = []
    for x, s in g.items():
        if x in c and c[x] != s:
            diffs.append((len(s ^ c[x]), x, sorted(s - c[x]), sorted(c[x] - s)))
    diffs.sort(reverse=True)
    if missing:
        print(f"MISSING: {len(missing)} golden genes absent from candidate, e.g. {missing[:5]}")
    for d, x, only_g, only_c in diffs[:10]:
        print(f"{x}: {len(g[x])} vs {len(c[x])} index SNPs; only golden {len(only_g)} e.g. {only_g[:3]}; only cand {len(only_c)} e.g. {only_c[:3]}")
    same = sum(1 for x, s in g.items() if x in c and c[x] == s)
    extra = len([x for x in c if x not in g])
    ok = not missing and not diffs
    print(f"{'PASS' if ok else 'FAIL'}: {len(g)} genes, {same} identical, {len(diffs)} differ, {len(missing)} missing"
          + (f", {extra} extra in candidate" if extra else ""))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
