#!/usr/bin/env python3
"""compare.py -- fidelity check between two standard-layout outputs.

    python compare.py <golden_dir> <candidate_dir> [--rtol 1e-5] [--atol 1e-7]

Both directories hold samples.txt, columns.tsv, scores.f32 (see
run_variant.sh for the contract). Columns are matched on (gene, channel),
samples on IID; order may differ. Exit 0 = PASS, 2 = FAIL.

A column PASSES when every sample satisfies
    |cand - gold| <= atol + rtol * |gold|
(the same shape of tolerance as numpy.allclose; rtol 1e-5 is what the
production `check` uses against plink2's ~7 printed digits).
"""
import argparse, os, sys
import numpy as np


def load(d):
    with open(os.path.join(d, "samples.txt")) as fh:
        iids = [l.split()[1] for l in fh if l.strip()]
    cols = []
    with open(os.path.join(d, "columns.tsv")) as fh:
        hdr = fh.readline().split()
        ig, ic = hdr.index("gene"), hdr.index("channel")
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if f:
                cols.append((f[ig], f[ic]))
    n, m = len(iids), len(cols)
    path = os.path.join(d, "scores.f32")
    size = os.path.getsize(path)
    if size != n * m * 4:
        sys.exit(f"{path}: {size} bytes, expected {n}x{m}x4 = {n*m*4}")
    S = np.memmap(path, dtype=np.float32, mode="r", shape=(n, m))
    return iids, cols, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden"); ap.add_argument("candidate")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-7)
    ap.add_argument("--show", type=int, default=15, help="worst columns to list")
    a = ap.parse_args()

    g_iid, g_cols, G = load(a.golden)
    c_iid, c_cols, C = load(a.candidate)

    # samples: candidate must contain every golden sample
    c_row = {s: i for i, s in enumerate(c_iid)}
    missing_s = [s for s in g_iid if s not in c_row]
    if missing_s:
        print(f"FAIL: {len(missing_s)} golden samples missing from candidate (e.g. {missing_s[:3]})")
        sys.exit(2)
    rows = np.array([c_row[s] for s in g_iid])
    same_order = bool(np.all(rows == np.arange(len(rows))))

    c_col = {c: j for j, c in enumerate(c_cols)}
    missing_c = [c for c in g_cols if c not in c_col]
    extra_c = [c for c in c_cols if c not in set(g_cols)]

    worst = []          # (max_abs, max_rel, n_bad, gene, channel)
    n_bad_cols = 0
    for j, key in enumerate(g_cols):
        if key not in c_col:
            continue
        g = np.asarray(G[:, j], dtype=np.float64)
        c = np.asarray(C[:, c_col[key]], dtype=np.float64)
        if not same_order:
            c = c[rows]
        d = np.abs(c - g)
        tol = a.atol + a.rtol * np.abs(g)
        bad = int(np.count_nonzero(d > tol))
        mabs = float(d.max()) if d.size else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(np.abs(g) > a.atol, d / np.abs(g), 0.0)   # cells below atol are judged by atol alone
        mrel = float(rel.max()) if rel.size else 0.0
        if bad:
            n_bad_cols += 1
        worst.append((mabs, mrel, bad, key[0], key[1]))

    worst.sort(reverse=True)
    n_cmp = len(worst)
    print(f"golden {a.golden}: {len(g_iid)} samples x {len(g_cols)} columns")
    print(f"cand   {a.candidate}: {len(c_iid)} samples x {len(c_cols)} columns")
    print(f"compared {n_cmp} columns on {len(g_iid)} samples  (rtol {a.rtol:g}, atol {a.atol:g}; sample order {'identical' if same_order else 'remapped'})")
    if extra_c:
        print(f"note: {len(extra_c)} extra columns in candidate (not compared), e.g. {extra_c[:3]}")
    print(f"{'gene':<18}{'channel':<14}{'max|diff|':>12}{'max rel':>12}{'bad':>8}")
    for mabs, mrel, bad, gene, ch in worst[: a.show]:
        print(f"{gene:<18}{ch:<14}{mabs:>12.3e}{mrel:>12.3e}{bad:>8}")

    ok = not missing_c and n_bad_cols == 0
    gmax_abs = max((w[0] for w in worst), default=0.0)
    gmax_rel = max((w[1] for w in worst), default=0.0)
    if missing_c:
        print(f"MISSING: {len(missing_c)} golden columns absent from candidate, e.g. {missing_c[:5]}")
    line = (f"{'PASS' if ok else 'FAIL'}: {n_cmp} cols, {n_bad_cols} bad cols, "
            f"{len(missing_c)} missing, max|diff| {gmax_abs:.3e}, max rel {gmax_rel:.3e}")
    print(line)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
