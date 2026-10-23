#!/usr/bin/env python3
"""feats_extract.py -- cut a scores-layout golden for a set of genes out of 12d's feats.f32.

    python feats_extract.py --meta <feats.meta.json> --genes <gene list> --out <dir>
                            [--gene-counts <12c gene_counts.tsv>] [--rows 2048]

feats.f32 (12d) is float32, C order, row = sample in info.csv order, column =
gene_index * n_thresholds + threshold_index, gene_index from gene_order.txt
(meta["gene_order"]). The output is the harness scores layout (README s2):
    samples.txt   FID<TAB>IID  (UKB: FID == IID; info.csv has sample_id only)
    columns.tsv   col gene channel n_snps   channel = "P<=<t>" with meta's threshold labels,
                  gene-major, bins adjacent (the order ct variants write); n_snps from
                  --gene-counts (12c's "n_P<=t" columns) or 0
    scores.f32    (n_samples, n_genes * T) float32
Rows are copied in chunks of --rows (2048 x 218,141 cols = 1.8 GB a chunk for
ad/eur), so reading the whole 332 GB file once at the file system's sequential
rate is the cost (~2 min at 3 GB/s); the output for chr19's 1,525 genes is 25 GB.
Production's 12c chunk scores were removed after 12d, so this is the stage-3
golden for the full cohort (README s11).
"""
import argparse, json, os, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True); ap.add_argument("--genes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gene-counts", default=None); ap.add_argument("--rows", type=int, default=2048)
    a = ap.parse_args()
    t0 = time.time()
    meta = json.load(open(a.meta))
    n, G, T = meta["n_samples"], meta["n_genes"], meta["n_thresholds"]
    labels = [str(t) for t in meta["thresholds"]]
    assert len(labels) == T
    order = [l.strip() for l in open(meta["gene_order"]) if l.strip()]
    if len(order) != G:
        raise SystemExit(f"gene_order has {len(order)} genes, meta says {G}")
    gidx = {g: i for i, g in enumerate(order)}
    genes = [l.split()[0] for l in open(a.genes) if l.strip()]
    missing = [g for g in genes if g not in gidx]
    if missing:
        raise SystemExit(f"{len(missing)} genes not in gene_order, e.g. {missing[:5]}")
    cols = np.asarray([gidx[g] * T + t for g in genes for t in range(T)], dtype=np.int64)
    info_path = meta["info"]
    iids = []
    with open(info_path) as fh:
        fh.readline()
        for l in fh:
            if l.strip():
                iids.append(l.split(",")[0])
    if len(iids) != n:
        raise SystemExit(f"info.csv has {len(iids)} rows, meta says {n}")
    path = meta["feats"]
    if os.path.getsize(path) != n * G * T * 4:
        raise SystemExit(f"{path}: {os.path.getsize(path)} bytes, expected {n} x {G * T} x 4")
    F = np.memmap(path, dtype=np.float32, mode="r", shape=(n, G * T))
    counts = {}
    if a.gene_counts:
        with open(a.gene_counts) as fh:
            hdr = fh.readline().split()
            lab2 = [h[2:] for h in hdr[2:]]                      # "n_P<=1e-5" -> "P<=1e-5"
            for l in fh:
                f = l.split()
                counts[f[0]] = dict(zip(lab2, f[2:]))
    os.makedirs(a.out, exist_ok=True)
    m = len(cols)
    out = np.memmap(os.path.join(a.out, "scores.f32"), dtype=np.float32, mode="w+", shape=(n, m))
    t1 = time.time(); done = 0
    for r0 in range(0, n, a.rows):
        r1 = min(n, r0 + a.rows)
        out[r0:r1] = np.take(np.asarray(F[r0:r1]), cols, axis=1)
        done = r1
        if (r0 // a.rows) % 20 == 0:
            el = time.time() - t1
            print(f"  rows {done}/{n}  {done * G * T * 4 / 1e9 / max(el, 1e-9):.2f} GB/s read  ({el:.0f}s)", flush=True)
    out.flush(); del out
    with open(os.path.join(a.out, "samples.txt"), "w") as w:
        for s in iids:
            w.write(f"{s}\t{s}\n")
    with open(os.path.join(a.out, "columns.tsv"), "w") as w:
        w.write("col\tgene\tchannel\tn_snps\n")
        j = 0
        for g in genes:
            for lab in labels:
                w.write(f"{j}\t{g}\tP<={lab}\t{counts.get(g, {}).get(f'P<={lab}', 0)}\n"); j += 1
    print(f"feats_extract: {n} samples x {m} columns ({len(genes)} genes x {T} thresholds) from {path} "
          f"-> {a.out} ({n * m * 4 / 1e9:.1f} GB) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
