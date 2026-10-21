#!/usr/bin/env python3
"""from_chunks.py -- convert 12_prsnet.py `score` output to the harness layout.

    python from_chunks.py <chunks_dir> <out_dir>

<chunks_dir> is what `12_prsnet.py score --out-dir` writes:
    samples.txt            FID<TAB>IID
    scores_manifest.tsv    chunk file n_genes n_index_snps
    genes_chunk<k>.txt     gene names of chunk k, in the chunk's gene order
    scores_chunk<k>.npy    (n_samples, n_genes_k, n_thresholds) float32
    gene_counts.tsv        gene n_index n_P<=<t1> ... n_P<=<tT>

<out_dir> gets samples.txt, columns.tsv, scores.f32 (see run_variant.sh).
Channel labels are the threshold labels from gene_counts.tsv's header
(e.g. "P<=1e-5"), so goldens and variants agree on names by construction.
"""
import os, sys
import numpy as np


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "samples.txt")) as fh:
        samples = [l.rstrip("\n") for l in fh if l.strip()]
    n = len(samples)
    with open(os.path.join(src, "gene_counts.tsv")) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        labels = [h[2:] for h in hdr[2:]]          # "n_P<=1e-5" -> "P<=1e-5"
        counts = {}
        for l in fh:
            f = l.rstrip("\n").split("\t")
            counts[f[0]] = f[2:]
    T = len(labels)
    chunks = []
    with open(os.path.join(src, "scores_manifest.tsv")) as fh:
        fh.readline()
        for l in fh:
            k, fn, G, _ = l.split()
            chunks.append((int(k), fn, int(G)))
    chunks.sort()
    m = sum(G for _, _, G in chunks) * T
    out = np.memmap(os.path.join(dst, "scores.f32"), dtype=np.float32, mode="w+", shape=(n, m))
    col = 0
    with open(os.path.join(dst, "columns.tsv"), "w") as w:
        w.write("col\tgene\tchannel\tn_snps\n")
        for k, fn, G in chunks:
            genes = [l.strip() for l in open(os.path.join(src, f"genes_chunk{k:03d}.txt")) if l.strip()]
            A = np.load(os.path.join(src, fn), mmap_mode="r")
            assert A.shape == (n, len(genes), T), (fn, A.shape, n, len(genes), T)
            out[:, col: col + len(genes) * T] = A.reshape(n, -1)
            for g in genes:
                cnt = counts.get(g, ["0"] * T)
                for t, lab in enumerate(labels):
                    w.write(f"{col}\t{g}\t{lab}\t{cnt[t]}\n")
                    col += 1
    out.flush(); del out
    with open(os.path.join(dst, "samples.txt"), "w") as w:
        w.write("\n".join(samples) + "\n")
    print(f"from_chunks: {n} samples x {col} columns ({col // T} genes x {T} thresholds) -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
