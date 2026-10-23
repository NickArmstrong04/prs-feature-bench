#!/usr/bin/env python3
"""
build_ggi_from_string.py -- rebuild the PRS-Net gene-gene interaction graph
from STRING over our own gene list, with a KNOWN node order (prsnet.md §4,
option 2). Needed because the authors' ggi_graph.bin has no gene names and
the repo never records the order its nodes were built in.

Run on a node that reaches string-db.org (the Bouchet login node does):

  python3 build_ggi_from_string.py --windows $ROOT/ref/prsnet/gene_windows.tsv \
      --out-dir $ROOT/ref/prsnet/string --version 12.0 --score 800
  # then
  cp $ROOT/ref/prsnet/string/gene_order.txt $ROOT/ref/prsnet/gene_order.txt
  # and in config.sh: PRSNET_N_GENES=19831  PRSNET_GRAPH=$ROOT/ref/prsnet/string/ggi_graph.npz

What it does
  1. downloads 9606.protein.links.v<ver>.txt.gz (combined scores) and
     9606.protein.info.v<ver>.txt.gz (Ensembl protein id -> preferred gene symbol)
     into --out-dir (skipped if present)
  2. gene list = the gene symbols in gene_windows.tsv (the 19,831 windowed
     genes), SORTED -> gene_order.txt, line i = node i
  3. keeps links with combined_score >= --score whose both proteins map to a
     symbol in the gene list; symmetric; one protein per symbol in STRING's
     human set, so symbol-level edges are the protein-level edges
  4. writes ggi_graph.npz (src, dst, n_nodes) in the AUTHORS' storage
     convention (every PPI stored twice per direction + one self-loop per
     node, which is what their GIN 'sum' actually aggregates -- see
     prsnet_gnn.md §1), and ggi_edges.tsv (gene_a gene_b score, once per pair)
  5. prints the counts to compare with the authors' graph:
       target  19,836 nodes / 57,578 undirected PPIs / 11,555 isolated nodes / max degree 607
     and a --calibrate mode that prints those counts for several score cutoffs
     and both the full and the physical subnetwork so the closest setting is
     chosen (and recorded) rather than guessed.

Deviation to record in the write-up: node set = our 19,831 windowed genes
(the authors' graph has 19,836, five of which have no BED window in the
repo); STRING version and cutoff as chosen here; PRS-Net's own paper gives
no cutoff, PRSNet-2 says 0.8 (combined) and ships a "ggi_graph_800.bin".
"""
import argparse, gzip, os, sys, urllib.request
import numpy as np

BASE = "https://stringdb-downloads.org/download"

def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"have     {dest}"); return
    print(f"download {url}"); urllib.request.urlretrieve(url, dest + ".part"); os.rename(dest + ".part", dest)

def load_symbols(info_gz):
    sym = {}
    with gzip.open(info_gz, "rt") as fh:
        next(fh)
        for line in fh:
            pid, name = line.split("\t")[:2]
            sym[pid] = name
    return sym

def load_links(links_gz, score_min, sym, genes):
    """-> dict (a,b) sorted-symbol pair -> max score, restricted to genes."""
    edges = {}
    idx = {g: i for i, g in enumerate(genes)}
    with gzip.open(links_gz, "rt") as fh:
        next(fh)
        for line in fh:
            p1, p2, s = line.split()
            s = int(s)
            if s < score_min:
                continue
            a, b = sym.get(p1), sym.get(p2)
            if a is None or b is None or a == b or a not in idx or b not in idx:
                continue
            k = (a, b) if a < b else (b, a)
            if edges.get(k, -1) < s:
                edges[k] = s
    return edges

def summarize(genes, edges):
    n = len(genes); idx = {g: i for i, g in enumerate(genes)}
    deg = np.zeros(n, dtype=int)
    for a, b in edges:
        deg[idx[a]] += 1; deg[idx[b]] += 1
    return {"nodes": n, "undirected_ppi": len(edges), "isolated": int((deg == 0).sum()), "max_degree": int(deg.max())}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True, help="gene_windows.tsv from 12_prsnet.sh setup (chr start end gene)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--version", default="12.0", help="STRING version: 12.0, 11.5, 11.0")
    ap.add_argument("--score", type=int, default=800, help="minimum combined score (0-1000)")
    ap.add_argument("--physical", action="store_true", help="use the physical subnetwork links file instead of the full one")
    ap.add_argument("--calibrate", action="store_true", help="print counts for several cutoffs (and both networks) and exit")
    ap.add_argument("--single-edges", action="store_true", help="store each PPI once per direction (NOT the authors' convention)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    genes = sorted({f[3] for f in (line.split() for line in open(a.windows)) if len(f) >= 4
                    and not f[0].startswith("#") and f[3].lower() != "gene"})
    print(f"genes from windows: {len(genes):,} (sorted; this is the node order)")

    v = a.version
    info = os.path.join(a.out_dir, f"9606.protein.info.v{v}.txt.gz")
    fetch(f"{BASE}/protein.info.v{v}/9606.protein.info.v{v}.txt.gz", info)
    sym = load_symbols(info)
    missing = [g for g in genes if g not in set(sym.values())]
    print(f"genes with no STRING protein of that preferred name: {len(missing):,} (they become isolated nodes)")
    open(os.path.join(a.out_dir, "genes_not_in_string.txt"), "w").write("".join(g + "\n" for g in missing))

    def links_file(physical):
        tag = "protein.physical.links" if physical else "protein.links"
        f = os.path.join(a.out_dir, f"9606.{tag}.v{v}.txt.gz")
        fetch(f"{BASE}/{tag}.v{v}/9606.{tag}.v{v}.txt.gz", f)
        return f

    if a.calibrate:
        print(f"\ntarget (authors' ggi_graph.bin): nodes 19,836  undirected_ppi 57,578  isolated 11,555  max_degree 607\n")
        for physical in (False, True):
            lf = links_file(physical)
            base = load_links(lf, 0, sym, genes)          # all scores, filter below
            for cut in (400, 500, 600, 700, 800, 900, 950):
                e = {k: s for k, s in base.items() if s >= cut}
                print(f"{'physical' if physical else 'full    '} score>={cut:4d}  {summarize(genes, e)}")
        return

    lf = links_file(a.physical)
    edges = load_links(lf, a.score, sym, genes)
    s = summarize(genes, edges)
    print(f"\nSTRING v{v} {'physical' if a.physical else 'full'} score>={a.score}: {s}")
    print("authors' graph:                       nodes 19836, undirected_ppi 57578, isolated 11555, max_degree 607")

    idx = {g: i for i, g in enumerate(genes)}
    src, dst = [], []
    rep = 1 if a.single_edges else 2
    for a_, b_ in edges:
        i, j = idx[a_], idx[b_]
        for _ in range(rep):
            src += [i, j]; dst += [j, i]
    src += list(range(len(genes))); dst += list(range(len(genes)))      # one self-loop per node, as stored by the authors
    np.savez(os.path.join(a.out_dir, "ggi_graph.npz"), src=np.array(src, dtype=np.int64), dst=np.array(dst, dtype=np.int64),
             n_nodes=len(genes))
    with open(os.path.join(a.out_dir, "gene_order.txt"), "w") as w:
        w.write("".join(g + "\n" for g in genes))
    with open(os.path.join(a.out_dir, "ggi_edges.tsv"), "w") as w:
        w.write("gene_a\tgene_b\tcombined_score\n")
        for (a_, b_), sc in sorted(edges.items()):
            w.write(f"{a_}\t{b_}\t{sc}\n")
    with open(os.path.join(a.out_dir, "PROVENANCE.txt"), "w") as w:
        w.write(f"STRING v{v} {'physical' if a.physical else 'full'} links, combined_score >= {a.score}\n"
                f"node set: {len(genes)} genes from {os.path.abspath(a.windows)}, sorted\n"
                f"storage: {'single' if a.single_edges else 'authors convention (each PPI twice per direction)'} + one self-loop per node\n"
                f"{s}\n")
    print(f"wrote {a.out_dir}/{{gene_order.txt, ggi_graph.npz, ggi_edges.tsv, PROVENANCE.txt}}  "
          f"({len(src):,} stored edges)")

if __name__ == "__main__":
    main()
