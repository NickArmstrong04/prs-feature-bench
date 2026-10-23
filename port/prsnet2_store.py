#!/usr/bin/env python3
"""prsnet2_store.py -- the trainer's view of a merged shard directory (runs/port_timed).

    from prsnet2_store import DosageStore
    store = DosageStore(data_dir, load="ram", gene_order=PRSNET_GENE_ORDER)
    store.n, store.m                 # people, SNP columns (genome-wide)
    store.labels, store.sample_ids   # from info.csv, in row order (like FeatureStore)
    store.rows(idx)                  # (len(idx), m) float32 torch, missing (3) -> 0 as src/dataset.py does
    store.rows_int8(idx)             # same as int8 numpy, missing kept as 3 unless missing_to_zero
    mi = store.model_inputs()        # what PRSNet2(...) takes:
        mi["n_snps"], mi["n_genes"], mi["gene_ids"], mi["gene2snp_len"], mi["snp_ids"], mi["pvalues"]
        (gene_ids = graph node indices from gene_order.txt, ascending, exactly as the
         authors' train.py sorts gene2snps by key; snp_ids / gene2snp_len follow that order)

What it reads (written by merge_shards.py):
    shards.tsv   chr  X2.npy path  n_snps  col_offset      one 2-bit sample-major shard per chromosome
    samples.txt  FID IID per row            info.csv  sample_id,label,ancestry (same rows)
    pvalues.npy  float64 (m,)                gene2snps.pkl  gene name -> genome-wide column indices

load="ram" (default) copies the packed shards into one (n, sum ceil(n_snps/4)) uint8
array -- 8.2 GB for ad/eur, 380,962 x 21,620 bytes -- so a batch is one row gather plus
22 shift-or unpacks (a 512 x 86,402 batch in ~0.15 s on one core). load="memmap" keeps
the shards on disk and gathers sorted rows from each; fine for evaluation passes.

    python prsnet2_store.py <data_dir> [--gene-order gene_order.txt] [--load ram|memmap]
                            [--export <dir>]        # X.npy int8 (n, m), Y.npy int64, pvalues.npy,
                                                    # gene2snps.pkl (node-index keys) = the authors'
                                                    # example_data layout for their train.py
prints a summary, checks three random rows against x2bit.Sharded, times one 512 batch.
"""
import argparse, os, pickle, sys, time
import numpy as np

sys.path.insert(0, os.environ.get("BENCH_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")))
import x2bit  # noqa: E402


class DosageStore:
    def __init__(self, data_dir, load="ram", gene_order=None, missing_to_zero=True, log=print):
        self.dir = data_dir
        self.missing_to_zero = missing_to_zero
        rows = [l.split("\t") for l in open(os.path.join(data_dir, "shards.tsv")).read().splitlines()[1:]]
        self.shards = [(r[0], self._locate(r[0], r[1]), int(r[2]), int(r[3])) for r in rows]
        self.ms = [m for _, _, m, _ in self.shards]
        self.col_off = np.concatenate([[0], np.cumsum(self.ms)])
        for (_, _, m, o), e in zip(self.shards, self.col_off[:-1]):
            if o != e:
                raise ValueError(f"shards.tsv: col_offset {o} != cumulative {e}; merge_shards.py writes them consecutively")
        self.m = int(self.col_off[-1])
        self.bs = [(m + 3) // 4 for m in self.ms]                      # bytes per row per shard
        self.byte_off = np.concatenate([[0], np.cumsum(self.bs)])
        self.B = int(self.byte_off[-1])

        fam = [l.split() for l in open(os.path.join(data_dir, "samples.txt")).read().splitlines() if l.strip()]
        self.fids = np.array([f[0] for f in fam]); self.iids = np.array([f[1] for f in fam])
        self.n = len(fam)
        info_path = os.path.join(data_dir, "info.csv")
        if os.path.exists(info_path):
            info = np.genfromtxt(info_path, delimiter=",", names=True, dtype=None, encoding=None)
            if len(info) != self.n:
                raise ValueError(f"info.csv has {len(info)} rows, samples.txt {self.n}")
            self.sample_ids = info["sample_id"].astype(str)
            if not np.array_equal(self.sample_ids, self.iids):
                raise ValueError("info.csv sample_id order differs from samples.txt IID order")
            self.labels = info["label"].astype(np.int64)
        else:
            self.sample_ids, self.labels = self.iids, None
        self.pvalues = np.load(os.path.join(data_dir, "pvalues.npy")).astype(np.float32)
        if len(self.pvalues) != self.m:
            raise ValueError(f"pvalues.npy has {len(self.pvalues)} entries, shards.tsv {self.m} SNPs")
        with open(os.path.join(data_dir, "gene2snps.pkl"), "rb") as f:
            self.gene2snps = pickle.load(f)                           # gene name -> [cols]
        self.gene_order = None
        if gene_order:
            self.gene_order = [l.strip() for l in open(gene_order) if l.strip()]
            known = set(self.gene_order)
            missing = [g for g in self.gene2snps if g not in known]
            if missing:
                raise ValueError(f"{len(missing)} genes of gene2snps.pkl are not in {gene_order}: {missing[:5]}")

        self.mmaps = [x2bit.open_x2(p) for _, p, _, _ in self.shards]
        for (c, p, m, _), x, b in zip(self.shards, self.mmaps, self.bs):
            if x.shape != (self.n, b):
                raise ValueError(f"chr{c} {p}: shape {x.shape}, expected ({self.n}, {b})")
        self.load = load
        if load == "ram":
            log(f"loading {self.n} x {self.B} packed bytes ({self.n * self.B / 2**30:.1f} GiB) into RAM ...")
            t0 = time.time()
            self.P = np.empty((self.n, self.B), dtype=np.uint8)
            step = max(1, 2**28 // self.B)                              # ~256 MB per copy
            for x, b0, b1 in zip(self.mmaps, self.byte_off[:-1], self.byte_off[1:]):
                for i in range(0, self.n, step):
                    self.P[i:i + step, b0:b1] = x[i:i + step]
            log(f"  loaded in {time.time() - t0:.0f} s")
            self.mmaps = None
        elif load != "memmap":
            raise ValueError("load must be 'ram' or 'memmap'")

    def _locate(self, chr_, p):
        """shards.tsv paths are as merge_shards.py saw them (often relative to the bench root);
        accept that, data_dir-relative, or the conventional chr<k>/X2.npy under data_dir."""
        for cand in (p, os.path.join(self.dir, p), os.path.join(self.dir, f"chr{chr_}", "X2.npy")):
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(f"chr{chr_}: shard {p} not found (cwd {os.getcwd()}, data_dir {self.dir})")

    # ------------------------------------------------------------------ rows
    def _unpack_packed(self, P):
        """(k, B) packed bytes -> (k, m) int8."""
        out = np.empty((P.shape[0], self.m), dtype=np.int8)
        for m, c0, c1, b0, b1 in zip(self.ms, self.col_off[:-1], self.col_off[1:], self.byte_off[:-1], self.byte_off[1:]):
            out[:, c0:c1] = x2bit.unpack(P[:, b0:b1], m)
        return out

    def rows_int8(self, idx):
        """Rows idx (any order, repeats allowed) as (len(idx), m) int8 codes; 3 -> 0 if missing_to_zero."""
        idx = np.asarray(idx, dtype=np.int64)
        if self.load == "ram":
            out = self._unpack_packed(self.P[idx])
        else:
            order = np.argsort(idx, kind="stable")
            sidx = idx[order]
            got = np.empty((len(idx), self.m), dtype=np.int8)
            for x, m, c0, c1 in zip(self.mmaps, self.ms, self.col_off[:-1], self.col_off[1:]):
                got[:, c0:c1] = x2bit.unpack(x[sidx], m)
            out = np.empty_like(got); out[order] = got
        if self.missing_to_zero:
            out[out == 3] = 0
        return out

    def rows(self, idx):
        """Like FeatureStore.rows: float32 torch tensor in the requested order."""
        import torch
        return torch.from_numpy(self.rows_int8(idx).astype(np.float32))

    # ------------------------------------------------------------------ model inputs
    def model_inputs(self):
        """gene_ids (node indices, ascending), gene2snp_len, snp_ids, pvalues, n_snps, n_genes.
        Needs gene_order; mirrors the authors' train.py: gene2snps sorted by key, snp_ids = concatenated values."""
        if self.gene_order is None:
            raise ValueError("model_inputs needs gene_order (gene_order.txt: node i of the GGI graph is line i)")
        node = {g: i for i, g in enumerate(self.gene_order)}
        g2 = sorted((node[g], list(cols)) for g, cols in self.gene2snps.items())
        gene_ids = [i for i, _ in g2]
        snp_ids = [c for _, cols in g2 for c in cols]
        return {"n_snps": self.m, "n_genes": len(self.gene_order), "gene_ids": gene_ids,
                "gene2snp_len": [len(cols) for _, cols in g2], "snp_ids": snp_ids, "pvalues": self.pvalues,
                "gene_names": [self.gene_order[i] for i in gene_ids]}

    # ------------------------------------------------------------------ export
    def export(self, out_dir, chunk=4096, log=print):
        """The authors' example_data layout: X.npy int8 (n, m) with 3 -> 0 applied by their Dataset
        (kept as 3 here, missing_to_zero is ignored), Y.npy int64, pvalues.npy, gene2snps.pkl keyed by node index."""
        os.makedirs(out_dir, exist_ok=True)
        from numpy.lib.format import open_memmap
        X = open_memmap(os.path.join(out_dir, "X.npy"), mode="w+", dtype=np.int8, shape=(self.n, self.m))
        keep = self.missing_to_zero; self.missing_to_zero = False
        t0 = time.time()
        try:
            for i in range(0, self.n, chunk):
                X[i:i + chunk] = self.rows_int8(np.arange(i, min(self.n, i + chunk)))
        finally:
            self.missing_to_zero = keep
        X.flush(); del X
        log(f"X.npy {self.n} x {self.m} int8 ({self.n * self.m / 2**30:.1f} GiB) in {time.time() - t0:.0f} s")
        if self.labels is not None:
            np.save(os.path.join(out_dir, "Y.npy"), self.labels.astype(np.int64))
        np.save(os.path.join(out_dir, "pvalues.npy"), np.load(os.path.join(self.dir, "pvalues.npy")))
        if self.gene_order is not None:
            node = {g: i for i, g in enumerate(self.gene_order)}
            g2 = {node[g]: [int(c) for c in cols] for g, cols in self.gene2snps.items()}
        else:
            g2 = {g: [int(c) for c in cols] for g, cols in self.gene2snps.items()}
        with open(os.path.join(out_dir, "gene2snps.pkl"), "wb") as f:
            pickle.dump(g2, f)
        log(f"exported to {out_dir}: Y.npy, pvalues.npy, gene2snps.pkl ({len(g2)} genes, "
            f"{'node-index' if self.gene_order else 'gene-name'} keys)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--gene-order", default=None)
    ap.add_argument("--load", choices=["ram", "memmap"], default="ram")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--export", default=None, help="write the authors' X.npy/Y.npy/pvalues.npy/gene2snps.pkl here")
    a = ap.parse_args()
    t0 = time.time()
    st = DosageStore(a.data_dir, load=a.load, gene_order=a.gene_order)
    print(f"{st.n} people x {st.m} SNPs in {len(st.shards)} shards ({st.B} packed bytes/row); "
          f"{len(st.gene2snps)} genes; labels: {'none' if st.labels is None else f'{int(st.labels.sum())} cases'}; opened in {time.time() - t0:.0f} s")
    rng = np.random.default_rng(0)
    chk = rng.choice(st.n, size=3, replace=False)
    ref = x2bit.Sharded([(p, m) for _, p, m, _ in st.shards]).batch(chk)
    got = DosageStore(a.data_dir, load="memmap", missing_to_zero=False, log=lambda *_: None).rows_int8(chk)
    assert np.array_equal(got, ref), "memmap rows differ from x2bit.Sharded"
    if a.load == "ram":
        st.missing_to_zero = False
        assert np.array_equal(st.rows_int8(chk), ref), "ram rows differ from x2bit.Sharded"
        st.missing_to_zero = True
    print(f"rows {chk.tolist()} match x2bit.Sharded; missing cells in them: {int((ref == 3).sum())}")
    idx = rng.integers(0, st.n, size=a.batch)
    st.rows(idx[:2])                                                 # torch import + warm-up, not timed
    t1 = time.time(); b = st.rows(idx); dt = time.time() - t1
    print(f"one batch of {a.batch} ({tuple(b.shape)}, {b.dtype}) in {dt * 1e3:.0f} ms; dosage mean {float(b.mean()):.3f}")
    if a.gene_order:
        mi = st.model_inputs()
        print(f"model_inputs: n_genes {mi['n_genes']}, {len(mi['gene_ids'])} genes with SNPs, "
              f"{len(mi['snp_ids'])} snp_ids, gene_ids[:3] {mi['gene_ids'][:3]} ({', '.join(mi['gene_names'][:3])})")
    if a.export:
        st.export(a.export)


if __name__ == "__main__":
    main()
