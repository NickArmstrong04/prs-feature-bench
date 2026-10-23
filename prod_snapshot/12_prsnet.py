#!/usr/bin/env python3
# =============================================================================
#  12_prsnet.py -- gene-level C+T features for PRS-Net, one pass per chromosome
#
#  The Python half of 12_prsnet.sh. Five sub-commands, called by the shell
#  steps (nothing here loads a module or touches Slurm):
#
#      windows   setup  the 19,831 per-gene BED files of the PRS-Net repo
#                       -> one gene_windows.tsv (chr start end gene)
#      snps      12b    per chromosome: which QC'd SNPs fall in which gene
#                       window (the authors' bedtools-intersect step) ->
#                       genes/<gene>.{snplist,assoc}, gene_manifest.tsv
#      score     12c    per chromosome: ONE pass over the cohort .bed that
#                       produces every gene x threshold SCORE1_AVG the
#                       authors get from ~900 plink2 runs per chromosome
#      assemble  12d    stack 22 chromosomes into the sample x (gene x 11)
#                       matrix in GRAPH NODE ORDER, write info.csv, and
#                       optionally the authors' one-.npy-per-sample layout
#      compare   check  score output vs the authors' own per-gene plink2
#                       .sscore files, gene by gene, threshold by threshold
#                       -- the fidelity test
#
#  WHAT IS BEING REPRODUCED (scripts/data_preprocess/2_data_generation.sh and
#  generate_prsnet_data.py in github.com/lihan97/PRS-Net):
#
#    for every gene g, SNP set S_g = QC'd SNPs with  start_g < BP <= end_g
#        (bedtools intersect of a [BP-1,BP) SNP bed with [start,end) windows)
#    plink --clump on S_g alone, r2 0.5 / 250 kb / p1 1, LD from the TARGET
#        -> index SNPs I_g
#    plink2 --score gwas.QC.txt 3 4 12 header --q-score-range <11 ranges>
#        --extract I_g
#        -> SCORE1_AVG(i, g, t) = sum_{j in I_g, P_j <= T_t} beta_j x_ij
#                                 / (2 * |{j in I_g : P_j <= T_t}|)
#        with x_ij the effect-allele dosage and, for a MISSING genotype,
#        x_ij = 2 * f_j (plink2's default mean imputation; the missing call
#        still counts in the denominator -- `no-mean-imputation` is what
#        would remove it, and the authors do not pass it)
#    a (g, t) cell with no SNP has no .sscore file -> the authors' loader
#        catches the exception and writes 0
#
#  `score` computes exactly that quantity. The per-SNP products beta_j x_ij
#  are the same whichever gene asks for them, so each SNP is read ONCE and its
#  contribution is added to the (gene, p-value bin) cells it belongs to; the
#  11 nested thresholds are a cumulative sum over the 11 bins at the end. The
#  clumping is NOT re-implemented: 12b still runs the authors' PLINK command
#  per gene (against a small per-chromosome .bed), so the index-SNP sets are
#  PLINK's own. `compare` is the proof: it runs the authors' plink2 command on
#  a sample of genes and demands agreement to float precision.
#
#  VERIFIED 2026-10-07 on ad/eur chr19: 43 genes x 11 thresholds = 473 cells,
#  index sets identical, scores within 4.5e-6 relative (plink2 prints ~7
#  significant digits). Two plink2-20250707 behaviours that `compare` now
#  allows for: an empty P-range yields a .sscore with ALLELE_CT 0 / SCORE1_AVG
#  nan for every sample (the authors' plink2 wrote no file), and ALLELE_CT is
#  per sample, excluding that sample's missing calls (the AVG denominator
#  does not exclude them).
#
#  Requires numpy and bed-reader (pip) -- see `12_prsnet.sh setup`.
# =============================================================================
import argparse
import json
import os
import re
import sys
import time

import numpy as np

THRESHOLDS_DEFAULT = "1e-5 1e-4 1e-3 0.01 0.05 0.1 0.2 0.3 0.4 0.5 1"
MISSING_INT8 = -127          # bed-reader's code for a missing genotype

# acc += beta * col, in place. BLAS daxpy (scipy) is ~4x the speed of the
# two-pass numpy version at 381k samples; both give the same double-precision
# result, so scipy is optional.
try:
    from scipy.linalg.blas import daxpy as _daxpy

    def axpy(col, acc, beta, tmp):
        _daxpy(col, acc, a=beta)          # acc is a contiguous float64 row -> updated in place
    AXPY_IMPL = "scipy daxpy"
except Exception:                          # pragma: no cover
    def axpy(col, acc, beta, tmp):
        np.multiply(col, beta, out=tmp)
        np.add(acc, tmp, out=acc)
    AXPY_IMPL = "numpy"

def log(*a):
    print(*a, flush=True)

def die(msg):
    print("ERROR: " + msg, file=sys.stderr, flush=True)
    sys.exit(1)

def parse_thresholds(s):
    t = [float(x) for x in s.split()]
    if t != sorted(t) or len(set(t)) != len(t):
        die("thresholds must be strictly increasing: " + s)
    return np.array(t), s.split()

# -----------------------------------------------------------------------------
#  windows
# -----------------------------------------------------------------------------
def cmd_windows(a):
    """Collapse data/gene_bed_files_10kb/chr*/<gene>.bed into one TSV."""
    rows = []
    for c in range(1, 23):
        d = os.path.join(a.bed_dir, f"chr{c}")
        if not os.path.isdir(d):
            die(f"no such directory: {d}")
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".bed"):
                continue
            gene = fn[:-4]
            with open(os.path.join(d, fn)) as fh:
                lines = [l.rstrip("\n").split("\t") for l in fh if l.strip()]
            if len(lines) != 1:
                die(f"{d}/{fn}: expected exactly one line, found {len(lines)}")
            chrom, start, end = lines[0][0], int(lines[0][1]), int(lines[0][2])
            if chrom != f"chr{c}":
                die(f"{d}/{fn}: chromosome field {chrom} does not match directory chr{c}")
            rows.append((c, start, end, gene))
    names = [r[3] for r in rows]
    if len(set(names)) != len(names):
        die("duplicate gene names across the BED directories")
    rows.sort()
    with open(a.out, "w") as w:
        w.write("chr\tstart\tend\tgene\n")
        for c, s, e, g in rows:
            w.write(f"{c}\t{s}\t{e}\t{g}\n")
    w_sizes = np.array([e - s for _, s, e, _ in rows])
    log(f"  genes: {len(rows)}  (written to {a.out})")
    log(f"  window size (bp, includes the 2 x {a.ext_kb} kb extension): "
        f"median {int(np.median(w_sizes)):,}  max {int(w_sizes.max()):,}")
    big = int((w_sizes > a.big_window_mb * 1e6).sum())
    log(f"  windows wider than {a.big_window_mb:g} Mb: {big}  "
        "(annotation artefacts -- see prsnet.md, 'giant windows')")

# -----------------------------------------------------------------------------
#  helpers shared by snps / score / compare
# -----------------------------------------------------------------------------
def read_windows(path, chrom):
    genes = []   # (start, end, gene), sorted by start
    with open(path) as fh:
        hdr = fh.readline().split()
        if hdr[:4] != ["chr", "start", "end", "gene"]:
            die(f"{path}: unexpected header {hdr}")
        for line in fh:
            c, s, e, g = line.rstrip("\n").split("\t")
            if int(c) == chrom:
                genes.append((int(s), int(e), g))
    genes.sort()
    return genes

def read_gwas(path):
    """gwas.QC.txt: CHR BP SNP A1 A2 N SE P OR INFO MAF BETA, tab, header.
    Returns (header line, dict SNP -> (BP, A1, BETA, P, fields))."""
    out = {}
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n")
        cols = hdr.split("\t")
        want = ["CHR", "BP", "SNP", "A1", "A2", "N", "SE", "P", "OR", "INFO", "MAF", "BETA"]
        if cols != want:
            die(f"{path}: columns are {cols}, expected {want}")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) != 12:
                die(f"{path}: malformed line: {line[:80]}")
            if f[2] in out:
                die(f"{path}: duplicate SNP id {f[2]}")
            out[f[2]] = (int(float(f[1])), f[3], float(f[11]), float(f[7]), f)
    return hdr, out

def read_bim(path):
    """Returns (ids, positions int array, a1, a2) in file order."""
    ids, pos, a1, a2 = [], [], [], []
    with open(path) as fh:
        for line in fh:
            f = line.split()
            ids.append(f[1]); pos.append(int(f[3])); a1.append(f[4]); a2.append(f[5])
    return ids, np.array(pos, dtype=np.int64), a1, a2

def gene_snp_indices(starts, ends, pos_sorted, order):
    """For each window, the bim row indices with start < pos <= end.
    pos_sorted = positions ascending, order = the argsort that produced it."""
    out = []
    for s, e in zip(starts, ends):
        lo = np.searchsorted(pos_sorted, s, side="right")   # first pos > s
        hi = np.searchsorted(pos_sorted, e, side="right")   # first pos > e
        out.append(order[lo:hi])
    return out

def read_clumped_index(path):
    """Index SNPs (column 3) of a PLINK .clumped file; None if absent."""
    if not os.path.isfile(path):
        return None
    ids = []
    with open(path) as fh:
        first = True
        for line in fh:
            f = line.split()
            if first:
                first = False
                if f[:3] != ["CHR", "F", "SNP"]:
                    die(f"{path}: not a .clumped file (header {f[:3]})")
                continue
            if len(f) >= 3:
                ids.append(f[2])
    return ids

# -----------------------------------------------------------------------------
#  snps  (12b, before the PLINK loop)
# -----------------------------------------------------------------------------
def cmd_snps(a):
    chrom = a.chr
    genes = read_windows(a.windows, chrom)
    hdr, gwas = read_gwas(a.gwas)
    ids, pos, _, _ = read_bim(a.bim)
    # Only SNPs in gwas.QC.txt can be scored: the authors intersect gwas.bed,
    # which is built FROM gwas.QC.txt, with the windows. With the 04 .bim this
    # is every row, but check rather than assume.
    in_gwas = np.array([i in gwas for i in ids])
    if not in_gwas.all():
        log(f"  note: {int((~in_gwas).sum())} .bim variants are not in gwas.QC.txt and are ignored")
    ids = [i for i, k in zip(ids, in_gwas) if k]
    pos = pos[in_gwas]
    order = np.argsort(pos, kind="stable")
    pos_sorted = pos[order]
    idx = gene_snp_indices([g[0] for g in genes], [g[1] for g in genes], pos_sorted, order)

    os.makedirs(a.out_dir, exist_ok=True)
    union = np.zeros(len(ids), dtype=bool)
    n_with = n_big = 0
    with open(a.manifest, "w") as man:
        man.write("gene\tstart\tend\tn_snps\tbig\n")
        for (s, e, g), ii in zip(genes, idx):
            n = len(ii)
            big = int(n > a.big_snps)
            man.write(f"{g}\t{s}\t{e}\t{n}\t{big}\n")
            if n == 0:
                continue
            n_with += 1; n_big += big
            union[ii] = True
            ii_sorted = np.sort(ii)              # bim order, as PLINK keeps it
            with open(os.path.join(a.out_dir, g + ".snplist"), "w") as w:
                for k in ii_sorted:
                    w.write(ids[k] + "\n")
            # The authors' per-gene ".assoc": header + the gene's gwas.QC rows,
            # space separated (generate_gene_gwas.py writes ' '.join(...)).
            with open(os.path.join(a.out_dir, g + ".assoc"), "w") as w:
                w.write(" ".join(hdr.split("\t")) + "\n")
                for k in ii_sorted:
                    w.write(" ".join(gwas[ids[k]][4]) + "\n")
    with open(a.union, "w") as w:
        for i, k in zip(ids, union):
            if k:
                w.write(i + "\n")
    log(f"  chr{chrom}: {len(genes)} genes, {n_with} with >= 1 SNP in their window, "
        f"{len(genes) - n_with} with none (-> zero features)")
    log(f"  SNPs in gwas.QC.txt: {len(ids):,}   in at least one window: {int(union.sum()):,} "
        f"({100.0 * union.sum() / max(len(ids), 1):.1f}%)")
    log(f"  genes with more than {a.big_snps:,} SNPs (clumped one at a time): {n_big}")

# -----------------------------------------------------------------------------
#  score  (12c)
# -----------------------------------------------------------------------------
def load_gene_plan(a, thr):
    """Per gene: list of (bim row, bin, beta) for its index SNPs, plus the
    per-SNP flip flag (target A1 is the gwas A2 -> dosage = 2 - x)."""
    hdr, gwas = read_gwas(a.gwas)
    ids, pos, bim_a1, bim_a2 = read_bim(a.bed + ".bim")
    row = {s: i for i, s in enumerate(ids)}
    flip = {}
    plan = []
    n_flip = n_nomatch = n_above = 0
    stat = {}
    with open(a.manifest) as fh:
        hdr_m = fh.readline().split()
        need = ["gene", "start", "end", "n_snps", "big", "n_index", "status"]
        if hdr_m != need:
            die(f"{a.manifest}: columns {hdr_m}, expected {need} (run 12b first)")
        for line in fh:
            g, s, e, n_snps, big, n_index, status = line.split()
            stat[status] = stat.get(status, 0) + 1
            if status == "failed":
                die(f"gene {g}: clumping failed in 12b -- fix or rerun 12b before scoring")
            entries = []
            if status == "ok":
                index = read_clumped_index(os.path.join(a.clumps, g + ".clumped"))
                if index is None:
                    die(f"gene {g}: manifest says ok but {a.clumps}/{g}.clumped is missing")
                if len(index) != int(n_index):
                    die(f"gene {g}: {len(index)} index SNPs in .clumped, manifest says {n_index}")
                for snp in index:
                    if snp not in row:
                        die(f"gene {g}: index SNP {snp} is not in {a.bed}.bim")
                    if snp not in gwas:
                        die(f"gene {g}: index SNP {snp} is not in gwas.QC.txt")
                    r = row[snp]
                    _, ga1, beta, p, _ = gwas[snp]
                    if bim_a1[r] == ga1:
                        f = False
                    elif bim_a2[r] == ga1:
                        f = True; n_flip += 1
                    else:
                        n_nomatch += 1           # plink2 skips such a variant
                        continue
                    if r in flip and flip[r] != f:
                        die(f"SNP {snp}: inconsistent flip state")
                    flip[r] = f
                    b = int(np.searchsorted(thr, p, side="left"))   # ranges are [0, T]
                    if b >= len(thr):
                        n_above += 1
                        continue
                    entries.append((r, b, beta))
            plan.append((g, int(s), int(e), entries))
    return plan, ids, flip, stat, n_flip, n_nomatch, n_above

def cmd_score(a):
    from bed_reader import open_bed
    t0 = time.time()
    thr, thr_lab = parse_thresholds(a.thresholds)
    T = len(thr)
    plan, bim_ids, flip, stat, n_flip, n_nomatch, n_above = load_gene_plan(a, thr)
    log(f"  genes: {len(plan)}  (clumped ok {stat.get('ok', 0)}, no SNP in window {stat.get('nosnp', 0)})")
    log(f"  index SNPs whose target A1 is the gwas A2 (dosage flipped): {n_flip}"
        + ("   <-- 12a used --a1-allele, so this should be 0; check target_data.a1" if n_flip else ""))
    log(f"  index SNPs whose gwas A1 matches neither target allele (skipped, as plink2 does): {n_nomatch}")
    log(f"  index SNPs with P above the largest threshold (never scored): {n_above}")

    bed = open_bed(a.bed + ".bed", count_A1=True, num_threads=a.threads)
    n = bed.iid_count
    if bed.sid_count != len(bim_ids):
        die("bed/bim mismatch")
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "samples.txt"), "w") as w:
        for f, i in zip(bed.fid, bed.iid):
            w.write(f"{f}\t{i}\n")
    log(f"  samples: {n:,}   variants in the cohort bed: {len(bim_ids):,}   (accumulator: {AXPY_IMPL})")

    # --- chunk the genes (position order; a giant window gets its own chunk)
    chunks, cur = [], []
    for item in plan:
        giant = (item[2] - item[1]) > a.giant_window_mb * 1e6
        if cur and (giant or len(cur) >= a.chunk_genes):
            chunks.append(cur); cur = []
        cur.append(item)
        if giant:
            chunks.append(cur); cur = []
    if cur:
        chunks.append(cur)
    log(f"  gene chunks: {len(chunks)}  (<= {a.chunk_genes} genes each; windows > {a.giant_window_mb:g} Mb alone)")

    man = open(os.path.join(a.out_dir, "scores_manifest.tsv"), "w")
    man.write("chunk\tfile\tn_genes\tn_index_snps\n")
    cnt = open(os.path.join(a.out_dir, "gene_counts.tsv"), "w")
    cnt.write("gene\tn_index\t" + "\t".join("n_P<=" + x for x in thr_lab) + "\n")
    total_reads = 0
    for k, chunk in enumerate(chunks):
        G = len(chunk)
        targets = {}                     # bim row -> [(gene_local, bin, beta)]
        for gi, (g, s, e, entries) in enumerate(chunk):
            for r, b, beta in entries:
                targets.setdefault(r, []).append((gi, b, beta))
        snp_rows = np.array(sorted(targets), dtype=np.int64)
        # S[g, bin, sample]: the sample axis is contiguous, so every per-SNP
        # update below is one contiguous axpy over the cohort.
        S = np.zeros((G, T, n), dtype=np.float64)
        C = np.zeros((G, T), dtype=np.int64)
        col = np.empty(n, dtype=np.float64)          # reused buffers: no per-SNP
        tmp = np.empty(n, dtype=np.float64)          # allocation in the hot loop
        for b0 in range(0, len(snp_rows), a.block_snps):
            blk = snp_rows[b0:b0 + a.block_snps]
            X = bed.read(index=np.s_[:, blk], dtype="int8", order="F", num_threads=a.threads)
            total_reads += len(blk)
            for j, r in enumerate(blk):
                r = int(r)
                x = X[:, j]
                miss = x == MISSING_INT8
                n_miss = int(np.count_nonzero(miss))
                nm = n - n_miss
                if nm == 0:
                    continue                 # every call missing: plink2 scores it as 0
                np.copyto(col, x)            # int8 -> float64 dosage of the bim A1
                if n_miss:
                    # mean dosage of the called samples (= 2 * A1 frequency);
                    # the -127 codes are still in col, so add them back
                    mean = (col.sum() + 127.0 * n_miss) / nm
                    np.copyto(col, mean, where=miss)
                if flip[r]:
                    np.subtract(2.0, col, out=col)   # dosage of the other allele (2 - mean for the imputed)
                for gi, b, beta in targets[r]:
                    axpy(col, S[gi, b], beta, tmp)       # S[gi, b]: contiguous row of length n
                    C[gi, b] += 1
        # nested thresholds = cumulative over bins; AVG = SUM / (2 * count)
        np.cumsum(S, axis=1, out=S)
        Cc = np.cumsum(C, axis=1)
        A = np.zeros((n, G, T), dtype=np.float32)
        for gi in range(G):
            inv = np.where(Cc[gi] > 0, 1.0 / (2.0 * np.maximum(Cc[gi], 1)), 0.0)
            A[:, gi, :] = (S[gi] * inv[:, None]).T
        fn = f"scores_chunk{k:03d}.npy"
        np.save(os.path.join(a.out_dir, fn), A)
        with open(os.path.join(a.out_dir, f"genes_chunk{k:03d}.txt"), "w") as w:
            for item in chunk:
                w.write(item[0] + "\n")
        man.write(f"{k}\t{fn}\t{G}\t{len(snp_rows)}\n")
        for gi, item in enumerate(chunk):
            cnt.write(item[0] + "\t" + str(int(Cc[gi, -1])) + "\t"
                      + "\t".join(str(int(x)) for x in Cc[gi]) + "\n")
        log(f"  chunk {k:3d}: {G:4d} genes, {len(snp_rows):8,} index SNPs read, "
            f"{(time.time() - t0) / 60:.1f} min elapsed")
        del S, A
    man.close(); cnt.close()
    log(f"  done: {total_reads:,} SNP columns read in {(time.time() - t0) / 60:.1f} min")

# -----------------------------------------------------------------------------
#  assemble  (12d)
# -----------------------------------------------------------------------------
def load_chunks(run_dir, chroms):
    """gene -> (chunk npy path, local index); checks the sample order."""
    gene_loc, samples0, files = {}, None, []
    for c in chroms:
        d = os.path.join(run_dir, f"chr{c}", "scores")
        mf = os.path.join(d, "scores_manifest.tsv")
        if not os.path.isfile(mf):
            die(f"missing {mf} -- 12c has not completed for chr{c}")
        with open(os.path.join(d, "samples.txt")) as fh:
            samples = [l.split() for l in fh]
        if samples0 is None:
            samples0 = samples
        elif samples != samples0:
            die(f"chr{c}: sample order differs from chr{chroms[0]} -- 12a must use one keep list")
        with open(mf) as fh:
            fh.readline()
            for line in fh:
                k, fn, G, _ = line.split()
                genes = [l.strip() for l in open(os.path.join(d, f"genes_chunk{int(k):03d}.txt"))]
                if len(genes) != int(G):
                    die(f"{d}: chunk {k} gene list length mismatch")
                p = os.path.join(d, fn)
                files.append((p, genes))
                for gi, g in enumerate(genes):
                    if g in gene_loc:
                        die(f"gene {g} appears in two chunks")
                    gene_loc[g] = (p, gi)
    return gene_loc, samples0, files

def npy_save_authors(path, arr):
    """The authors' save(): a .npy v1.0 header padded to 128 bytes. np.save
    produces the identical bytes for these shapes; kept explicit because
    their load() parses the header by byte offset."""
    magic = b"\x93NUMPY\x01\x00v\x00"
    header = ("{'descr': '" + arr.dtype.descr[0][1] + "', 'fortran_order': False, 'shape': "
              + str(arr.shape) + ", }").ljust(127 - len(magic)) + "\n"
    with open(path, "wb") as fh:
        fh.write(magic); fh.write(header.encode("utf-8")); fh.write(arr.tobytes())

def cmd_assemble(a):
    t0 = time.time()
    thr, thr_lab = parse_thresholds(a.thresholds); T = len(thr)
    chroms = [int(c) for c in a.chroms.split(",")] if a.chroms else list(range(1, 23))
    gene_loc, samples, files = load_chunks(a.run_dir, chroms)
    log(f"  chunk files: {len(files)}   genes with features: {len(gene_loc):,}   samples: {len(samples):,}")

    # --- gene order = graph node order --------------------------------------
    order = [l.strip() for l in open(a.gene_order) if l.strip()]
    if len(set(order)) != len(order):
        die("gene order file has duplicate names")
    if a.expect_n_genes and len(order) != a.expect_n_genes:
        die(f"gene order has {len(order)} entries, the graph has {a.expect_n_genes} nodes")
    oset = set(order)
    missing = [g for g in order if g not in gene_loc]
    extra = [g for g in gene_loc if g not in oset]
    log(f"  gene order: {len(order):,} nodes; {len(order) - len(missing):,} have features, "
        f"{len(missing)} have none (all-zero columns); {len(extra)} scored genes are not in the order (dropped)")
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "genes_without_features.txt"), "w") as w:
        w.write("".join(g + "\n" for g in missing))
    with open(os.path.join(a.out_dir, "genes_not_in_order.txt"), "w") as w:
        w.write("".join(g + "\n" for g in extra))
    dest = {g: i for i, g in enumerate(order)}

    # --- samples: those in the phenotype file, in .fam order -----------------
    pheno = {}
    with open(a.pheno) as fh:
        for line in fh:
            f = line.split()
            if not f or f[0] in ("FID", "#FID"):
                continue
            pheno[f[1]] = f[2]
    keep, labels = [], []
    for k, (fid, iid) in enumerate(samples):
        if iid not in pheno or int(iid) <= 0:
            continue
        v = pheno[iid]
        if v not in ("1", "2"):
            die(f"sample {iid}: phenotype {v} is not 1/2")
        keep.append(k); labels.append(1 if v == "2" else 0)
    log(f"  samples kept: {len(keep):,} of {len(samples):,}  (cases {sum(labels):,})")
    with open(os.path.join(a.out_dir, "info.csv"), "w") as w:
        w.write("sample_id,label,ancestry\n")
        for k, y in zip(keep, labels):
            w.write(f"{samples[k][1]},{y},{a.ancestry}\n")
    keep = np.array(keep, dtype=np.int64)

    # --- the matrix ----------------------------------------------------------
    n_feat = len(order) * T
    out_path = os.path.join(a.out_dir, "feats.f32")
    F = None
    prog_path = os.path.join(a.out_dir, "assemble.progress")
    n_done = 0
    if a.export in ("memmap", "both"):
        want_bytes = len(keep) * n_feat * 4
        prog = None
        if os.path.isfile(prog_path) and os.path.isfile(out_path) and os.path.getsize(out_path) == want_bytes:
            prog = json.load(open(prog_path))
            if prog.get("sample_block") == a.sample_block and prog.get("n_samples") == len(keep) \
                    and prog.get("n_feat") == n_feat and 0 < prog.get("n_done", 0) < len(keep):
                n_done = int(prog["n_done"])
            else:
                prog = None
        if prog is not None:
            F = np.memmap(out_path, dtype=np.float32, mode="r+", shape=(len(keep), n_feat))
            log(f"  resuming: {n_done:,} of {len(keep):,} samples already in {out_path}")
        else:
            F = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(len(keep), n_feat))
            if os.path.isfile(prog_path):
                os.remove(prog_path)
                log("  previous progress file did not match this run (block size / sample count) -- starting from zero")
    feat_dir = os.path.join(a.out_dir, "feats")
    if a.export in ("per-sample", "both"):
        os.makedirs(feat_dir, exist_ok=True)
    plans = []
    for p, genes in files:
        src = [i for i, g in enumerate(genes) if g in dest]
        if src:
            plans.append((p, np.array(src), np.array([dest[genes[i]] for i in src])))
    for s0 in range(n_done, len(keep), a.sample_block):
        rows = keep[s0:s0 + a.sample_block]
        B = len(rows)
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        blk = np.zeros((B, len(order), T), dtype=np.float32)
        for p, src, dst in plans:
            arr = np.load(p, mmap_mode="r")
            # one contiguous read of the sample range, then pick the kept rows
            sub = np.asarray(arr[lo:hi])[rows - lo]
            blk[:, dst, :] = sub[:, src, :]
        flat = blk.reshape(B, n_feat)
        if F is not None:
            F[s0:s0 + B] = flat
            F.flush()                                   # bound the dirty page-cache; makes the block durable
            with open(prog_path + ".tmp", "w") as w:
                json.dump({"n_done": s0 + B, "n_samples": len(keep), "n_feat": n_feat,
                           "sample_block": a.sample_block}, w)
            os.replace(prog_path + ".tmp", prog_path)
        if a.export in ("per-sample", "both"):
            for r, row in enumerate(rows):
                npy_save_authors(os.path.join(feat_dir, f"{samples[row][1]}.npy"), flat[r].reshape(1, -1))
        log(f"  samples {s0:,}-{s0 + B:,} written  ({(time.time() - t0) / 60:.1f} min)")
    if F is not None:
        F.flush(); del F
        if os.path.isfile(prog_path):
            os.remove(prog_path)
    meta = {"n_samples": int(len(keep)), "n_genes": len(order), "n_thresholds": T,
            "thresholds": thr_lab, "dtype": "float32",
            "layout": "C order; row = sample (info.csv order), column = gene_index * n_thresholds + threshold_index",
            "gene_order": os.path.abspath(a.gene_order),
            "feats": out_path if a.export != "per-sample" else None,
            "per_sample_dir": feat_dir if a.export != "memmap" else None,
            "info": os.path.join(a.out_dir, "info.csv")}
    with open(os.path.join(a.out_dir, "feats.meta.json"), "w") as w:
        json.dump(meta, w, indent=2)
    log(f"  done in {(time.time() - t0) / 60:.1f} min -> {a.out_dir}")

# -----------------------------------------------------------------------------
#  compare  (check): our chunk output vs the authors' per-gene .sscore files
# -----------------------------------------------------------------------------
def cmd_compare(a):
    thr, thr_lab = parse_thresholds(a.thresholds)
    d = a.scores_dir
    with open(os.path.join(d, "samples.txt")) as fh:
        iids = [l.split()[1] for l in fh]
    row = {s: i for i, s in enumerate(iids)}
    loc = {}
    with open(os.path.join(d, "scores_manifest.tsv")) as fh:
        fh.readline()
        for line in fh:
            k, fn, G, _ = line.split()
            genes = [l.strip() for l in open(os.path.join(d, f"genes_chunk{int(k):03d}.txt"))]
            for gi, g in enumerate(genes):
                loc[g] = (os.path.join(d, fn), gi)
    counts = {}
    with open(os.path.join(d, "gene_counts.tsv")) as fh:
        fh.readline()
        for line in fh:
            f = line.split()
            counts[f[0]] = [int(x) for x in f[2:]]
    genes = [l.strip() for l in open(a.genes) if l.strip()]
    worst_abs = worst_rel = 0.0
    n_cells = n_bad = n_empty_both = 0
    print(f"{'gene':<18}{'thr':>7}{'n_ours':>8}{'n_auth':>8}{'max|diff|':>12}{'max rel':>10}  verdict")
    for g in genes:
        if g not in loc:
            print(f"{g:<18} not scored by 12c (no SNP in its window?)"); continue
        p, gi = loc[g]
        A = np.load(p, mmap_mode="r")[:, gi, :]
        for t, lab in enumerate(thr_lab):
            ss = os.path.join(a.sscore_dir, f"{g}.assoc.clumped.{lab}.sscore")
            ours = np.asarray(A[:, t], dtype=np.float64)
            n_cells += 1
            if not os.path.isfile(ss):
                # no .sscore = no SNP in that range; the authors' loader -> zeros
                ok = counts[g][t] == 0 and not ours.any()
                n_bad += (not ok); n_empty_both += ok
                if not ok:
                    print(f"{g:<18}{lab:>7}{counts[g][t]:>8}{'-':>8}{'':>12}{'':>10}  FAIL: plink2 wrote no file but 12c has SNPs")
                continue
            theirs = np.full(len(iids), np.nan); n_auth = None
            with open(ss) as fh:
                hdr = fh.readline().split()
                ci = hdr.index("IID"); cs = hdr.index("SCORE1_AVG"); ca = hdr.index("ALLELE_CT")
                for line in fh:
                    f = line.split()
                    theirs[row[f[ci]]] = float(f[cs])
                    # ALLELE_CT is per sample and excludes that sample's missing
                    # calls (the AVG denominator does not); the variant count is
                    # the max over samples, not the last row's value
                    n_auth = max(n_auth or 0, int(f[ca]) // 2)
            if n_auth is None or n_auth == 0 or np.isnan(theirs).all():
                # plink2 20250707 writes a .sscore for an EMPTY P-range too, with
                # ALLELE_CT 0 / SCORE1_AVG nan for every sample (the authors'
                # plink2 wrote no file). Same rule as a missing file: 12c must
                # have no SNP there either.
                ok = counts[g][t] == 0 and not ours.any()
                n_bad += (not ok); n_empty_both += ok
                if not ok:
                    print(f"{g:<18}{lab:>7}{counts[g][t]:>8}{0:>8}{'':>12}{'':>10}  FAIL: plink2 .sscore has no rows but 12c has SNPs")
                continue
            if np.isnan(theirs).any():
                die(f"{ss}: {int(np.isnan(theirs).sum())} of our samples are missing from it")
            diff = np.abs(ours - theirs)
            scale = max(float(np.abs(theirs).max()), 1e-12)
            mabs = float(diff.max()); mrel = mabs / scale
            ok = (mrel <= a.rtol) and (n_auth == counts[g][t])
            worst_abs = max(worst_abs, mabs); worst_rel = max(worst_rel, mrel)
            n_bad += (not ok)
            print(f"{g:<18}{lab:>7}{counts[g][t]:>8}{n_auth:>8}{mabs:>12.3e}{mrel:>10.2e}  {'ok' if ok else 'FAIL'}")
    print()
    print(f"cells compared: {n_cells}  (empty on both sides: {n_empty_both})")
    print(f"worst |diff|: {worst_abs:.3e}   worst relative to the gene's max |score|: {worst_rel:.2e}   (tolerance {a.rtol:.0e})")
    print("RESULT: " + ("PASS -- the one-pass scores equal the authors' plink2 SCORE1_AVG"
                        if n_bad == 0 else f"FAIL -- {n_bad} gene x threshold cells differ"))
    sys.exit(0 if n_bad == 0 else 2)

# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="gene-level C+T features for PRS-Net (see header)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("windows", help="collapse the per-gene BED files into gene_windows.tsv")
    p.add_argument("--bed-dir", required=True); p.add_argument("--out", required=True)
    p.add_argument("--ext-kb", type=int, default=10); p.add_argument("--big-window-mb", type=float, default=10)
    p.set_defaults(fn=cmd_windows)

    p = sub.add_parser("snps", help="per chromosome: SNPs per gene window")
    p.add_argument("--chr", type=int, required=True); p.add_argument("--windows", required=True)
    p.add_argument("--gwas", required=True); p.add_argument("--bim", required=True)
    p.add_argument("--out-dir", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--union", required=True); p.add_argument("--big-snps", type=int, default=20000)
    p.set_defaults(fn=cmd_snps)

    p = sub.add_parser("score", help="per chromosome: one-pass gene x threshold scores")
    p.add_argument("--bed", required=True, help="PLINK prefix of the 12a cohort bed")
    p.add_argument("--gwas", required=True); p.add_argument("--clumps", required=True)
    p.add_argument("--manifest", required=True, help="clump_manifest.tsv from 12b")
    p.add_argument("--out-dir", required=True); p.add_argument("--thresholds", default=THRESHOLDS_DEFAULT)
    p.add_argument("--chunk-genes", type=int, default=100); p.add_argument("--block-snps", type=int, default=1024)
    p.add_argument("--giant-window-mb", type=float, default=10); p.add_argument("--threads", type=int, default=4)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("assemble", help="22 chromosomes -> sample x (gene x T) matrix in graph order")
    p.add_argument("--run-dir", required=True, help="$PNET: holds chr<N>/scores/")
    p.add_argument("--gene-order", required=True, help="file: one gene per line, graph node order")
    p.add_argument("--pheno", required=True); p.add_argument("--ancestry", default="EUR")
    p.add_argument("--out-dir", required=True); p.add_argument("--thresholds", default=THRESHOLDS_DEFAULT)
    p.add_argument("--export", choices=["memmap", "per-sample", "both"], default="memmap")
    p.add_argument("--sample-block", type=int, default=20000)
    p.add_argument("--expect-n-genes", type=int, default=0, help="e.g. 19836; 0 = do not check")
    p.add_argument("--chroms", default="", help="comma list, for tests only (default 1-22)")
    p.set_defaults(fn=cmd_assemble)

    p = sub.add_parser("compare", help="12c output vs the authors' per-gene plink2 .sscore files")
    p.add_argument("--scores-dir", required=True); p.add_argument("--sscore-dir", required=True)
    p.add_argument("--genes", required=True, help="file: genes to compare, one per line")
    p.add_argument("--thresholds", default=THRESHOLDS_DEFAULT); p.add_argument("--rtol", type=float, default=1e-5)
    p.set_defaults(fn=cmd_compare)

    a = ap.parse_args()
    a.fn(a)

if __name__ == "__main__":
    main()
