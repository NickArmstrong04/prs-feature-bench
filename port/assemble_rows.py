#!/usr/bin/env python
"""
port/assemble_rows.py -- production's 12d (12_prsnet.py assemble --export memmap)
split over K jobs that share ONE preallocated feats.f32, each writing a
contiguous slice of the sample rows. The placement is production's: this
module imports load_chunks / parse_thresholds from 12_prsnet.py and copies
cmd_assemble's row loop line for line, so the bytes are the same (verified
genome-wide against production's feats.f32 with cmp, README s11).

Why rows and not chromosomes: gene_order.txt is the 19,831 symbols SORTED
(build_ggi_from_string.py), so a chromosome's genes are scattered across every
row as 44-byte pieces -- a chromosome job cannot write its own columns. Rows
are contiguous, so a slice of samples is one contiguous byte range of the
file and one contiguous byte range of every chunk file.

    assemble_rows.py prep   <common>                 info.csv, genes_*.txt, sparse feats.f32 (truncate), rows.json
    assemble_rows.py rows   <common> --part i --n-parts K [--sample-block 5000]
    assemble_rows.py finish <common> [--cmp /path/to/production/feats.f32]
  <common> = --run-dir R --gene-order F --pheno F --ancestry EUR --thresholds "..."
             --out-dir D --expect-n-genes 19831 --prsnet-script $ROOT/scripts/slurm/12_prsnet.py

prep is idempotent: an existing feats.f32 of the right size is kept (so a
failed part can be resubmitted alone). rows resumes from its own progress
file. finish refuses until every part has written its .done marker.
Only --export memmap is reproduced (production's default); per-sample .npy
export stays with the original.
"""
import argparse, importlib.util, json, os, subprocess, sys, time

T0 = time.time()
def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)
def die(msg):
    print("ERROR: " + msg, file=sys.stderr); sys.exit(1)

def load_prod(path):
    spec = importlib.util.spec_from_file_location("prsnet12", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def common(a):
    """Everything cmd_assemble computes before touching the matrix, in its order."""
    import numpy as np
    P = load_prod(a.prsnet_script)
    thr, thr_lab = P.parse_thresholds(a.thresholds); T = len(thr)
    chroms = [int(c) for c in a.chroms.split(",")] if a.chroms else list(range(1, 23))
    gene_loc, samples, files = P.load_chunks(a.run_dir, chroms)
    order = [l.strip() for l in open(a.gene_order) if l.strip()]
    if len(set(order)) != len(order):
        die("gene order file has duplicate names")
    if a.expect_n_genes and len(order) != a.expect_n_genes:
        die(f"gene order has {len(order)} entries, the graph has {a.expect_n_genes} nodes")
    oset = set(order)
    missing = [g for g in order if g not in gene_loc]
    extra = [g for g in gene_loc if g not in oset]
    dest = {g: i for i, g in enumerate(order)}
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
    keep = np.array(keep, dtype=np.int64)
    plans = []
    for p, genes in files:
        src = [i for i, g in enumerate(genes) if g in dest]
        if src:
            plans.append((p, np.array(src), np.array([dest[genes[i]] for i in src])))
    return dict(T=T, thr_lab=thr_lab, samples=samples, order=order, missing=missing, extra=extra,
                keep=keep, labels=labels, plans=plans, n_feat=len(order) * T,
                out_path=os.path.join(a.out_dir, "feats.f32"))

def blocks(n, sample_block):
    return [(s0, min(s0 + sample_block, n)) for s0 in range(0, n, sample_block)]

def part_blocks(n, sample_block, part, n_parts):
    """Contiguous group of sample blocks for one part (same split for every caller)."""
    bl = blocks(n, sample_block)
    lo = len(bl) * part // n_parts; hi = len(bl) * (part + 1) // n_parts
    return bl[lo:hi]

# -----------------------------------------------------------------------------
def cmd_prep(a):
    c = common(a)
    os.makedirs(a.out_dir, exist_ok=True)
    log(f"chunk files: {len(c['plans'])}   genes with features: {len(c['order']) - len(c['missing']):,}   "
        f"samples kept: {len(c['keep']):,} of {len(c['samples']):,}  (cases {sum(c['labels']):,})")
    with open(os.path.join(a.out_dir, "genes_without_features.txt"), "w") as w:
        w.write("".join(g + "\n" for g in c["missing"]))
    with open(os.path.join(a.out_dir, "genes_not_in_order.txt"), "w") as w:
        w.write("".join(g + "\n" for g in c["extra"]))
    with open(os.path.join(a.out_dir, "info.csv"), "w") as w:
        w.write("sample_id,label,ancestry\n")
        for k, y in zip(c["keep"], c["labels"]):
            w.write(f"{c['samples'][k][1]},{y},{a.ancestry}\n")
    want = int(len(c["keep"])) * c["n_feat"] * 4
    out = c["out_path"]
    if os.path.isfile(out) and os.path.getsize(out) == want and not a.force:
        log(f"keeping existing {out} ({want:,} bytes); parts overwrite their rows")
    else:
        with open(out, "wb") as fh:
            fh.truncate(want)                  # sparse: unwritten columns (genes without features) read as 0
        log(f"created sparse {out} ({want:,} bytes)")
    pdir = os.path.join(a.out_dir, "parts"); os.makedirs(pdir, exist_ok=True)
    for f in os.listdir(pdir):
        os.remove(os.path.join(pdir, f))
    json.dump({"n_samples": int(len(c["keep"])), "n_feat": c["n_feat"], "n_parts": a.n_parts,
               "sample_block": a.sample_block, "bytes": want},
              open(os.path.join(a.out_dir, "rows.json"), "w"), indent=2)
    log(f"prep done: {a.n_parts} parts of {len(blocks(len(c['keep']), a.sample_block))} blocks x {a.sample_block} samples")

# -----------------------------------------------------------------------------
def npy_header(fh):
    """(shape, data offset) of an open .npy; np.save writes v1.0 for these shapes."""
    import numpy as np
    ver = np.lib.format.read_magic(fh)
    rd = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(ver)
    if rd is None:
        die(f"{fh.name}: .npy version {ver} not handled")
    shape, fortran, dtype = rd(fh)
    if dtype != np.dtype("<f4") or len(shape) != 3:
        die(f"{fh.name}: unexpected .npy layout {shape} {dtype} fortran={fortran}")
    return shape, fh.tell(), fortran

# ct_sparse_v5 saves ST.T.astype(f32).reshape(n, G, T): for G > 1 the reshape copies to C order,
# for G == 1 it is a view of the transpose and np.save records fortran_order=True (uc/eur chr16
# chunk 60, 2026-10-24). Production's assembler np.loads either; the row reader below needs C
# order, so such a chunk (n x 11 x 4 = 17 MB) is loaded whole once and sliced from memory.
_FORTRAN_CACHE = {}

def fortran_rows(path, lo, hi):
    import numpy as np
    if path not in _FORTRAN_CACHE:
        _FORTRAN_CACHE[path] = np.ascontiguousarray(np.load(path))
    return _FORTRAN_CACHE[path][lo:hi]

def read_rows(path, lo, hi, G, T):
    """Rows lo:hi of a (n, G, T) float32 .npy as one contiguous read (no mmap, so
    the pages are not charged to this job's RSS the way production's were)."""
    import numpy as np
    with open(path, "rb") as fh:
        shape, start, fortran = npy_header(fh)
        if shape[1:] != (G, T):
            die(f"{path}: shape {shape}, expected (*, {G}, {T})")
        if fortran:
            return fortran_rows(path, lo, hi)
        buf = np.empty((hi - lo, G, T), dtype=np.float32)
        mv = memoryview(buf).cast("B")
        fh.seek(start + lo * G * T * 4)
        got = 0
        while got < len(mv):
            k = fh.readinto(mv[got:])
            if not k:
                die(f"{path}: short read at row {lo} (+{got} bytes)")
            got += k
    return buf

def write_all(fh, mv, offset):
    fh.seek(offset)
    done = 0
    while done < len(mv):
        done += fh.write(mv[done:])

def cmd_rows(a):
    import numpy as np
    c = common(a)
    meta = json.load(open(os.path.join(a.out_dir, "rows.json")))
    if meta["n_samples"] != len(c["keep"]) or meta["n_feat"] != c["n_feat"] or meta["n_parts"] != a.n_parts \
            or meta["sample_block"] != a.sample_block:
        die("rows.json does not match this invocation (run prep again with the same --n-parts/--sample-block)")
    keep, plans, n_feat, T, order = c["keep"], c["plans"], c["n_feat"], c["T"], c["order"]
    my = part_blocks(len(keep), a.sample_block, a.part, a.n_parts)
    pdir = os.path.join(a.out_dir, "parts")
    prog_path = os.path.join(pdir, f"part{a.part:02d}.progress")
    n_done = 0
    if os.path.isfile(prog_path):
        n_done = json.load(open(prog_path)).get("next_s0", 0)
        log(f"resuming part {a.part} at sample {n_done:,}")
    log(f"part {a.part}/{a.n_parts}: samples {my[0][0]:,}-{my[-1][1]:,} in {len(my)} blocks; "
        f"{len(plans)} chunk files, {n_feat:,} columns")
    # G per chunk file, read from the header once
    shapes = {}
    for p, _, _ in plans:
        with open(p, "rb") as fh:
            shapes[p] = npy_header(fh)[0][1]
    nf = sum(1 for p in shapes if npy_header(open(p, "rb"))[2])
    if nf:
        log(f"{nf} chunk file(s) in fortran order (single-gene chunks from ct_sparse_v5): loaded whole, sliced in memory")
    t_read = t_gather = t_write = 0.0
    with open(c["out_path"], "r+b", buffering=0) as out:
        for s0, s1 in my:
            if s0 < n_done:
                continue
            rows = keep[s0:s1]
            B = len(rows)
            lo, hi = int(rows[0]), int(rows[-1]) + 1
            blk = np.zeros((B, len(order), T), dtype=np.float32)
            for p, src, dst in plans:
                t = time.time()
                arr = read_rows(p, lo, hi, shapes[p], T)
                t_read += time.time() - t; t = time.time()
                sub = arr[rows - lo]                       # production: np.asarray(arr[lo:hi])[rows - lo]
                blk[:, dst, :] = sub[:, src, :]
                t_gather += time.time() - t
            flat = blk.reshape(B, n_feat)
            t = time.time()
            write_all(out, memoryview(flat).cast("B"), s0 * n_feat * 4)
            t_write += time.time() - t
            json.dump({"next_s0": s1}, open(prog_path + ".tmp", "w")); os.replace(prog_path + ".tmp", prog_path)
            log(f"  samples {s0:,}-{s1:,} written  (read {t_read:.0f} s, gather {t_gather:.0f} s, write {t_write:.0f} s)")
        t = time.time(); os.fsync(out.fileno()); t_write += time.time() - t
    with open(os.path.join(pdir, f"part{a.part:02d}.done"), "w") as w:
        w.write(f"{my[0][0]}\t{my[-1][1]}\t{time.time() - T0:.0f}\n")
    log(f"part {a.part} done: read {t_read:.0f} s, gather {t_gather:.0f} s, write+fsync {t_write:.0f} s")

# -----------------------------------------------------------------------------
def cmd_finish(a):
    c = common(a)
    meta = json.load(open(os.path.join(a.out_dir, "rows.json")))
    pdir = os.path.join(a.out_dir, "parts")
    done = sorted(f for f in os.listdir(pdir) if f.endswith(".done"))
    if len(done) != meta["n_parts"]:
        die(f"{len(done)} of {meta['n_parts']} parts done: {done}")
    covered = []
    for f in done:
        s0, s1, _ = open(os.path.join(pdir, f)).read().split()
        covered.append((int(s0), int(s1)))
    covered.sort()
    exp = [(b[0][0], b[-1][1]) for b in (part_blocks(meta["n_samples"], meta["sample_block"], i, meta["n_parts"])
                                         for i in range(meta["n_parts"])) if b]
    if covered != exp:
        die(f"parts cover {covered}, expected {exp}")
    if os.path.getsize(c["out_path"]) != meta["bytes"]:
        die("feats.f32 size changed")
    out = {"n_samples": int(len(c["keep"])), "n_genes": len(c["order"]), "n_thresholds": c["T"],
           "thresholds": c["thr_lab"], "dtype": "float32",
           "layout": "C order; row = sample (info.csv order), column = gene_index * n_thresholds + threshold_index",
           "gene_order": os.path.abspath(a.gene_order),
           "feats": c["out_path"], "per_sample_dir": None,
           "info": os.path.join(a.out_dir, "info.csv")}
    with open(os.path.join(a.out_dir, "feats.meta.json"), "w") as w:
        json.dump(out, w, indent=2)
    for f in os.listdir(pdir):
        os.remove(os.path.join(pdir, f))
    os.rmdir(pdir)
    log(f"feats.meta.json written; {meta['n_parts']} parts, {meta['bytes']:,} bytes -> {a.out_dir}")
    if a.cmp:
        log(f"cmp against {a.cmp} ...")
        r = subprocess.run(["cmp", a.cmp, c["out_path"]])
        print("FEATS_IDENTICAL" if r.returncode == 0 else f"FEATS_DIFFER (cmp exit {r.returncode})", flush=True)
        log("cmp done")
        sys.exit(r.returncode)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prep", "rows", "finish"])
    ap.add_argument("--run-dir", required=True); ap.add_argument("--gene-order", required=True)
    ap.add_argument("--pheno", required=True); ap.add_argument("--ancestry", required=True)
    ap.add_argument("--thresholds", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expect-n-genes", type=int, default=0); ap.add_argument("--chroms", default=None)
    ap.add_argument("--prsnet-script", required=True, help="production 12_prsnet.py (load_chunks, parse_thresholds)")
    ap.add_argument("--n-parts", type=int, default=8); ap.add_argument("--part", type=int, default=None)
    ap.add_argument("--sample-block", type=int, default=5000)
    ap.add_argument("--force", action="store_true", help="prep: recreate feats.f32 even if the size matches")
    ap.add_argument("--cmp", default=None, help="finish: cmp feats.f32 against this file")
    a = ap.parse_args()
    if a.step == "rows":
        if a.part is None:
            a.part = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
        if not 0 <= a.part < a.n_parts:
            die(f"--part {a.part} not in 0..{a.n_parts - 1}")
        cmd_rows(a)
    elif a.step == "prep":
        cmd_prep(a)
    else:
        cmd_finish(a)

if __name__ == "__main__":
    main()
