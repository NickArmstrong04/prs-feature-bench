#!/usr/bin/env python3
"""smoke_test.py -- exercises the harness pieces on synthetic data, no PLINK needed.

    python tests/smoke_test.py            (from the bench root)

1. from_chunks.py: a fake 12c chunk layout -> standard layout, values preserved
2. compare.py: PASS on identical, PASS with shuffled rows/cols, FAIL on corruption,
   FAIL on a missing column
3. eqtl_sparse_v0/score.py: synthetic bed (missing calls, weights on both
   alleles) against a brute-force implementation of the plink2 scoresum rule
"""
import os, subprocess, sys, tempfile
import numpy as np

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = [sys.executable]
rng = np.random.default_rng(1)


def run(cmd, ok=0):
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == ok, f"rc {r.returncode} != {ok}\n{cmd}\n{r.stdout}\n{r.stderr}"
    return r.stdout


def write_std(d, samples, cols, S):
    os.makedirs(d, exist_ok=True)
    open(f"{d}/samples.txt", "w").write("".join(f"{s}\t{s}\n" for s in samples))
    with open(f"{d}/columns.tsv", "w") as w:
        w.write("col\tgene\tchannel\tn_snps\n")
        for j, (g, c) in enumerate(cols):
            w.write(f"{j}\t{g}\t{c}\t1\n")
    np.asarray(S, dtype=np.float32).tofile(f"{d}/scores.f32")


W = tempfile.mkdtemp()
n, T = 50, 3
labels = ["1e-5", "0.05", "1"]

# ---- 1. fake chunk layout, 2 chunks of 3 and 2 genes
src = f"{W}/chunks"; os.makedirs(src)
samples = [f"S{i}" for i in range(n)]
open(f"{src}/samples.txt", "w").write("".join(f"{s}\t{s}\n" for s in samples))
genes = [["GA", "GB", "GC"], ["GD", "GE"]]
truth = {}
with open(f"{src}/scores_manifest.tsv", "w") as man, open(f"{src}/gene_counts.tsv", "w") as cnt:
    man.write("chunk\tfile\tn_genes\tn_index_snps\n")
    cnt.write("gene\tn_index\t" + "\t".join("n_P<=" + x for x in labels) + "\n")
    for k, gs in enumerate(genes):
        A = rng.normal(size=(n, len(gs), T)).astype(np.float32)
        np.save(f"{src}/scores_chunk{k:03d}.npy", A)
        open(f"{src}/genes_chunk{k:03d}.txt", "w").write("\n".join(gs) + "\n")
        man.write(f"{k}\tscores_chunk{k:03d}.npy\t{len(gs)}\t100\n")
        for gi, g in enumerate(gs):
            cnt.write(f"{g}\t9\t3\t6\t9\n")
            for t in range(T):
                truth[(g, "P<=" + labels[t])] = A[:, gi, t]
out = f"{W}/std"
print(run(PY + [f"{B}/lib/from_chunks.py", src, out]).strip())
cols = [tuple(l.split("\t")[1:3]) for l in open(f"{out}/columns.tsv").read().splitlines()[1:]]
S = np.fromfile(f"{out}/scores.f32", dtype=np.float32).reshape(n, len(cols))
assert len(cols) == 5 * T
for j, key in enumerate(cols):
    assert np.array_equal(S[:, j], truth[key]), key
print("1. from_chunks: values and labels preserved  OK")

# ---- 2. compare.py
print(run(PY + [f"{B}/compare.py", out, out]).splitlines()[-1])
perm_r = rng.permutation(n); perm_c = rng.permutation(len(cols))
write_std(f"{W}/shuf", [samples[i] for i in perm_r], [cols[j] for j in perm_c], S[perm_r][:, perm_c])
line = run(PY + [f"{B}/compare.py", out, f"{W}/shuf"]).splitlines()[-1]; assert line.startswith("PASS"); print(line, "(shuffled rows+cols)")
S2 = S.copy(); S2[7, 4] += 1e-3
write_std(f"{W}/bad", samples, cols, S2)
line = run(PY + [f"{B}/compare.py", out, f"{W}/bad"], ok=2).splitlines()[-1]; assert line.startswith("FAIL") and "1 bad cols" in line; print(line, "(one corrupted cell)")
write_std(f"{W}/miss", samples, cols[:-1], S[:, :-1])
line = run(PY + [f"{B}/compare.py", out, f"{W}/miss"], ok=2).splitlines()[-1]; assert "1 missing" in line; print(line, "(dropped column)")
print("2. compare: PASS/FAIL behaviour  OK")

# ---- 3. eqtl_sparse_v0 vs brute force
from bed_reader import to_bed
m = 400; ns = 300
G = rng.integers(0, 3, size=(ns, m)).astype(np.float32)
G[rng.random(G.shape) < 0.03] = np.nan                     # 3% missing
G[:, 5] = np.nan                                           # one all-missing SNP
pre = f"{W}/syn"
sid = [f"19:{1000 + i}" for i in range(m)]
a1 = rng.choice(list("ACGT"), m); a2 = np.array([rng.choice([x for x in "ACGT" if x != y]) for y in a1])
to_bed(pre + ".bed", G, properties={"fid": [f"F{i}" for i in range(ns)], "iid": [f"I{i}" for i in range(ns)],
                                    "sid": sid, "chromosome": ["19"] * m, "bp_position": list(range(1000, 1000 + m)),
                                    "allele_1": a1, "allele_2": a2})
# weights: 20 genes x 3 channels, 2-15 SNPs each, half given on a2
wf = f"{W}/w.tsv"; truth = {}
flip_of = rng.random(m) < 0.5                               # one effect allele per SNP (the contract)
with open(wf, "w") as w:
    w.write("gene\tsnp\ta1\tchannel\tbeta\n")
    for g in range(20):
        for ch in ("Tcell", "Bcell", "mono"):
            k = rng.integers(2, 16); js = rng.choice(m, k, replace=False)
            betas = rng.normal(size=k); flip = flip_of[js]
            truth[(f"G{g}", ch)] = (js, betas, flip)
            for j, b, f in zip(js, betas, flip):
                w.write(f"G{g}\t{sid[j]}\t{a2[j] if f else a1[j]}\t{ch}\t{float(b)!r}\n")
    w.write("G0\t19:99999\tA\tTcell\t0.5\n")                 # SNP not in bim -> dropped
    w.write(f"G0\t{sid[1]}\tN\tTcell\t0.5\n")                # bad allele -> dropped
# brute force: plink2 scoresum rule
D = G.astype(np.float64)                                   # dosage of a1, double arithmetic
mean = np.nanmean(D, axis=0); mean = np.where(np.isnan(mean), 0.0, mean)
ii, jj = np.where(np.isnan(D)); D[ii, jj] = mean[jj]
ref_cols, R = [], []
for (g, ch), (js, betas, flip) in truth.items():
    s = np.zeros(ns)
    for j, b, f in zip(js, betas, flip):
        s += b * ((2 - D[:, j]) if f else D[:, j])
    ref_cols.append((g, ch)); R.append(s)
write_std(f"{W}/gold", [f"I{i}" for i in range(ns)], ref_cols, np.array(R).T)
print(run(PY + [f"{B}/variants/eqtl_sparse_v0/score.py", "--bed", pre, "--weights", wf, "--out", f"{W}/v0", "--block", "64", "--threads", "1"]).strip())
line = run(PY + [f"{B}/compare.py", f"{W}/gold", f"{W}/v0"]).splitlines()[-1]; assert line.startswith("PASS"); print(line)
print("3. eqtl_sparse_v0 == brute-force plink2 rule (missing, flipped alleles)  OK")

# ---- 4. weights_to_plink2 wide file consistency (column set, drop counts)
print(run(PY + [f"{B}/eqtl/weights_to_plink2.py", wf, pre + ".bim", f"{W}/wide.tsv"]).strip())
wcols = [tuple(l.split("\t")[1:3]) for l in open(f"{W}/wide.tsv.columns.tsv").read().splitlines()[1:]]
assert set(wcols) == set(ref_cols), "wide file column set"
hdr = open(f"{W}/wide.tsv").readline().split("\t"); assert len(hdr) == 2 + len(wcols)
j0 = int(next(iter(truth.values()))[0][0])                    # a SNP that is in the file
other = a1[j0] if flip_of[j0] else a2[j0]                         # the allele the file did NOT use
open(f"{W}/w_conflict.tsv", "w").write(open(wf).read() + f"G0\t{sid[j0]}\t{other}\tTcell\t0.1\n")
r = subprocess.run(PY + [f"{B}/eqtl/weights_to_plink2.py", f"{W}/w_conflict.tsv", pre + ".bim", f"{W}/wide2.tsv"], capture_output=True, text=True)
assert r.returncode != 0 and "one a1 per SNP" in r.stdout + r.stderr, r.stdout + r.stderr
print("4. weights_to_plink2: 60 columns, 2 rows dropped; conflicting a1 rejected  OK")
# ---- 5. shared kernel, ct mode: clumps -> weights -> nested averages vs brute force
thr = ["1e-5", "1e-3", "0.05", "0.5", "1"]            # 5 thresholds, bins nested
C = f"{W}/clumps"; os.makedirs(C)
with open(f"{W}/gwas.QC.txt", "w") as g, open(f"{W}/SNP.pvalue", "w") as pv:
    g.write("\t".join(f"c{i}" for i in range(1, 13)) + "\n"); pv.write("SNP\tP\n")
    beta_all = rng.normal(size=m); p_all = 10 ** rng.uniform(-7, 0.3, size=m)   # some P > 1 -> above top threshold? no: P<=1 always; use 0.3 to get P>1 rows
    eff_all = [a2[j] if rng.random() < 0.5 else a1[j] for j in range(m)]       # effect allele, either bim allele
    for j in range(m):
        g.write(f"x\tx\t{sid[j]}\t{eff_all[j]}\tx\tx\tx\tx\tx\tx\tx\t{float(beta_all[j])!r}\n")
        pv.write(f"{sid[j]}\t{float(p_all[j])!r}\n")
gene_snps = {}
with open(f"{W}/clump_manifest.tsv", "w") as man:
    man.write("gene\tchr\tstart\tend\tn_snps\tbig\tn_index\tstatus\n")
    for gq in range(12):
        js = sorted(rng.choice(m, int(rng.integers(1, 25)), replace=False))
        gene_snps[f"CG{gq}"] = js
        with open(f"{C}/CG{gq}.clumped", "w") as cf:
            cf.write(" CHR    F         SNP         BP        P    TOTAL   NSIG    S05    S01   S001  S0001    SP2\n")
            for j in js:
                cf.write(f" 19    1  {sid[j]}  {1000+j}  {p_all[j]:.3g}  0 0 0 0 0 0 NONE\n")
        man.write(f"CG{gq}\t19\t0\t0\t{len(js)}\t0\t{len(js)}\tok\n")
    man.write("CG_EMPTY\t19\t0\t0\t0\t0\t0\tnosnp\n")
# brute force 12c: AVG(t) = sum_{j in I_g, P_j<=t} beta_j * dose_j(effect allele) / (2*count), 0 if count 0
R, ref_cols = [], []
for gname in [f"CG{q}" for q in range(12)] + ["CG_EMPTY"]:
    for t in thr:
        js = [j for j in gene_snps.get(gname, []) if p_all[j] <= float(t)]
        sc = np.zeros(ns)
        for j in js:
            dose = D[:, j] if eff_all[j] == a1[j] else 2 - D[:, j]
            sc += beta_all[j] * dose
        R.append(sc / (2 * len(js)) if js else sc); ref_cols.append((gname, f"P<={t}"))
write_std(f"{W}/gold_ct", [f"I{i}" for i in range(ns)], ref_cols, np.array(R).T)
print(run(PY + [f"{B}/lib/make_ct_weights.py", "--clumps", C, "--manifest", f"{W}/clump_manifest.tsv", "--gwas", f"{W}/gwas.QC.txt",
               "--pvalue", f"{W}/SNP.pvalue", "--thresholds", " ".join(thr), "--out", f"{W}/ctw.tsv", "--genes-out", f"{W}/genes_all.txt"]).strip())
print(run(PY + [f"{B}/lib/sparse_score.py", "--bed", pre, "--weights", f"{W}/ctw.tsv", "--out", f"{W}/ct_v0", "--mode", "ct",
               "--thresholds", " ".join(thr), "--all-genes", f"{W}/genes_all.txt", "--block", "50", "--threads", "1"]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold_ct", f"{W}/ct_v0"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
print("5. sparse_score --mode ct == brute-force 12c formula (nested bins, zero gene, both effect alleles)  OK")
# and sum mode of the shared kernel against the same eQTL brute force as step 3
print(run(PY + [f"{B}/lib/sparse_score.py", "--bed", pre, "--weights", wf, "--out", f"{W}/sum_v0", "--mode", "sum", "--block", "64", "--threads", "1"]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold", f"{W}/sum_v0"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
print("6. sparse_score --mode sum == eqtl brute force  OK")

# ct_sparse_v1's chunked kernel: chunks of 2 genes (uneven last chunk) and of 1 column (sum mode)
V1 = f"{B}/variants/ct_sparse_v1/sparse_score.py"
print(run(PY + [V1, "--bed", pre, "--weights", f"{W}/ctw.tsv", "--out", f"{W}/ct_v1", "--mode", "ct",
               "--thresholds", " ".join(thr), "--all-genes", f"{W}/genes_all.txt", "--block", "50", "--threads", "1",
               "--chunk-cols", str(2 * len(thr))]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold_ct", f"{W}/ct_v1"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
assert open(f"{W}/ct_v1/columns.tsv").read() == open(f"{W}/ct_v0/columns.tsv").read(), "v1 columns.tsv differs from v0"
print("7. ct_sparse_v1 chunked kernel --mode ct (2-gene chunks) == brute force, columns.tsv == v0  OK")
print(run(PY + [V1, "--bed", pre, "--weights", wf, "--out", f"{W}/sum_v1", "--mode", "sum", "--block", "64", "--threads", "1",
               "--chunk-cols", "1"]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold", f"{W}/sum_v1"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
print("8. ct_sparse_v1 chunked kernel --mode sum (1-column chunks) == eqtl brute force  OK")

# ct_sparse_v2's packed-bed decode: same two checks, plus a block size that does not divide the SNP count
V2 = f"{B}/variants/ct_sparse_v2/sparse_score.py"
print(run(PY + [V2, "--bed", pre, "--weights", f"{W}/ctw.tsv", "--out", f"{W}/ct_v2", "--mode", "ct",
               "--thresholds", " ".join(thr), "--all-genes", f"{W}/genes_all.txt", "--block", "7", "--threads", "1",
               "--chunk-cols", str(2 * len(thr))]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold_ct", f"{W}/ct_v2"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
assert open(f"{W}/ct_v2/columns.tsv").read() == open(f"{W}/ct_v0/columns.tsv").read(), "v2 columns.tsv differs from v0"
print("9. ct_sparse_v2 packed-bed kernel --mode ct (block 7, 2-gene chunks) == brute force, columns.tsv == v0  OK")
print(run(PY + [V2, "--bed", pre, "--weights", wf, "--out", f"{W}/sum_v2", "--mode", "sum", "--block", "64", "--threads", "1",
               "--chunk-cols", "1"]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold", f"{W}/sum_v2"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
print("10. ct_sparse_v2 packed-bed kernel --mode sum (1-column chunks) == eqtl brute force  OK")

# ct_sparse_v3's row-streamed scatter (transposed chunk result, np.fromfile reads): same two checks
V3 = f"{B}/variants/ct_sparse_v3/sparse_score.py"
print(run(PY + [V3, "--bed", pre, "--weights", f"{W}/ctw.tsv", "--out", f"{W}/ct_v3", "--mode", "ct",
               "--thresholds", " ".join(thr), "--all-genes", f"{W}/genes_all.txt", "--block", "7", "--threads", "1",
               "--chunk-cols", str(2 * len(thr))]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold_ct", f"{W}/ct_v3"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
assert open(f"{W}/ct_v3/columns.tsv").read() == open(f"{W}/ct_v0/columns.tsv").read(), "v3 columns.tsv differs from v0"
print("11. ct_sparse_v3 scatter kernel --mode ct (block 7, 2-gene chunks) == brute force, columns.tsv == v0  OK")
print(run(PY + [V3, "--bed", pre, "--weights", wf, "--out", f"{W}/sum_v3", "--mode", "sum", "--block", "64", "--threads", "1",
               "--chunk-cols", "1"]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold", f"{W}/sum_v3"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
print("12. ct_sparse_v3 scatter kernel --mode sum (1-column chunks) == eqtl brute force  OK")

# ct_sparse_v5: v3's kernel writing 12c's chunk layout; compare.py reads it on either side; from_chunks round-trips it
V5 = f"{B}/variants/ct_sparse_v5/sparse_score.py"
print(run(PY + [V5, "--bed", pre, "--weights", f"{W}/ctw.tsv", "--out", f"{W}/ct_v5", "--mode", "ct",
               "--thresholds", " ".join(thr), "--all-genes", f"{W}/genes_all.txt", "--block", "7", "--threads", "1",
               "--chunk-cols", str(2 * len(thr)), "--out-layout", "chunks"]).splitlines()[-1])
man = [l.split() for l in open(f"{W}/ct_v5/scores_manifest.tsv").read().splitlines()[1:]]
assert [int(r[2]) for r in man][:-1] == [2] * (len(man) - 1) and sum(int(r[2]) for r in man) == len(open(f"{W}/genes_all.txt").read().split()), man
assert not os.path.exists(f"{W}/ct_v5/scores.f32") and not os.path.exists(f"{W}/ct_v5/columns.tsv")
line = run(PY + [f"{B}/compare.py", f"{W}/gold_ct", f"{W}/ct_v5"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line, "(flat golden, chunk candidate)")
line = run(PY + [f"{B}/compare.py", f"{W}/ct_v5", f"{W}/ct_v3"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line, "(chunk golden, flat candidate)")
run(PY + [f"{B}/lib/from_chunks.py", f"{W}/ct_v5", f"{W}/ct_v5_flat"])
assert open(f"{W}/ct_v5_flat/columns.tsv").read() == open(f"{W}/ct_v0/columns.tsv").read(), "from_chunks(v5) columns.tsv differs from v0 (gene_counts.tsv cumulative counts)"
assert np.array_equal(np.fromfile(f"{W}/ct_v5_flat/scores.f32", dtype=np.float32), np.fromfile(f"{W}/ct_v3/scores.f32", dtype=np.float32)), "v5 chunks != v3 flat bit for bit"
gc = [l.split("\t") for l in open(f"{W}/ct_v5/gene_counts.tsv").read().splitlines()]
assert gc[0][:2] == ["gene", "n_index"] and gc[0][2] == "n_" + thr_lab0 if (thr_lab0 := f"P<={thr[0]}") else False
print("13a. ct_sparse_v5 --out-layout chunks == v3 bit for bit via from_chunks; compare.py reads chunk layout as golden or candidate  OK")

# eqtl_sparse_v1 is the v3 kernel in --mode sum: its copy must not drift
import filecmp
assert filecmp.cmp(V3, f"{B}/variants/eqtl_sparse_v1/sparse_score.py", shallow=False), "eqtl_sparse_v1/sparse_score.py differs from ct_sparse_v3's"
print("13. eqtl_sparse_v1/sparse_score.py identical to ct_sparse_v3's  OK")

# eqtl_sparse_v2's per-gene GEMM: both modes, tiny blocks so genes straddle SNP blocks, tiny chunks
V4 = f"{B}/variants/eqtl_sparse_v2/sparse_score.py"
print(run(PY + [V4, "--bed", pre, "--weights", wf, "--out", f"{W}/sum_v4", "--mode", "sum", "--block", "5", "--threads", "1",
               "--chunk-cols", "7"]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold", f"{W}/sum_v4"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
print("14. eqtl_sparse_v2 per-gene GEMM --mode sum (block 5, 7-column chunks) == eqtl brute force  OK")
print(run(PY + [V4, "--bed", pre, "--weights", f"{W}/ctw.tsv", "--out", f"{W}/ct_v4", "--mode", "ct",
               "--thresholds", " ".join(thr), "--all-genes", f"{W}/genes_all.txt", "--block", "7", "--threads", "1",
               "--chunk-cols", str(2 * len(thr))]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold_ct", f"{W}/ct_v4"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
assert open(f"{W}/ct_v4/columns.tsv").read() == open(f"{W}/ct_v0/columns.tsv").read(), "v4 columns.tsv differs from v0"
print("15. eqtl_sparse_v2 per-gene GEMM --mode ct (block 7, 2-gene chunks) == brute force, columns.tsv == v0  OK")

# eqtl_sparse_v3: v2's GEMM with byte-table stats, preallocated per-gene buffers, genes by position.
# sum mode on a gene-shuffled copy of the weights (exercises the position sort), bit-identical to v2.
V5 = f"{B}/variants/eqtl_sparse_v3/sparse_score.py"
lines = open(wf).read().splitlines(); hdr, body = lines[0], lines[1:]
rng.shuffle(body); open(f"{W}/w_shuf.tsv", "w").write(hdr + "\n" + "\n".join(body) + "\n")
print(run(PY + [V5, "--bed", pre, "--weights", f"{W}/w_shuf.tsv", "--out", f"{W}/sum_v5", "--mode", "sum", "--block", "5", "--threads", "1",
               "--chunk-cols", "7"]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold", f"{W}/sum_v5"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
line = run(PY + [f"{B}/compare.py", f"{W}/sum_v4", f"{W}/sum_v5", "--rtol", "0", "--atol", "0"]).splitlines()[-1]; assert line.startswith("PASS"), line
print("16. eqtl_sparse_v3 --mode sum (shuffled weights, block 5, 7-column chunks) == brute force, bit-identical to v2  OK")
print(run(PY + [V5, "--bed", pre, "--weights", f"{W}/ctw.tsv", "--out", f"{W}/ct_v5", "--mode", "ct",
               "--thresholds", " ".join(thr), "--all-genes", f"{W}/genes_all.txt", "--block", "7", "--threads", "1",
               "--chunk-cols", str(2 * len(thr))]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold_ct", f"{W}/ct_v5"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
assert open(f"{W}/ct_v5/columns.tsv").read() == open(f"{W}/ct_v0/columns.tsv").read(), "v5 columns.tsv differs from v0"
print("17. eqtl_sparse_v3 --mode ct (block 7, 2-gene chunks) == brute force, columns.tsv == v0  OK")

# eqtl_sparse_v4: v3 with the sample-major chunk result (GEMM into S[:, cols], contiguous write), both modes
V6 = f"{B}/variants/eqtl_sparse_v4/sparse_score.py"
print(run(PY + [V6, "--bed", pre, "--weights", f"{W}/w_shuf.tsv", "--out", f"{W}/sum_v6", "--mode", "sum", "--block", "5", "--threads", "1",
               "--chunk-cols", "7"]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold", f"{W}/sum_v6"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
line = run(PY + [f"{B}/compare.py", f"{W}/sum_v5", f"{W}/sum_v6", "--rtol", "0", "--atol", "0"]).splitlines()[-1]; assert line.startswith("PASS"), line
print("18. eqtl_sparse_v4 --mode sum (shuffled weights, block 5, 7-column chunks) == brute force, bit-identical to v3  OK")
print(run(PY + [V6, "--bed", pre, "--weights", f"{W}/ctw.tsv", "--out", f"{W}/ct_v6", "--mode", "ct",
               "--thresholds", " ".join(thr), "--all-genes", f"{W}/genes_all.txt", "--block", "7", "--threads", "1",
               "--chunk-cols", str(2 * len(thr))]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold_ct", f"{W}/ct_v6"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
assert open(f"{W}/ct_v6/columns.tsv").read() == open(f"{W}/ct_v0/columns.tsv").read(), "v6 columns.tsv differs from v0"
print("19. eqtl_sparse_v4 --mode ct (block 7, 2-gene chunks) == brute force, columns.tsv == v0  OK")

# eqtl_sparse_v5: v3's kernel, scores.f32 written with pwrite (per-row segments with column chunks,
# whole rows with one chunk); must be byte-identical to v3's memmap output. Also ct mode.
V7 = f"{B}/variants/eqtl_sparse_v5/sparse_score.py"
for tag, extra in (("7-column chunks", ["--chunk-cols", "7"]), ("one chunk", [])):
    print(run(PY + [V7, "--bed", pre, "--weights", f"{W}/w_shuf.tsv", "--out", f"{W}/sum_v7", "--mode", "sum", "--block", "5", "--threads", "1",
                   "--write", "pwrite"] + extra).splitlines()[-1])
    line = run(PY + [f"{B}/compare.py", f"{W}/gold", f"{W}/sum_v7"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
    if extra:
        assert open(f"{W}/sum_v7/scores.f32", "rb").read() == open(f"{W}/sum_v5/scores.f32", "rb").read(), "v7 pwrite bytes differ from v3 memmap"
print("20. eqtl_sparse_v5 --mode sum --write pwrite (7-column chunks: byte-identical to v3; one chunk) == brute force  OK")

# ct_sparse_v4: eqtl_sparse_v3's kernel with genes position-sorted in ct mode too; column order differs from
# v0 (allowed by the contract), values must match the ct golden exactly (compare.py matches columns by name)
V8 = f"{B}/variants/ct_sparse_v4/sparse_score.py"
print(run(PY + [V8, "--bed", pre, "--weights", f"{W}/ctw.tsv", "--out", f"{W}/ct_v8", "--mode", "ct",
               "--thresholds", " ".join(thr), "--all-genes", f"{W}/genes_all.txt", "--block", "7", "--threads", "1",
               "--chunk-cols", str(2 * len(thr))]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold_ct", f"{W}/ct_v8"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
cols8 = {tuple(l.split("	")[1:]) for l in open(f"{W}/ct_v8/columns.tsv").read().splitlines()[1:]}
cols0 = {tuple(l.split("	")[1:]) for l in open(f"{W}/ct_v0/columns.tsv").read().splitlines()[1:]}
assert cols8 == cols0, "ct_sparse_v4 (gene, channel, n_snps) set differs from v0"
print("22. ct_sparse_v4 --mode ct, genes by position (block 7, 2-gene chunks) == brute force; same (gene, channel, n_snps) set as v0  OK")
print(run(PY + [V7, "--bed", pre, "--weights", f"{W}/ctw.tsv", "--out", f"{W}/ct_v7", "--mode", "ct", "--write", "pwrite",
               "--thresholds", " ".join(thr), "--all-genes", f"{W}/genes_all.txt", "--block", "7", "--threads", "1",
               "--chunk-cols", str(2 * len(thr))]).splitlines()[-1])
line = run(PY + [f"{B}/compare.py", f"{W}/gold_ct", f"{W}/ct_v7"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
assert open(f"{W}/ct_v7/columns.tsv").read() == open(f"{W}/ct_v0/columns.tsv").read(), "v7 columns.tsv differs from v0"
print("21. eqtl_sparse_v5 --mode ct --write pwrite (block 7, 2-gene chunks) == brute force, columns.tsv == v0  OK")

# ---- dosage layout (PRSNet-2 features): dosage_v0 vs brute force, the --recode A converter, compare_dosage.py
import pickle
FX = f"{W}/fixdos"; os.makedirs(f"{FX}/genes", exist_ok=True)
with open(f"{FX}/gene_manifest.tsv", "w") as man:
    man.write("gene\tchr\tstart\tend\tn_snps\tbig\n")
    for g, js in gene_snps.items():
        man.write(f"{g}\t19\t0\t0\t{len(js)}\t0\n")
        open(f"{FX}/genes/{g}.snplist", "w").write("".join(sid[j] + "\n" for j in js))
    man.write("CG_EMPTY\t19\t0\t0\t0\t0\n")
pthr = 0.05
print(run(PY + [f"{B}/variants/dosage_v0/dosage.py", "--bed", pre, "--pvalue", f"{W}/SNP.pvalue", "--genes-dir", f"{FX}/genes",
               "--manifest", f"{FX}/gene_manifest.tsv", "--out", f"{W}/dos_v0", "--pthr", str(pthr), "--block", "7"]).splitlines()[-1])
sel = sorted({j for js in gene_snps.values() for j in js if p_all[j] < pthr})
Xref = G[:, sel].copy(); Xref[np.isnan(Xref)] = 3; Xref = Xref.astype(np.int8)          # a1 copies, 3 = missing
X0 = np.load(f"{W}/dos_v0/X.npy"); assert X0.dtype == np.int8 and X0.shape == (ns, len(sel)) and np.array_equal(X0, Xref)
assert [l.split("\t")[1] for l in open(f"{W}/dos_v0/snps.tsv").read().splitlines()[1:]] == [sid[j] for j in sel]
g2 = pickle.load(open(f"{W}/dos_v0/gene2snps.pkl", "rb"))
g2_ref = {g: [sel.index(j) for j in sorted(js) if p_all[j] < pthr] for g, js in gene_snps.items()}
assert g2 == {g: v for g, v in g2_ref.items() if v}, "gene2snps"
assert np.array_equal(np.load(f"{W}/dos_v0/pvalues.npy"), p_all[sel])
print(f"23. dosage_v0 (block 7): X int8 == bed (a1 copies, 3 = missing), {len(sel)} SNPs, gene2snps, pvalues  OK")

# reference path: a plink --recode A file written from the same genotypes, converted by lib/dosage_from_raw.py
with open(f"{W}/fake.raw", "w") as raw, open(f"{W}/pairs.tsv", "w") as pairs:
    raw.write("FID IID PAT MAT SEX PHENOTYPE " + " ".join(f"{sid[j]}_{a1[j]}" for j in sel) + "\n")
    for i in range(ns):
        raw.write(f"F{i} I{i} 0 0 0 -9 " + " ".join("NA" if np.isnan(G[i, j]) else str(int(G[i, j])) for j in sel) + "\n")
    for g, js in gene_snps.items():
        for j in js:
            if p_all[j] < pthr:
                pairs.write(f"{g}\t{sid[j]}\n")
print(run(PY + [f"{B}/lib/dosage_from_raw.py", "--raw", f"{W}/fake.raw", "--bim", pre + ".bim", "--pvalue", f"{W}/SNP.pvalue",
               "--pairs", f"{W}/pairs.tsv", "--out", f"{W}/dos_ref"]).strip())
assert np.array_equal(np.load(f"{W}/dos_ref/X.npy"), Xref)
line = run(PY + [f"{B}/compare_dosage.py", f"{W}/dos_ref", f"{W}/dos_v0"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
# shuffled rows and columns still PASS; one cell, a dropped gene SNP, a flipped allele label FAIL
def write_dos(d, X, rows, cols, a1_override=None, g2tsv=None):
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/X.npy", np.ascontiguousarray(X[rows][:, cols]))
    open(f"{d}/samples.txt", "w").write("".join(f"F{i}\tI{i}\n" for i in rows))
    with open(f"{d}/snps.tsv", "w") as w:
        w.write("col\tsnp\ta1\tbp\tp\n")
        for c, j in enumerate(cols):
            w.write(f"{c}\t{sid[sel[j]]}\t{(a1_override or a1)[sel[j]]}\t0\t{p_all[sel[j]]!r}\n")
    np.save(f"{d}/pvalues.npy", p_all[np.asarray(sel)[cols]])
    open(f"{d}/gene2snps.tsv", "w").write(g2tsv or open(f"{W}/dos_v0/gene2snps.tsv").read())
    open(f"{d}/gene2snps.pkl", "wb").write(b"")
pr, pc = rng.permutation(ns), rng.permutation(len(sel))
write_dos(f"{W}/dos_shuf", Xref, pr, pc)
line = run(PY + [f"{B}/compare_dosage.py", f"{W}/dos_v0", f"{W}/dos_shuf"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line, "(shuffled rows+cols)")
Xbad = Xref.copy(); Xbad[3, 2] = (Xbad[3, 2] + 1) % 3
write_dos(f"{W}/dos_bad", Xbad, np.arange(ns), np.arange(len(sel)))
line = run(PY + [f"{B}/compare_dosage.py", f"{W}/dos_v0", f"{W}/dos_bad"], ok=2).splitlines()[-1]; assert "1 bad snps, 1 bad cells" in line, line; print(line, "(one corrupted cell)")
g2txt = open(f"{W}/dos_v0/gene2snps.tsv").read().splitlines()
gl = g2txt[1].split("\t"); gl[2] = ",".join(gl[2].split(",")[1:] or ["x"]); g2txt[1] = "\t".join(gl)
write_dos(f"{W}/dos_badg", Xref, np.arange(ns), np.arange(len(sel)), g2tsv="\n".join(g2txt) + "\n")
line = run(PY + [f"{B}/compare_dosage.py", f"{W}/dos_v0", f"{W}/dos_badg"], ok=2).splitlines()[-1]; assert "1 bad genes" in line, line; print(line, "(one gene's SNP set changed)")
a1_flip = list(a1); a1_flip[sel[0]] = a2[sel[0]]
write_dos(f"{W}/dos_bada1", Xref, np.arange(ns), np.arange(len(sel)), a1_override=a1_flip)
r = subprocess.run(PY + [f"{B}/compare_dosage.py", f"{W}/dos_v0", f"{W}/dos_bada1"], capture_output=True, text=True)
assert r.returncode == 2 and "ALLELE: 1 SNPs" in r.stdout, r.stdout; print(r.stdout.splitlines()[-1], "(one allele label flipped)")
print("24. dosage_from_raw == dosage_v0; compare_dosage PASS/FAIL behaviour  OK")

# dosage_v1: gap-split spans; must be byte-identical to v0 at every gap setting (gap 1 = one span per run of adjacent rows)
for gap in ("1", "3", "1000"):
    out = run(PY + [f"{B}/variants/dosage_v1/dosage.py", "--bed", pre, "--pvalue", f"{W}/SNP.pvalue", "--genes-dir", f"{FX}/genes",
                    "--manifest", f"{FX}/gene_manifest.tsv", "--out", f"{W}/dos_v1_{gap}", "--pthr", str(pthr), "--block", "7", "--max-gap", gap])
    assert open(f"{W}/dos_v1_{gap}/X.npy", "rb").read() == open(f"{W}/dos_v0/X.npy", "rb").read(), f"v1 gap {gap} differs from v0"
    for f in ("snps.tsv", "gene2snps.tsv", "samples.txt"):
        assert open(f"{W}/dos_v1_{gap}/{f}").read() == open(f"{W}/dos_v0/{f}").read(), f
    print("   gap", gap, "->", [l for l in out.splitlines() if "spans" in l][0].split("span rows")[1].strip())
line = run(PY + [f"{B}/compare_dosage.py", f"{W}/dos_ref", f"{W}/dos_v1_3"]).splitlines()[-1]; assert line.startswith("PASS"), line
print("25. dosage_v1 (max-gap 1, 3, 1000) byte-identical to dosage_v0; PASS vs reference  OK")

# dosage_raw_v0: a synthetic "raw release" with more people (shuffled, fixture people interleaved), extra SNPs
# interleaved, and half the shared SNPs stored on the other allele; the fixture bed `pre` is what 12a would
# have made from it. The seek path must reproduce dosage_v0's output cell for cell.
n_extra_p, n_extra_s = 150, 250
rawp = f"{W}/rawrel"
raw_people = [f"I{i}" for i in range(ns)] + [f"X{i}" for i in range(n_extra_p)]
pp = rng.permutation(len(raw_people)); raw_people = [raw_people[i] for i in pp]
raw_sids = sid + [f"19:{900000 + i}" for i in range(n_extra_s)]
ps = rng.permutation(len(raw_sids)); raw_sids = [raw_sids[i] for i in ps]
raw_flip = {s: bool(rng.random() < 0.5) for s in sid}
Graw = np.full((len(raw_people), len(raw_sids)), np.nan, dtype=np.float32)
Gx = rng.integers(0, 3, size=(len(raw_people), len(raw_sids))).astype(np.float32); Gx[rng.random(Gx.shape) < 0.05] = np.nan
Graw[:] = Gx
rp = {p: i for i, p in enumerate(raw_people)}; rs = {s: j for j, s in enumerate(raw_sids)}
ra1, ra2 = [], []
for s in raw_sids:
    if s in rs and s in raw_flip:
        j = sid.index(s)
        col = G[:, j] if not raw_flip[s] else 2 - G[:, j]                      # stored on the other allele -> 2 - x
        for i in range(ns):
            Graw[rp[f"I{i}"], rs[s]] = col[i]
        ra1.append(a2[j] if raw_flip[s] else a1[j]); ra2.append(a1[j] if raw_flip[s] else a2[j])
    else:
        ra1.append("A"); ra2.append("C")
to_bed(rawp + ".bed", Graw, properties={"fid": [f"F{p[1:]}" if p[0] == "I" else "FX" for p in raw_people], "iid": raw_people,
                                        "sid": raw_sids, "chromosome": ["19"] * len(raw_sids),
                                        "bp_position": [int(s.split(":")[1]) for s in raw_sids], "allele_1": ra1, "allele_2": ra2})
for bm in ("16", "1"):
    out = run(PY + [f"{B}/variants/dosage_raw_v0/dosage_raw.py", "--raw", rawp, "--bed", pre, "--pvalue", f"{W}/SNP.pvalue",
                    "--genes-dir", f"{FX}/genes", "--manifest", f"{FX}/gene_manifest.tsv", "--out", f"{W}/dos_raw_{bm}",
                    "--pthr", str(pthr), "--block-mb", bm, "--sub", "5", "--max-gap", "2"])
    Xr = np.load(f"{W}/dos_raw_{bm}/X.npy")
    assert Xr.shape == Xref.shape and np.array_equal(Xr, Xref), f"raw path differs from the fixture path (block-mb {bm})"
    for f in ("snps.tsv", "gene2snps.tsv", "samples.txt"):
        assert open(f"{W}/dos_raw_{bm}/{f}").read() == open(f"{W}/dos_v0/{f}").read(), f
    print("   " + [l for l in out.splitlines() if l.startswith("raw:")][0])
line = run(PY + [f"{B}/compare_dosage.py", f"{W}/dos_ref", f"{W}/dos_raw_1"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line)
print("26. dosage_raw_v0 from a shuffled raw release with flipped alleles == dosage_v0 (block-mb 16 and 1); PASS vs reference  OK")

# lib/x2bit: pack/unpack round trip (odd widths, batch gather), then dosage_pack_v0 on an in-order raw release
sys.path.insert(0, f"{B}/lib"); import x2bit
for k in (1, 3, 4, 5, 89, 400):
    A = rng.integers(0, 4, size=(37, k)).astype(np.int8)
    P = x2bit.pack(A); assert P.shape == (37, (k + 3) // 4) and P.dtype == np.uint8
    assert np.array_equal(x2bit.unpack(P, k), A), k
    assert np.array_equal(x2bit.batch(P, [5, 0, 5, 36], k), A[[5, 0, 5, 36]])
# an in-order raw release: the fixture's SNPs in fixture order, extra SNPs appended, same people shuffle and flips as above
raw2 = f"{W}/rawrel2"
ord_sids = sid + [f"19:{900000 + i}" for i in range(n_extra_s)]
G2 = np.stack([Graw[:, rs[s]] for s in ord_sids], axis=1)
ra1b = [ra1[rs[s]] for s in ord_sids]; ra2b = [ra2[rs[s]] for s in ord_sids]
# every third shared SNP is stored on the other strand in this release (both alleles complemented, same codes):
# the counted allele must then be matched through its complement, as step 03's mismatch.R convention implies
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
n_strand = 0
for j, s in enumerate(ord_sids):
    if s in raw_flip and sid.index(s) % 3 == 0 and COMP[ra1b[j]] != ra2b[j]:   # skip palindromic pairs (A/T, C/G)
        ra1b[j], ra2b[j] = COMP[ra1b[j]], COMP[ra2b[j]]; n_strand += 1
assert n_strand > 10
to_bed(raw2 + ".bed", G2, properties={"fid": [f"F{p[1:]}" if p[0] == "I" else "FX" for p in raw_people], "iid": raw_people,
                                      "sid": ord_sids, "chromosome": ["19"] * len(ord_sids),
                                      "bp_position": [int(s.split(":")[1]) for s in ord_sids], "allele_1": ra1b, "allele_2": ra2b})
os.environ["BENCH_LIB"] = f"{B}/lib"
for bm in ("16", "1"):
    out = run(PY + [f"{B}/variants/dosage_pack_v0/dosage_pack.py", "--raw", raw2, "--bed", pre, "--pvalue", f"{W}/SNP.pvalue",
                    "--genes-dir", f"{FX}/genes", "--manifest", f"{FX}/gene_manifest.tsv", "--out", f"{W}/dos_pack_{bm}",
                    "--pthr", str(pthr), "--block-mb", bm, "--sub", "5", "--max-gap", "2"])
    X2 = np.load(f"{W}/dos_pack_{bm}/X2.npy")
    assert X2.dtype == np.uint8 and X2.shape == (ns, (len(sel) + 3) // 4), X2.shape
    assert np.array_equal(x2bit.unpack(X2, len(sel)), Xref), f"packed store differs (block-mb {bm})"
    assert not os.path.exists(f"{W}/dos_pack_{bm}/X.npy")
    for f in ("snps.tsv", "gene2snps.tsv", "samples.txt"):
        assert open(f"{W}/dos_pack_{bm}/{f}").read() == open(f"{W}/dos_v0/{f}").read(), f
    print("   " + [l for l in out.splitlines() if l.startswith("dosage[pack_v0]")][0])
line = run(PY + [f"{B}/compare_dosage.py", f"{W}/dos_ref", f"{W}/dos_pack_1"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line, "(X2 candidate vs int8 golden)")
line = run(PY + [f"{B}/compare_dosage.py", f"{W}/dos_pack_1", f"{W}/dos_shuf"]).splitlines()[-1]; assert line.startswith("PASS"), line; print(line, "(X2 golden vs shuffled int8 candidate)")
X2bad = np.load(f"{W}/dos_pack_1/X2.npy").copy(); X2bad[3, 0] ^= 0b11
os.makedirs(f"{W}/dos_pack_bad", exist_ok=True); np.save(f"{W}/dos_pack_bad/X2.npy", X2bad)
for f in ("snps.tsv", "gene2snps.tsv", "samples.txt", "pvalues.npy", "gene2snps.pkl"):
    open(f"{W}/dos_pack_bad/{f}", "wb").write(open(f"{W}/dos_pack_1/{f}", "rb").read())
line = run(PY + [f"{B}/compare_dosage.py", f"{W}/dos_ref", f"{W}/dos_pack_bad"], ok=2).splitlines()[-1]; assert "1 bad snps, 1 bad cells" in line, line; print(line, "(one packed cell corrupted)")
print("27. x2bit pack/unpack/batch round trip; dosage_pack_v0 X2.npy unpacks to dosage_v0's X; compare_dosage handles X2 on either side  OK")

# production options: --keep (FID IID list, here the fixture people in a different order) and --a1 (SNP allele
# overrides, here the fixture A1 spelled out for every SNP plus a flip for one SNP) ; and x2bit.Sharded
keep_order = rng.permutation(ns)
open(f"{W}/keep.txt", "w").write("".join(f"F{i} I{i}\n" for i in keep_order))
jflip = sel[7]
with open(f"{W}/a1_over.txt", "w") as w:
    for j in range(m):
        w.write(f"{sid[j]} {a2[j] if j == jflip else a1[j]}\n")
run(PY + [f"{B}/variants/dosage_pack_v0/dosage_pack.py", "--raw", raw2, "--bed", pre, "--pvalue", f"{W}/SNP.pvalue",
          "--genes-dir", f"{FX}/genes", "--manifest", f"{FX}/gene_manifest.tsv", "--out", f"{W}/dos_pack_prod",
          "--pthr", str(pthr), "--block-mb", "1", "--sub", "5", "--keep", f"{W}/keep.txt", "--a1", f"{W}/a1_over.txt"])
Xp = x2bit.unpack(np.load(f"{W}/dos_pack_prod/X2.npy"), len(sel))
Xexp = Xref[keep_order].copy(); col7 = Xexp[:, 7]; Xexp[:, 7] = np.where(col7 == 3, 3, 2 - col7)
assert np.array_equal(Xp, Xexp), "--keep order / --a1 flip not honoured"
assert open(f"{W}/dos_pack_prod/samples.txt").read() == "".join(f"F{i}\tI{i}\n" for i in keep_order)
assert [l.split("\t")[2] for l in open(f"{W}/dos_pack_prod/snps.tsv").read().splitlines()[1:]][7] == a2[jflip]
line = run(PY + [f"{B}/compare_dosage.py", f"{W}/dos_ref", f"{W}/dos_pack_prod"], ok=2).splitlines()
assert any(l.startswith("ALLELE: 1 SNPs") for l in line), line[-1]        # the deliberate flip is caught against the golden
sh = x2bit.Sharded([(f"{W}/dos_pack_1/X2.npy", len(sel)), (f"{W}/dos_pack_16/X2.npy", len(sel))])
assert sh.n == ns and sh.m == 2 * len(sel)
bb = sh.batch([3, 1, 3]); assert np.array_equal(bb[:, :len(sel)], Xref[[3, 1, 3]]) and np.array_equal(bb[:, len(sel):], Xref[[3, 1, 3]])
print("28. dosage_pack_v0 --keep (reordered people) and --a1 (one override) honoured and caught by compare; x2bit.Sharded concatenates shards  OK")

# port/merge_shards.py over two copies of the packed shard posing as chr1 and chr2
import shutil
MD = f"{W}/merged"; os.makedirs(MD, exist_ok=True)
for c in (1, 2):
    shutil.copytree(f"{W}/dos_pack_1", f"{MD}/chr{c}")
    g2t = open(f"{MD}/chr{c}/gene2snps.tsv").read().replace("CG", f"C{c}G")        # distinct gene names per shard
    open(f"{MD}/chr{c}/gene2snps.tsv", "w").write(g2t)
open(f"{W}/pheno.tab", "w").write("".join(f"F{i} I{i} {1 + (i % 3 == 0)}\n" for i in range(ns)))
print(run(PY + [f"{B}/port/merge_shards.py", MD, "--chrs", "1-2", "--pheno", f"{W}/pheno.tab"]).strip())
info = open(f"{MD}/info.csv").read().splitlines(); assert info[0] == "sample_id,label,ancestry" and len(info) == 1 + ns
assert info[1:4] == ["I0,1,EUR", "I1,0,EUR", "I2,0,EUR"], info[1:4]
sh_rows = [l.split("\t") for l in open(f"{MD}/shards.tsv").read().splitlines()[1:]]
assert [(r[0], int(r[2]), int(r[3])) for r in sh_rows] == [("1", len(sel), 0), ("2", len(sel), len(sel))]
mg = pickle.load(open(f"{MD}/gene2snps.pkl", "rb"))
for g, cols in g2.items():
    assert mg[f"C1{g[1:]}"] == cols and mg[f"C2{g[1:]}"] == [c + len(sel) for c in cols], g
assert np.array_equal(np.load(f"{MD}/pvalues.npy"), np.concatenate([p_all[sel], p_all[sel]]))
msn = open(f"{MD}/snps.tsv").read().splitlines(); assert len(msn) == 1 + 2 * len(sel) and msn[1 + len(sel)].split("\t")[5] == "2"
shx = x2bit.Sharded([(r[1], int(r[2])) for r in sh_rows]); assert np.array_equal(shx.batch([0, 9])[:, len(sel):], Xref[[0, 9]])
print("29. merge_shards: genome-wide offsets, gene2snps, pvalues, snps.tsv, shards.tsv feed x2bit.Sharded  OK")

# port/prsnet2_store.py: the trainer's view of the merged directory (ram and memmap), model inputs, export
sys.path.insert(0, f"{B}/port")
from prsnet2_store import DosageStore
Xfull = np.concatenate([Xref, Xref], axis=1)                      # chr1 and chr2 are copies of the same shard
Xzero = np.where(Xfull == 3, 0, Xfull)
genes_all = sorted(mg) ; extra = ["ZZZ_NOSNP", "AAA_NOSNP"]
order = extra[:1] + genes_all[::-1] + extra[1:]                    # node ids in a deliberately non-sorted order
open(f"{W}/gene_order.txt", "w").write("".join(g + "\n" for g in order))
cwd0 = os.getcwd(); os.chdir(B)                                   # shards.tsv paths are as merge_shards saw them
try:
    st_ram = DosageStore(MD, load="ram", gene_order=f"{W}/gene_order.txt", log=lambda *_: None)
    st_mm = DosageStore(MD, load="memmap", gene_order=f"{W}/gene_order.txt", missing_to_zero=False)
finally:
    os.chdir(cwd0)
assert st_ram.n == ns and st_ram.m == 2 * len(sel) and st_ram.labels.tolist() == [int(i % 3 == 0) for i in range(ns)]
pick = [5, 0, 5, ns - 1, 2]
assert np.array_equal(st_ram.rows_int8(pick), Xzero[pick]), "ram rows (3 -> 0)"
assert np.array_equal(st_mm.rows_int8(pick), Xfull[pick]), "memmap rows (3 kept), unsorted index order"
assert (Xfull[pick] == 3).any(), "test data should contain missing calls"
try:
    import torch
    t = st_ram.rows(pick); assert t.dtype == torch.float32 and np.array_equal(t.numpy(), Xzero[pick].astype(np.float32))
except ImportError:
    pass
mi = st_ram.model_inputs()
node = {g: i for i, g in enumerate(order)}
assert mi["gene_ids"] == sorted(node[g] for g in mg) and mi["n_genes"] == len(order) and mi["n_snps"] == 2 * len(sel)
exp_snp = [c for gid in mi["gene_ids"] for c in mg[order[gid]]]
assert mi["snp_ids"] == exp_snp and mi["gene2snp_len"] == [len(mg[order[gid]]) for gid in mi["gene_ids"]]
assert mi["gene_names"] == [order[i] for i in mi["gene_ids"]] and not any(g in mi["gene_names"] for g in extra)
EX = f"{W}/export"; st_ram.export(EX, chunk=3, log=lambda *_: None)
assert np.array_equal(np.load(f"{EX}/X.npy"), Xfull) and np.load(f"{EX}/X.npy").dtype == np.int8
assert np.array_equal(np.load(f"{EX}/Y.npy"), st_ram.labels) and np.array_equal(np.load(f"{EX}/pvalues.npy"), np.load(f"{MD}/pvalues.npy"))
eg = pickle.load(open(f"{EX}/gene2snps.pkl", "rb")); assert eg == {node[g]: cols for g, cols in mg.items()}
print("30. prsnet2_store: ram/memmap rows match the shards (3 -> 0 as src/dataset.py), model_inputs sorted by node id, export = authors' layout  OK")

# variants/bed_seek_v0: plink's --keep --extract --a1-allele --make-bed rules, byte for byte, against a brute force
BS = f"{W}/bedseek"; os.makedirs(f"{BS}/raw", exist_ok=True)
nr, mr = 37, 25                                                   # raw people (not a multiple of 4), raw variants
codes = rng.integers(0, 4, size=(mr, nr)).astype(np.uint8)        # raw bed codes 0 homA1 1 missing 2 het 3 homA2
alle = [("A", "G"), ("C", "T"), ("G", "A"), ("T", "C")]
ids = [f"rs{j}" for j in range(mr)]
with open(f"{BS}/raw/r.bim", "w") as w:
    for j in range(mr):
        w.write(f"5\t{ids[j]}\t0\t{1000 + 10 * j}\t{alle[j % 4][0]}\t{alle[j % 4][1]}\n")
open(f"{BS}/raw/r.fam", "w").write("".join(f"F{i} I{i} 0 0 {1 + i % 2} -9\n" for i in range(nr)))
with open(f"{BS}/raw/r.bed", "wb") as w:
    w.write(b"\x6c\x1b\x01"); w.write(x2bit.pack(codes).tobytes())
want = sorted(rng.choice(mr, size=15, replace=False)); keep_p = sorted(rng.choice(nr, size=22, replace=False))
open(f"{BS}/extract.txt", "w").write("".join(ids[j] + "\n" for j in rng.permutation(want)))          # shuffled: order must not matter
open(f"{BS}/keep.txt", "w").write("".join(f"F{i} I{i}\n" for i in rng.permutation(keep_p)))
# counted allele: raw A1 for some, raw A2 (swap) for some, impossible ('N') for two, absent for the rest (minor rule)
kind = {j: ["a1", "a2", "n", "none"][k % 4] for k, j in enumerate(want)}
with open(f"{BS}/a1.txt", "w") as w:
    for j in want:
        if kind[j] == "a1": w.write(f"{ids[j]} {alle[j % 4][0]}\n")
        elif kind[j] == "a2": w.write(f"{ids[j]} {alle[j % 4][1]}\n")
        elif kind[j] == "n": w.write(f"{ids[j]} N\n")
sub = codes[np.ix_(want, keep_p)].copy()                          # brute force expectation
exp_swap = []
for r, j in enumerate(want):
    if kind[j] == "a2": sw = True
    elif kind[j] == "a1": sw = False
    else:                                                          # plink default: A1 = minor allele among kept people
        c = sub[r]; n1 = 2 * (c == 0).sum() + (c == 2).sum(); tot = 2 * (c != 1).sum()
        sw = bool(2 * n1 > tot)
    if sw:
        sub[r] = np.array([3, 1, 2, 0], dtype=np.uint8)[sub[r]]; exp_swap.append(j)
assert exp_swap and len(exp_swap) < len(want)
out = run(PY + [f"{B}/variants/bed_seek_v0/bed_seek.py", "--raw", f"{BS}/raw/r", "--extract", f"{BS}/extract.txt", "--keep", f"{BS}/keep.txt",
                "--a1", f"{BS}/a1.txt", "--out", f"{BS}/out/geno", "--block", "4", "--max-gap", "1", "--threads", "3"])
got = np.fromfile(f"{BS}/out/geno.bed", dtype=np.uint8, offset=3).reshape(len(want), (len(keep_p) + 3) // 4)
assert np.array_equal(x2bit.unpack(got, len(keep_p)), sub), "bed codes differ from the brute force"
bim_o = [l.split() for l in open(f"{BS}/out/geno.bim")]
assert [b[1] for b in bim_o] == [ids[j] for j in want], "variants not in raw .bim order"
for b, j in zip(bim_o, want):
    a1e, a2e = alle[j % 4][::-1] if j in exp_swap else alle[j % 4]
    assert b[4] == a1e and b[5] == a2e, (b, j in exp_swap)
assert [l.split()[:2] for l in open(f"{BS}/out/geno.fam")] == [[f"F{i}", f"I{i}"] for i in keep_p], "people not in raw .fam order"
assert "minor-allele rule swapped" in out
# compare_bed: PASS against itself, FAIL naming swapped alleles when the a1 list flips one SNP, FAIL on a corrupted byte
os.makedirs(f"{BS}/gold", exist_ok=True)
for e in ("bed", "bim", "fam"): open(f"{BS}/gold/geno.{e}", "wb").write(open(f"{BS}/out/geno.{e}", "rb").read())
line = run(PY + [f"{B}/compare_bed.py", f"{BS}/gold", f"{BS}/out"]).splitlines()[-1]; assert line.startswith("PASS"), line
# bed_seek_v1 (fused gather+pack, packed-byte swaps, minor rule) must give the same bytes; n = 22 is not a multiple of 4
out1 = run(PY + [f"{B}/variants/bed_seek_v1/bed_seek.py", "--raw", f"{BS}/raw/r", "--extract", f"{BS}/extract.txt", "--keep", f"{BS}/keep.txt",
                 "--a1", f"{BS}/a1.txt", "--out", f"{BS}/v1/geno", "--block", "4", "--max-gap", "1", "--threads", "3"])
assert out1.split("minor-allele rule swapped")[1].split(";")[0] == out.split("minor-allele rule swapped")[1].split(";")[0]
line = run(PY + [f"{B}/compare_bed.py", f"{BS}/gold", f"{BS}/v1"]).splitlines()[-1]; assert line.startswith("PASS"), line
j1 = [j for j in want if kind[j] == "a1"][0]
open(f"{BS}/a1_flip.txt", "w").write(open(f"{BS}/a1.txt").read().replace(f"{ids[j1]} {alle[j1 % 4][0]}", f"{ids[j1]} {alle[j1 % 4][1]}"))
run(PY + [f"{B}/variants/bed_seek_v0/bed_seek.py", "--raw", f"{BS}/raw/r", "--extract", f"{BS}/extract.txt", "--keep", f"{BS}/keep.txt",
          "--a1", f"{BS}/a1_flip.txt", "--out", f"{BS}/flip/geno", "--threads", "1"])
txt = run(PY + [f"{B}/compare_bed.py", f"{BS}/gold", f"{BS}/flip"], ok=2)
assert "1 of 15 variants differ, 1 of them with swapped .bim alleles" in txt and ids[j1] in txt, txt
bad = bytearray(open(f"{BS}/out/geno.bed", "rb").read()); bad[3 + 2 * got.shape[1]] ^= 0b11
os.makedirs(f"{BS}/bad", exist_ok=True); open(f"{BS}/bad/geno.bed", "wb").write(bad)
for e in ("bim", "fam"): open(f"{BS}/bad/geno.{e}", "w").write(open(f"{BS}/out/geno.{e}").read())
txt = run(PY + [f"{B}/compare_bed.py", f"{BS}/gold", f"{BS}/bad"], ok=2); assert "1 of 15 variants differ, 0 of them" in txt and "(1 people)" in txt, txt
# --unassignable complement: 'N' is still an error there, so a strand-flipped allele instead (complement of raw A2 -> swap)
jn = [j for j in want if kind[j] == "n"][0]; comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
open(f"{BS}/a1_comp.txt", "w").write(open(f"{BS}/a1.txt").read().replace(f"{ids[jn]} N", f"{ids[jn]} {comp[alle[jn % 4][1]]}"))
out = run(PY + [f"{B}/variants/bed_seek_v0/bed_seek.py", "--raw", f"{BS}/raw/r", "--extract", f"{BS}/extract.txt", "--keep", f"{BS}/keep.txt",
                "--a1", f"{BS}/a1_comp.txt", "--out", f"{BS}/comp/geno", "--threads", "2", "--unassignable", "complement"])
assert "1 matched on the other strand" in out, out
bim_c = {l.split()[1]: l.split()[4:6] for l in open(f"{BS}/comp/geno.bim")}
assert bim_c[ids[jn]] == list(alle[jn % 4][::-1]), bim_c[ids[jn]]
run(PY + [f"{B}/variants/bed_seek_v0/bed_seek.py", "--raw", f"{BS}/raw/r", "--extract", f"{BS}/extract.txt", "--keep", f"{BS}/keep.txt",
          "--a1", f"{BS}/a1.txt", "--out", f"{BS}/err/geno", "--unassignable", "error"], ok=1)
print("31. bed_seek_v0 + v1 (same bytes): plink --keep/--extract/--a1-allele/--make-bed rules reproduced (raw order, swaps, minor-allele default, complement mode); compare_bed PASS/FAIL  OK")

# lib/feats_extract.py: a synthetic 12d feats.f32 (gene_order of 7 genes, 3 thresholds) -> scores layout for 3 of them
import json
FE = f"{W}/feats"; os.makedirs(FE, exist_ok=True)
n_fe, G_fe, T_fe = 50, 7, 3; thr_fe = ["1e-3", "0.05", "1"]
order_fe = [f"G{i}" for i in range(G_fe)]; open(f"{FE}/gene_order.txt", "w").write("".join(g + "\n" for g in order_fe))
feats = rng.random((n_fe, G_fe * T_fe), dtype=np.float32); feats.tofile(f"{FE}/feats.f32")
open(f"{FE}/info.csv", "w").write("sample_id,label,ancestry\n" + "".join(f"S{i},{i % 2},EUR\n" for i in range(n_fe)))
json.dump({"n_samples": n_fe, "n_genes": G_fe, "n_thresholds": T_fe, "thresholds": thr_fe, "gene_order": f"{FE}/gene_order.txt",
           "feats": f"{FE}/feats.f32", "info": f"{FE}/info.csv"}, open(f"{FE}/feats.meta.json", "w"))
open(f"{FE}/genes.txt", "w").write("G5\nG0\nG3\n")
open(f"{FE}/gene_counts.tsv", "w").write("gene\tn_index\tn_P<=1e-3\tn_P<=0.05\tn_P<=1\nG5\t9\t1\t4\t9\nG0\t0\t0\t0\t0\n")
run(PY + [f"{B}/lib/feats_extract.py", "--meta", f"{FE}/feats.meta.json", "--genes", f"{FE}/genes.txt", "--out", f"{FE}/out",
          "--gene-counts", f"{FE}/gene_counts.tsv", "--rows", "7"])
got = np.fromfile(f"{FE}/out/scores.f32", dtype=np.float32).reshape(n_fe, 9)
exp = np.concatenate([feats[:, g * T_fe:(g + 1) * T_fe] for g in (5, 0, 3)], axis=1)
assert np.array_equal(got, exp), "extracted columns differ"
cols = [l.split("\t") for l in open(f"{FE}/out/columns.tsv").read().splitlines()[1:]]
assert [c[1] for c in cols] == ["G5"] * 3 + ["G0"] * 3 + ["G3"] * 3 and [c[2] for c in cols[:3]] == ["P<=1e-3", "P<=0.05", "P<=1"]
assert [c[3] for c in cols] == ["1", "4", "9", "0", "0", "0", "0", "0", "0"]
assert open(f"{FE}/out/samples.txt").read().splitlines()[:2] == ["S0\tS0", "S1\tS1"]
line = run(PY + [f"{B}/compare.py", f"{FE}/out", f"{FE}/out", "--rtol", "1e-5", "--atol", "1e-7"]).splitlines()[-1]; assert line.startswith("PASS"), line
print("32. feats_extract: gene columns cut out of a 12d feats.f32 in chunks, labels/counts/samples as the scores layout  OK")

print("\nALL SMOKE TESTS PASSED")
