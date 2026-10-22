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

print("\nALL SMOKE TESTS PASSED")
