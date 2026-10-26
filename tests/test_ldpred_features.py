#!/usr/bin/env python3
"""test_ldpred_features.py -- synthetic end-to-end test of the LDpred2 feature path
(README s14), no PLINK needed:

    python tests/test_ldpred_features.py            (from the bench root)

1. lib/make_ldpred_weights.py: rsID -> target ID through --pos-map, A1 kept as
   given, diverged grid channel refused, zero betas write no row, shared vs
   unique window assignment (nearest window centre), intergenic-mass report
2. variants/ldp_sum_v0/sparse_score.py --mode sum --all-genes --channels
   --out-layout chunks == brute-force plink2 scoresum rule (missing calls
   mean-imputed), zero columns for genes without weight, compare.py PASS against
   a flat golden, gene_counts.tsv header carries the channel names
3. production's 12d (prod_snapshot/12_prsnet.py assemble) takes the chunk layout
   with T = 2 pseudo-thresholds and lays genes out in gene-order with all-zero
   columns for genes without features -- what train_prsnet.py then reads
"""
import importlib.util, os, subprocess, sys, tempfile, types
import numpy as np
from bed_reader import to_bed

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = [sys.executable]
rng = np.random.default_rng(7)


def run(cmd, ok=0):
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == ok, f"rc {r.returncode} != {ok}\n{' '.join(cmd)}\n{r.stdout}\n{r.stderr}"
    return r.stdout + r.stderr


W = tempfile.mkdtemp()
ns, m = 120, 60
pos = np.sort(rng.choice(np.arange(10_000, 90_000), m, replace=False))
tid = [f"19:{p}" for p in pos]                               # the target's IDs (chr:pos)
rsid = [f"rs{1000 + i}" for i in range(m)]                   # LDpred2's IDs
a1 = rng.choice(list("ACGT"), m); a2 = np.array([rng.choice([x for x in "ACGT" if x != y]) for y in a1])
G = rng.integers(0, 3, size=(ns, m)).astype(np.float32)
G[rng.random(G.shape) < 0.04] = np.nan
G[:, 3] = np.nan                                             # an all-missing SNP
pre = f"{W}/syn"
to_bed(pre + ".bed", G, properties={"fid": [f"F{i}" for i in range(ns)], "iid": [str(1000 + i) for i in range(ns)],
                                    "sid": tid, "chromosome": ["19"] * m, "bp_position": [int(p) for p in pos],
                                    "allele_1": list(a1), "allele_2": list(a2)})
# the step-04 .bim: the same SNPs plus one on another chromosome (must be ignored)
with open(f"{W}/target.bim", "w") as w:
    for i in range(m):
        w.write(f"19\t{tid[i]}\t0\t{pos[i]}\t{a1[i]}\t{a2[i]}\n")
    w.write("20\t20:5000\t0\t5000\tA\tC\n")

# windows: G1 and G2 overlap (G2 narrower), G3 separate, G4 empty, G5 on chr20
windows = [(19, 9_000, 40_000, "G1"), (19, 30_000, 45_000, "G2"), (19, 60_000, 80_000, "G3"),
           (19, 95_000, 99_000, "G4"), (20, 1, 9_000, "G5")]
with open(f"{W}/gene_windows.tsv", "w") as w:
    w.write("chr\tstart\tend\tgene\n")
    for c, s, e, g in windows:
        w.write(f"{c}\t{s}\t{e}\t{g}\n")
with open(f"{W}/gene_manifest.tsv", "w") as w:
    w.write("gene\tstart\tend\tn_snps\tbig\n")
    for c, s, e, g in windows:
        if c == 19:
            w.write(f"{g}\t{s}\t{e}\t0\t0\n")

# betas.txt: rsIDs, A1 = the target's A1 or A2 (sign reconciled upstream), inf/auto/g001 (diverged, zeros)/g011
chan_true = {"auto": rng.normal(size=m) * 1e-3, "inf": rng.normal(size=m) * 1e-3, "g011": rng.normal(size=m) * 1e-3}
chan_true["g011"][rng.random(m) < 0.3] = 0.0                # a sparse-ish model: zeros write no row
on_a2 = rng.random(m) < 0.5
beta_a1 = {c: np.where(on_a2, -v, v) for c, v in chan_true.items()}   # effect per copy of the target's A1
with open(f"{W}/betas.txt", "w") as w:
    w.write("SNP\tCHR\tA1\tinf\tauto\tg001\tg011\n")
    for i in range(m):
        if i == 7:
            continue                                         # an HM3+ SNP the target lacks is simply absent
        al = a2[i] if on_a2[i] else a1[i]
        w.write(f"{rsid[i]}\t19\t{al}\t{chan_true['inf'][i]:.6e}\t{chan_true['auto'][i]:.6e}\t0.000000e+00\t{chan_true['g011'][i]:.6e}\n")
    w.write("rs999999\t19\tA\t1e-3\t1e-3\t0\t1e-3\n")         # not in the target: unmatched
    w.write("rs888888\t20\tA\t1e-3\t1e-3\t0\t1e-3\n")         # other chromosome: skipped
with open(f"{W}/grid_params.tsv", "w") as w:
    w.write("p\th2\tsparse\tlabel\tdiverged\tn_nonzero\n")
    w.write("1e-05\t0.0308\tFALSE\tg001\tTRUE\t0\n")
    w.write("0.0032\t0.0308\tFALSE\tg011\tFALSE\t1104605\n")
with open(f"{W}/pos_map.tsv", "w") as w:
    w.write("rsid\tchr\tpos\n")
    for i in range(m):
        w.write(f"{rsid[i]}\t19\t{pos[i]}\n")
    w.write("rs999999\t19\t1\n")

MK = f"{B}/lib/make_ldpred_weights.py"
common = ["--betas", f"{W}/betas.txt", "--grid-params", f"{W}/grid_params.tsv", "--bim", f"{W}/target.bim",
          "--windows", f"{W}/gene_windows.tsv", "--chr", "19", "--manifest", f"{W}/gene_manifest.tsv",
          "--pos-map", f"{W}/pos_map.tsv"]
# diverged channel refused; a channel absent from betas.txt refused
out = run(PY + [MK] + common + ["--channels", "g001", "--out", f"{W}/x.tsv", "--genes-out", f"{W}/x.genes", "--channels-out", f"{W}/x.ch"], ok=1)
assert "DIVERGED" in out, out
out = run(PY + [MK] + common + ["--channels", "g012", "--out", f"{W}/x.tsv", "--genes-out", f"{W}/x.genes", "--channels-out", f"{W}/x.ch"], ok=1)
assert "is not in" in out, out


def build(assign, chans, tag):
    out = run(PY + [MK] + common + ["--channels", chans, "--assign", assign, "--out", f"{W}/{tag}.tsv",
                                    "--genes-out", f"{W}/{tag}.genes", "--channels-out", f"{W}/{tag}.ch", "--report", f"{W}/{tag}.rep"])
    rows = [l.rstrip("\n").split("\t") for l in open(f"{W}/{tag}.tsv")][1:]
    return out, rows


def in_win(p, s, e):
    return s < p <= e


out, rows = build("shared", "auto,g011,inf", "shared")
chans = open(f"{W}/shared.ch").read().split()
assert chans == ["auto", "g011", "inf"], chans
assert open(f"{W}/shared.genes").read().split() == ["G1", "G2", "G3", "G4"]
# expected rows: every (gene, matched SNP in window, channel) with a nonzero beta; SNP 7 absent
exp = set()
for i in range(m):
    if i == 7:
        continue
    for c, s, e, g in windows:
        if c == 19 and in_win(pos[i], s, e):
            for ch in chans:
                if chan_true[ch][i] != 0.0:
                    exp.add((g, tid[i], ch))
got = {(r[0], r[1], r[3]) for r in rows}
assert got == exp, (len(got), len(exp), sorted(got ^ exp)[:5])
for r in rows:
    i = tid.index(r[1])
    assert r[2] == (a2[i] if on_a2[i] else a1[i]), "A1 must be passed through as given"
    assert abs(float(r[4]) - chan_true[r[3]][i]) <= 1e-9 * max(1.0, abs(chan_true[r[3]][i]))
shared_snps = [i for i in range(m) if i != 7 and in_win(pos[i], 30_000, 40_000)]
assert shared_snps, "the synthetic windows must overlap over some SNP"
assert "1 unmatched" in out.replace(",", "") or "1 unmatched" in out, out
# intergenic mass: SNPs between 45,000 and 60,000 and above 80,000 are outside every window
mass_all = sum(abs(chan_true["auto"][i]) for i in range(m) if i != 7)
mass_in = sum(abs(chan_true["auto"][i]) for i in range(m) if i != 7 and any(in_win(pos[i], s, e) for c, s, e, g in windows if c == 19))
rep = [l.split("\t") for l in open(f"{W}/shared.rep")][1:]
r_auto = [r for r in rep if r[1] == "auto"][0]
assert abs(float(r_auto[-1]) - (1 - mass_in / mass_all)) < 1e-3, (r_auto, 1 - mass_in / mass_all)
print(f"1a. make_ldpred_weights shared: {len(rows)} rows, {len(shared_snps)} SNPs in two windows, "
      f"intergenic {100 * float(r_auto[-1]):.1f}%  OK")

out, urows = build("unique", "auto", "unique")
ug = {}
for r in urows:
    assert r[1] not in ug, "unique assignment gives one gene per SNP"
    ug[r[1]] = r[0]
for i in shared_snps:
    # nearest window centre: G1 centre 24,500, G2 centre 37,500
    want = "G1" if abs(pos[i] - 24_500) < abs(pos[i] - 37_500) else "G2"
    if chan_true["auto"][i] != 0.0:
        assert ug[tid[i]] == want, (tid[i], ug[tid[i]], want)
assert len(urows) == len({r[1] for r in rows if r[3] == "auto"}), "unique keeps every SNP that shared had"
print(f"1b. make_ldpred_weights unique: {len(urows)} rows, one gene per SNP, nearest centre  OK")

# ---- 2. the scorer in sum mode, fixed layout, chunk layout, vs brute force
SC = f"{B}/variants/ldp_sum_v0/sparse_score.py"
sd = f"{W}/chr19/scores"
run(PY + [SC, "--bed", pre, "--weights", f"{W}/shared.tsv", "--out", sd, "--mode", "sum", "--all-genes", f"{W}/shared.genes",
          "--channels", "auto g011 inf", "--block", "16", "--threads", "1", "--chunk-cols", "6", "--out-layout", "chunks"])
D = G.astype(np.float64)
mean = np.nanmean(D, axis=0); mean = np.where(np.isnan(mean), 0.0, mean)
ii, jj = np.where(np.isnan(D)); D[ii, jj] = mean[jj]
truth = {}
for g in ["G1", "G2", "G3", "G4"]:
    for ch in chans:
        s = np.zeros(ns)
        for r in rows:
            if r[0] == g and r[3] == ch:
                i = tid.index(r[1]); b = float(r[4])
                s += b * ((2 - D[:, i]) if r[2] == a2[i] else D[:, i])
        truth[(g, ch)] = s
hdr = open(f"{sd}/gene_counts.tsv").readline().split()
assert hdr == ["gene", "n_index", "n_auto", "n_g011", "n_inf"], hdr
man = [l.split() for l in open(f"{sd}/scores_manifest.tsv")][1:]
assert len(man) == 2, man                                     # 4 genes x 3 channels = 12 cols in chunks of 6
got = {}
for k, fn, Gk, _ in man:
    A = np.load(f"{sd}/{fn}"); genes = open(f"{sd}/genes_chunk{int(k):03d}.txt").read().split()
    assert A.shape == (ns, len(genes), 3), A.shape
    for gi, g in enumerate(genes):
        for t, ch in enumerate(chans):
            got[(g, ch)] = A[:, gi, t].astype(np.float64)
assert set(got) == set(truth)
worst = max(np.max(np.abs(got[k] - truth[k]) / np.maximum(1e-7, np.abs(truth[k]))) for k in truth)
assert worst < 1e-5, worst
assert not np.any(got[("G4", "auto")]), "a gene without weight gets zero columns"
# compare.py against a flat golden of the truth (G4 absent there -> extra columns, not compared)
gd = f"{W}/gold"; os.makedirs(gd)
open(f"{gd}/samples.txt", "w").write("".join(f"F{i}\t{1000 + i}\n" for i in range(ns)))
cols = [(g, ch) for g in ["G1", "G2", "G3"] for ch in chans]
with open(f"{gd}/columns.tsv", "w") as w:
    w.write("col\tgene\tchannel\tn_snps\n")
    for j, (g, c) in enumerate(cols):
        w.write(f"{j}\t{g}\t{c}\t1\n")
np.stack([truth[c] for c in cols], 1).astype(np.float32).tofile(f"{gd}/scores.f32")
line = run(PY + [f"{B}/compare.py", gd, sd]).splitlines()[-1]
assert line.startswith("PASS"), line
print(f"2. ldp_sum_v0 sum mode, chunk layout: max rel diff {worst:.1e} vs brute force; {line}  OK")

# ---- 3. production 12d over the chunk layout with T = 3 pseudo-thresholds
spec = importlib.util.spec_from_file_location("prsnet12", f"{B}/prod_snapshot/12_prsnet.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
order = ["G0", "G1", "G2", "G3", "G4", "G5"]                 # a graph node order with genes that have no features
open(f"{W}/gene_order.txt", "w").write("".join(g + "\n" for g in order))
with open(f"{W}/pheno.txt", "w") as w:
    for i in range(ns):
        if i % 10 == 9:
            continue                                         # people without a phenotype are dropped
        w.write(f"F{i}\t{1000 + i}\t{1 + (i % 3 == 0)}\n")
a = types.SimpleNamespace(run_dir=W, chroms="19", gene_order=f"{W}/gene_order.txt", expect_n_genes=6, pheno=f"{W}/pheno.txt",
                          ancestry="EUR", out_dir=f"{W}/genome", export="memmap", sample_block=25, thresholds="1 2 3")
mod.cmd_assemble(a)
import json
meta = json.load(open(f"{W}/genome/feats.meta.json"))
assert meta["n_thresholds"] == 3 and meta["n_genes"] == 6
F = np.memmap(f"{W}/genome/feats.f32", dtype=np.float32, mode="r", shape=(meta["n_samples"], 18))
info = [l.split(",") for l in open(f"{W}/genome/info.csv")][1:]
assert len(info) == meta["n_samples"] == ns - ns // 10
for r, (iid, lab, anc) in enumerate(info):
    i = int(iid) - 1000
    for gi, g in enumerate(order):
        for t, ch in enumerate(chans):
            want = truth.get((g, ch), np.zeros(ns))[i]
            assert abs(F[r, gi * 3 + t] - want) <= 1e-7 + 1e-5 * abs(want), (iid, g, ch, F[r, gi * 3 + t], want)
print(f"3. 12d assemble: {meta['n_samples']} x 6 genes x 3 channels, gene-order placement, zero columns for G0/G5  OK")
print("all OK")
