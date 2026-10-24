# prs-feature-bench

A fixed test environment for versions of the PRS-Net per-gene feature scripts,
built so that the move from C+T gene scores to eQTL-weighted gene × cell-type
scores (the scPRS + PRS-Net combination) can be developed and optimized with
consistent, comparable measurements.

Every variant is run the same way on the same fixture and gets two numbers:
**fidelity** (does it reproduce the reference, cell by cell) and **cost** (wall
time, CPU time, peak RSS). Results accumulate in `results.tsv`, one row per run,
tagged with the git commit.

Status (Oct 21, 2026): scaffold written and smoke-tested on synthetic data
(`tests/smoke_test.py`); **not yet run on Bouchet**. The fixture has not been
built and no golden exists yet — those are steps 1–3 below.

---

## 1. Layout

```
prs-feature-bench/
├── README.md               this file
├── CLAUDE.md               rules for the Claude Code session that works here
├── make_fixture.sh         ONE-TIME, on Bouchet: builds fixture/ from ad/eur chr19
├── run_variant.sh          runs variants/<v>, times it, compares, logs a row
├── compare.py              fidelity check: golden vs candidate, scores layout
├── compare_dosage.py       exact check for the dosage layout (PRSNet-2 features, §9)
├── compare_bed.py          exact check for the bed layout (12a's cohort .bed, §11)
├── compare_clumps.py       exact check for the clumps layout (12b's index SNPs per gene, §11)
├── results.tsv             append-only log (created by the first run)
├── fixture/                the data (not in git; ~200 MB)
├── golden/<name>/          reference outputs (not in git)
├── runs/<variant>_<stamp>/ per-run output + run.log + compare.txt
├── lib/
│   ├── ref_12_prsnet.py    pinned copy of the production scorer (from make_fixture)
│   ├── from_chunks.py      12c chunk layout -> standard layout
│   ├── make_ct_weights.py  clumps -> long weights (ct variants)
│   └── dosage_from_raw.py  plink --recode A -> dosage layout (ref_dosage)
├── eqtl/
│   └── weights_to_plink2.py  long-format weights -> plink2 wide score file
├── tests/smoke_test.py     synthetic end-to-end test of the Python pieces
├── port/                   single submissions: prsnet_chain.sbatch (PRS-Net 12a-12d, §11); PRSNet-2 features sbatch + merge_shards.py (§10)
├── prod_snapshot/          READ-ONLY copies of the production scripts (12_prsnet.{sh,py}, trainer,
│                           config.sh, …); MANIFEST.txt = provenance + md5s. Two optimisation
│                           lines branch from them: PRS-Net (ct/dosage variants) and the eQTL addition
└── variants/
    ├── ref_ct/             reference: production 12b + 12c   -> golden/ct
    ├── ref_eqtl_plink2/    oracle: plink2 multi-column --score -> golden/eqtl_<set>
    ├── eqtl_sparse_v0/     first fast eQTL implementation (bed-reader + scipy)
    ├── ref_dosage/         reference: plink --recode A -> golden/dosage (PRSNet-2 features, §9)
    ├── dosage_v0/          first fast PRSNet-2 dosage implementation (packed-bed decode)
    ├── ref_bed/            reference: the fixture's own plink-made bed -> golden/bed (§11)
    ├── bed_seek_v0/        12a's cohort .bed by seeking the raw release, no plink (§11)
    ├── bed_seek_v1/        v0 with a fused gather+pack kernel (§11)
    ├── ct_sparse_v5/       ct_sparse_v3 writing 12c's chunk layout (§11)
    ├── ref_clump/          reference: 12b's per-gene plink --clump -> golden/clumps (§11)
    └── clump_np_v0/        12b without plink: one LD pass per chromosome + greedy per gene (§11)
```

`.gitignore` should hold `fixture/ golden/ runs/ lib/ref_12_prsnet.py`.

## 2. The contract

A variant is a directory `variants/<name>/` with a `run.sh` that takes
`<fixture_dir> <out_dir>` and leaves three files in `<out_dir>`:

| file | content |
|---|---|
| `samples.txt` | `FID<TAB>IID`, one per row, in the row order of `scores.f32` |
| `columns.tsv` | header `col gene channel n_snps`; one row per column; `channel` is `P<=1e-5` … for C+T, a cell type for eQTL |
| `scores.f32` | float32, C order, shape (n_samples, n_columns) |

Anything else the variant writes goes under `<out_dir>/work/` (deleted after a
PASS unless `--keep`). The variant receives `BENCH_THREADS`, `BENCH_LIB` and
`BENCH_PY` and must not read `config.sh` or anything outside the fixture.

**Fidelity rule.** For every (gene, channel) column present in the golden and
every sample: `|cand − gold| ≤ atol + rtol·|gold|`, rtol 1e-5, atol 1e-7
(the production `check` tolerance). Missing columns fail; extra columns are
reported but not compared. Rows and columns may be in any order.

**Second layout: dosage (PRSNet-2 features, §9).** A variant may instead leave
these files; `run_variant.sh` tells the layouts apart by the presence of
`X.npy` and compares with `compare_dosage.py`, exactly (no tolerance):

| file | content |
|---|---|
| `samples.txt` | `FID<TAB>IID`, row order of `X.npy` |
| `snps.tsv` | header `col snp a1 bp p`; one row per column; `a1` = the counted allele |
| `X.npy` | int8, shape (n_samples, n_snps): copies of `a1`, 0/1/2, **3 = missing** |
| or `X2.npy` | the same codes 2-bit packed, uint8 (n_samples, ⌈n_snps/4⌉), SNP j in bits 2(j mod 4) of byte ⌊j/4⌋ (`lib/x2bit.py`); `compare_dosage.py` unpacks it, so either side may use either |
| `pvalues.npy` | float64 (n_snps,): GWAS P per column |
| `gene2snps.tsv` | header `gene n_snps snps`; `snps` = comma-separated snp ids |
| `gene2snps.pkl` | dict gene → list of column indices (what the PRSNet-2 loader reads) |

PASS: every golden sample and SNP present, same `a1`, every cell of `X`
equal, same P, and the same SNP set for every golden gene. Rows and columns
may be in any order; a golden and a candidate must share a layout.

## 3. First run on Bouchet (in a small `salloc`, e.g. 2 CPU / 8 GB)

```bash
cd ~ && git clone <wherever you put it> prs-feature-bench && cd prs-feature-bench
# python: the production venv already has numpy/scipy/bed-reader
export BENCH_PY=$ROOT/tools/prsnet_venv/bin/python     # whatever PRSNET_VENV is in config.sh

# 1. fixture: 2,000 people of ad/eur, chr19, every window SNP (~3 min)
DISEASE=ad COHORT=eur CHR=19 N=2000 bash make_fixture.sh

# 2. reference C+T run -> golden/ct  (~1,500 plink --clump calls on 2k people: a few minutes)
bash run_variant.sh ref_ct --as-golden ct --threads 2

# 3. prove the harness detects a change: rerun the reference, expect PASS
bash run_variant.sh ref_ct --tag "repeat of golden"
```

Then the eQTL side, once a weight file exists (section 5):

```bash
WEIGHTS=onek1k bash run_variant.sh ref_eqtl_plink2 --as-golden eqtl_onek1k
WEIGHTS=onek1k bash run_variant.sh eqtl_sparse_v0 --golden eqtl_onek1k
```

Two things to expect on the first `ref_eqtl_plink2` run: plink2's `cols=`
modifier spelling and the `_SUM` column names have changed across releases, so
the script prints the `.sscore` header and asserts the column order — if it
trips, adjust the `--score` line once and record the plink2 version in this file.

## 4. What the fixture is and is not

The fixture is chr19 of the ad/eur PRS-Net build (the chromosome whose 12c
output passed `check` against plink2 to 4.5e-6), subsampled to N=2,000 people
stratified by case/control, with the same variants and the same A1 orientation.
`make_fixture.sh` verifies the `.bim` A1 column is byte-identical to the source.

Because LD is estimated in the fixture's 2,000 people, **the C+T index sets and
scores differ from production**. That is fine: the harness tests
*implementation equivalence* (does variant B compute what variant A computes),
not production numbers. The production gate stays `12_prsnet.sh check` on the
cluster, run once on the candidate that wins here.

At 2,000 × 305k the fixture is too small to show NFS or memory effects. When a
variant is about throughput at 381k people, make a second fixture with
`N=50000 CHR=19` (≈3.8 GB) and point `run_variant.sh` at it with
`BENCH_FIXTURE=…`. Implemented: `fixture_20k` (N=20,000) is the timing
fixture used from §7 item 2 on; goldens live in `<fixture>/golden/`. The
scale-up plan for the PRSNet-2 path (§9) is `fixture_100k`, then the whole
ad/eur chr19 cohort, each made with
`BENCH_FIXTURE=$BENCH/fixture_<tag> N=<n> CHR=19 bash make_fixture.sh`
inside an salloc (the raw fallback streams the chromosome once).

## 5. The eQTL weight file (`fixture/weights/<set>.tsv`)

Long format, tab-separated, header `gene snp a1 channel beta`: one row per
(gene, SNP, cell type), `beta` = effect on expression per copy of `a1`, `snp`
= the fixture's `.bim` id (chr:pos, hg19). Duplicate (gene, snp, channel) rows
are an error in the source — the oracle overwrites, v0 sums; de-duplicate
upstream.

Source: OneK1K sc-eQTL (Yazar et al. 2022, Science) — 14 PBMC cell types,
summary statistics on GWAS Catalog / the OneK1K portal. The conversion script
is **not written yet**; it needs (1) the download location and column names of
the current release, (2) a liftover if the release is hg38, and (3) a choice of
which eQTLs to keep as weights (lead eQTL per gene × cell type, or all
conditional eQTLs, or a cis-window with marginal betas — this choice *is* the
"what does a node mean" question and should be recorded as a deviation from
both papers). Until then `ref_eqtl_plink2` and `eqtl_sparse_v0` can be
exercised with a synthetic weight file (see `tests/smoke_test.py` for the
format).

The computation for an eQTL channel is `score_ig c = Σ_j w_jgc · dosage_ij`
with missing calls mean-imputed — **no clumping, no thresholds, sums not
averages**. 12b disappears entirely; 12c's one-pass structure is the thing to
keep and widen.

## 6. Working method

- One change per variant. Copy the closest existing variant to a new
  directory, change one thing, run it, commit. The results row carries the
  commit, so `results.tsv` is the experiment log.
- Fidelity first. A variant that fails is not faster, whatever its wall time.
  Deliberate numeric deviations (a float16 store, a subsampled LD panel) get a
  new golden with a name that says so, never a loosened tolerance.
- Measure at fixed threads. `--threads` is recorded; compare rows at equal
  thread counts.
- The reference variants (`ref_*`) are never edited. If a reference is wrong,
  that is a bug in production and goes back to the cluster scripts first.

## 7. Candidate optimizations (the order they are worth trying)

For the eQTL path, starting from `eqtl_sparse_v0`:

1. **Read only SNPs that carry weight** (v0 already does) and sort them so bed
   reads are sequential per block.
2. ~~**Decode dosages from the packed bed bytes directly** (2 bits/genotype →
   int8) instead of float32 through bed-reader; impute from per-SNP allele
   counts computed in the same pass.~~
   **Done: `ct_sparse_v2`** (fixture_20k, 2 threads, 1 GiB chunks; row
   `20261021_183438` vs `ct_sparse_v1` row `20261021_181102`). PASS, bit-identical
   to golden/ct (max|diff| 0). Kernel 16.6 s vs 22.1 s (−25%); wall 1:35.92 vs
   1:32.39 is clumping noise. Kernel split: read 0.1 s, decode+impute 3.3 s,
   cast-to-float64 3.9 s, sparse matmul 5.4 s, write 1.0 s — the float64 path
   (the block cast plus scipy's `(c, n)` temporary) is 9.3 s of 16.6 and is the
   next target. Peak RSS 3700 MB vs 2667 MB: the `.bed` memmap's file-backed
   pages count toward RSS; read the block's row span with `np.fromfile` instead.
   **Follow-up done: `ct_sparse_v3`** (rows `20261021_184355`, `20261022_130215`
   vs v2 row `20261021_183438`, same chunking) — the float64 block + scipy
   product replaced by a per-SNP axpy into the transposed chunk result, and the
   `.bed` read with `np.fromfile`. PASS, bit-identical. Kernel 12.9 s / 10.7 s
   vs 16.6 s (split: read 0.3, decode+impute 3.1, scatter 2.1, write 2.7 — the
   write is now a transposed copy); peak RSS 1952 MB vs 3700 MB. The two v3
   runs were on different nodes; ~2 s is the kernel noise floor. The inner
   loop is Python per nonzero weight, so the many-columns-per-SNP eQTL case
   must be measured before this kernel is adopted for `--mode sum`.
   **eQTL path, `eqtl_sparse_v1` = the v3 kernel in `--mode sum`** (fixture_20k,
   2 threads; rows `20261022_131041` and `20261022_132450` vs `eqtl_sparse_v0`
   rows `20261022_131033`, `20261022_132352`; goldens `fixture_20k/golden/eqtl_synth`
   and `eqtl_synth14` from `ref_eqtl_plink2`). PASS at the same max|diff| as v0.
   `synth` (3 ch × 10 SNPs, 4,575 cols): 3.49 s / 1214 MB vs 5.59 s / 1969 MB.
   `synth14` (14 ch × 30 SNPs, 21,350 cols, 637k nnz): 19.54 s / 2519 MB vs
   27.25 s / 5217 MB; kernel split read 1.9, decode+impute 2.6, scatter 7.9,
   write 3.7 s. The scatter runs 14 axpys per SNP at ~1.6 GFLOP/s; an eQTL
   weight matrix is block-dense per gene (k_g SNPs × C cell types), so the next
   step is one small GEMM per gene. **Note on the oracle:** plink2 on
   `synth14` at 20k people took 12:11.93 and 31.5 GB (row `20261022_131138`);
   it is a fixture-scale tool only, and the production-scale gate for the eQTL
   step needs a different reference (a validated sparse implementation, or
   plink2 on a column subset) — a design decision to record in `claude/`.
   **Follow-up done: `eqtl_sparse_v2` = one GEMM per gene** (fixture_20k,
   2 threads; rows `20261022_133027`, `20261022_133033` vs v1 rows
   `20261022_131041`, `20261022_132450`). PASS at the same max|diff| as v1.
   `synth14`: 10.82 s / 3020 MB vs 19.54 s / 2519 MB (−45% wall); kernel
   10.6 s vs 19.3 s, split read 2.4, decode+impute 3.4, gemm 1.0, write 1.6 s
   — the 7.9 s scatter became a 1.0 s GEMM as expected. `synth`: 3.56 s vs
   3.49 s, even (the 3-channel case was never scatter-bound). The higher RSS
   is the chunk budget, not the kernel: v1 counted the matmul temporary and
   so ran 0.5 GiB chunk blocks at `--chunk-gb 1.0`; v2 has no temporary and
   runs the full 1.0 GiB block (4 chunks of ≤ 6710 cols vs v1's 7 of 3355).
   Equal-memory comparison (row `20261022_133422`, `CHUNK_GB=0.5`, 7 chunks
   like v1): 11.51 s / 2703 MB vs v1's 19.54 s / 2519 MB — the +180 MB is the
   int8 block and its missing mask. Threads do nothing (row `20261022_133504`,
   4 threads: 10.97 s, gemm still 1.0 s): the per-gene tiles are too small
   for BLAS threading, so the kernel is one core's worth of decode, read and
   write. Remaining split read 2.4 + decode+impute 3.4 + gemm 1.0 + write 1.6
   of 10.6 s. A local micro-benchmark of one 4000-SNP block at n = 20k found
   the per-gene `astype(float64)` dominated by the fresh 4.8 MB allocation it
   makes each call (0.13 s vs 0.04 s cast into a reused buffer), and the three
   int8 passes for sums and the missing mask cost more than the same statistics
   from the packed bytes via 256-entry tables.
   **Next: `eqtl_sparse_v3`** — no int8 block; per-SNP means from byte
   tables; each gene decoded and cast into two preallocated buffers, imputed
   in place with `np.copyto(where=)`; genes ordered by first SNP row so a
   block's row span stays tight for any weights-file order. Bit-identical to
   v2 on the smoke data; the local block benchmark went 0.40 s → 0.25 s for
   decode+impute+gemm, so expect ~1.5 s off the synth14 kernel.
   **Done: `eqtl_sparse_v3`** (rows `20261022_134816`, `20261022_134821` vs
   v2 rows `20261022_133027`, `20261022_133033`, same chunking). PASS at the
   same max|diff|. `synth14`: 7.40 s / 2797 MB vs 10.82 s / 3020 MB (−32%);
   kernel 7.1 s vs 10.6 s, split read 0.3, decode+impute 2.1, gemm 1.0, write
   1.6 s. `synth`: 2.55 s / 1201 MB vs 3.56 s / 1588 MB. decode+impute fell
   3.4 → 2.1 s as predicted, but the larger win was unplanned: read fell
   2.4 → 0.3 s, so v2's genes (weights-file order) were not in .bim order and
   each block's row span over-read heavily; ordering genes by first SNP row
   fixed it. Of the 7.1 s kernel, 2.1 s is now outside the timed parts
   (weights parsing of 637k rows in Python, setup), and does not scale with
   n; the parts that do are decode+impute 2.1, write 1.6 (the transposed
   copy of the (c × n) chunk result into the (n × m) memmap), gemm 1.0.
   **Next: `eqtl_sparse_v4`** — sample-major chunk result. The transposed
   S^T layout came from ct_sparse_v3's axpy loop and the GEMM does not need
   it: `np.matmul(G_g.T, W_g, out=S[:, a:b])` writes the column slice through
   BLAS at the same speed as the contiguous form (local 1 GiB chunk: 0.117 s
   vs 0.121 s), and the memmap write becomes a row-contiguous float64→float32
   copy (0.09 s vs 0.86 s per chunk locally). Bit-identical to v3 on the smoke
   data; expect write 1.6 → ~0.2 s on synth14.
   **Tried, slower: `eqtl_sparse_v4`** (rows `20261022_135350` 2 threads,
   `20261022_135428` 1 thread vs v3 row `20261022_134821`). PASS, but synth14
   8.08 s vs 7.40 s. Two things the local benchmark got wrong on Bouchet's
   BLAS: the `G_g^T @ W_g` orientation (M = n tall, K = 30, N = 14, strided C)
   runs at half the speed of v3's `W_g^T @ G_g` (gemm 2.2 s vs 1.0 s at 2
   threads, 4.1 s at 1 thread — so this shape does use threads while v3's did
   not), and the row-contiguous memmap write only fell 1.6 → 1.1 s, not to
   0.2 s; with 4 column chunks each row is written as four 27 KB segments and
   the memmap's page-fault/dirtying cost does not go away. synth (1 chunk,
   full rows): write 0.3 → 0.2 s, gemm 0.4 → 0.9 s. v3 stays the best.
   The per-gene tile transpose (GEMM into a (C_g × n) buffer, then
   `S[:, a:b] = tile.T` while hot) was benchmarked locally at 0.15 s per
   chunk — about what the transposed write costs on Bouchet beyond v4's
   contiguous one — so it was not made a variant.
   **Next: `eqtl_sparse_v5`** — v3's kernel unchanged; `scores.f32` written
   with `os.pwrite` (one call per row segment, one per 256 MB step when a
   chunk spans all columns) instead of through the numpy memmap, selectable
   with `WRITE=pwrite|mmap`; and the setup phases (weights parse, permute,
   plan) plus the kernel's untimed remainder logged, to attribute the 2.1 s
   outside the timed parts. Byte-identical output to v3 on the smoke data.
   **Tried, much slower: `eqtl_sparse_v5 WRITE=pwrite`** (row `20261022_140304`
   vs the same commit with `WRITE=mmap`, row `20261022_140359`). PASS, but
   write 19.1 s vs 2.0 s on synth14: with 4 column chunks the pwrite path
   issues 80k 27 KB partial-row writes, and on the cluster filesystem each is
   a round trip; with one chunk (synth, whole rows) it is 0.3 s, same as the
   memmap. The memmap stays. The setup line answers the other question:
   weights parsing is 1.2 s of the 2.1 s (637k rows in Python), "other"
   (csc slicing, const add, chunk loop) 0.8–1.2 s. Neither scales with n.
   **Where the eQTL path stands (fixture_20k, 2 threads):** `eqtl_sparse_v3`
   7.40 s / 2797 MB on synth14 vs `ref_eqtl_plink2` 12:11.93 / 31.5 GB (row
   `20261022_131138`) and `eqtl_sparse_v0` 27.25 s / 5217 MB; synth 2.55 s vs
   5.59 s. Of the remaining 7.1 s kernel, the parts that scale with n are
   decode+impute 2.1 (float64 tile per gene, bandwidth-bound), write 1.6–2.0
   (memmap transpose) and gemm 1.0; the rest is fixed cost. This is the
   kernel to port to production (`12e`); further gains need a compiled
   decode+GEMM, not more numpy.
3. ~~**Column chunking**: with ~20k genes × 14 cell types = 280k columns, `S` at
   381k people is 850 GB in float64 — chunk the columns (by gene block) and
   write each chunk's float32 to the memmap as it finishes; the SNP reads are
   then repeated per column chunk, so size the chunk to the memory budget.~~
   **Done: `ct_sparse_v1`** (fixture_20k, 2 threads, rows `20261021_181102`
   and `20261021_181256` vs `ct_sparse_v0` row `20261021_153458`). PASS at the
   same max|diff| 5.96e-8. Wall 1:32 / 1:37 vs 1:29 (noise; the kernel is
   22.1 s / 19.8 s vs 20.8 s, the rest is clumping). Peak RSS 2667 MB at 1 GiB
   chunks and 1972 MB at 0.25 GiB chunks vs 3984 MB; the kernel's memory is
   now set by `--chunk-gb`, not by n × m. SNP re-reads 1.23× / 1.34× of the
   unique set. The chunks are unbalanced: the first 76 genes of the manifest
   hold 22k of the 44k index SNPs, so a later variant should size chunks by
   nnz rather than by column count.
4. **One job per chromosome, no 12b**, resource lines at ≤16 GB / ≤4 h to
   backfill on this account.
5. For the C+T path the same list applies to 12c, plus A1/A4 from
   `prsnet_features.md` (one LD table per chromosome; per-bin sums + counts).
   ~~**Kernel side: `ct_sparse_v4`** = ct_sparse_v3's run.sh (12b clumping
   unchanged) with eqtl_sparse_v3's kernel in `--mode ct`, plus genes
   position-sorted in ct mode too.~~ **Done, PASS but not a win** (fixture_20k,
   2 threads, row `20261023_125813` vs ct_sparse_v3 rows `20261022_130215`,
   `20261021_184355`): bit-identical to golden/ct; wall 1:20.07 vs 1:26.41 /
   1:31.67 is clumping noise, the kernel is 13.7 s vs 10.7 / 12.9 s (read 0.7,
   decode+impute 7.0, gemm 1.5, write 2.1), and peak RSS 4970 MB vs 1952 MB.
   Two reasons, both ct-specific: the per-gene decode re-reads 1.81× the
   unique SNPs (79,639 row decodes for 43,888 index SNPs) where v3 decoded
   each block once, and the per-gene buffers are sized by the largest gene,
   which on chr19 is the giant-window ERICH4 (113,666 window SNPs). The
   per-gene GEMM kernel stays the eQTL choice; `ct_sparse_v3` stays best for
   C+T.
   **A1 shelved (2026-10-22):** the target model turned out to be PRSNet-2,
   whose inputs need no clumping at all (§9). The C+T path stays as is unless
   the original PRSNet is wanted too.

## 8. Relationship to the production tree

Production: `$ROOT/scripts/slurm/12_prsnet.{sh,py}` and `submit_12b_split.sh`
(design in project docs `claude/prsnet.md`, `claude/prsnet_features.md`). The
bench never writes there. A variant that wins here is ported back as a new
step (`12e`, say), run on one chromosome, and gated by `12_prsnet.sh check` or
its eQTL equivalent before it touches the multi-disease plan.

## 9. PRSNet-2 dosage features

PRSNet-2 (Li, Shan, …, Zhang; bioRxiv 10.1101/2025.11.22.689899, Nov 2025;
code `github.com/lihan97/PRSNet-2`) replaces the original PRSNet's per-gene
C+T PRS with a learned multi-kernel aggregator over raw genotypes. What the
paper and the released example data (`example_data/AD`) fix as the input:

- SNPs with GWAS **P < 0.01** (default; "Only SNPs with a P-value below this
  threshold are retained"), **no clumping**.
- A SNP is linked to a gene "if the SNP falls within the gene body extended by
  **10 kb** both upstream and downstream" (18,963 protein-coding genes); one SNP
  may link to several genes.
- `X.npy` int8 (n_samples × n_snps) with values 0/1/2 and **3 = missing**
  (`src/dataset.py` does `X[X==3] = 0`); `pvalues.npy` float64; `gene2snps.pkl`
  dict gene → SNP column indices. `scripts/train.py` loads `X.npy` whole.
- The preprocessing notebook the README names (`scripts/get_data.ipynb`) is not
  in the repository, so the exact gene annotation and missing-call handling
  upstream of `X.npy` are unpublished.

Bench decisions (recorded here because they are not derivable from the code):
the counted allele is the fixture's `.bim` A1 = GWAS effect allele (plink
`--recode A` counts the same allele, so a positive beta means risk-increasing
dosage); missing stays 3, as the released loader expects; a gene's window is
the fixture's `genes/<gene>.snplist`, i.e. production 12a's window — which
`config.sh` shows **is** the paper's rule: the windows come from the authors'
`gene_bed_files_10kb` (19,831 genes, hg19, ±10 kb baked in), so ERICH4's
42 Mb window is the authors' annotation, not ours; SNPs significant but in
no gene window are dropped (the model only
ever gathers SNPs through `gene2snps`). Columns are in `.bim` order, so each
gene's SNPs are a near-contiguous run.

What carries over from §7: the packed-bed LUT decode (now straight to the
output dtype, no float, no GEMM, no imputation), position-ordered span reads,
and the memmapped transposed write. At 20k × M the whole thing is one pass;
at UKB scale (411k × ~500k ≈ 200 GB int8) the production port needs
per-chromosome shards and a loader that does not `np.load` the matrix whole.

Variants: `ref_dosage` (awk selection + plink 1.9 `--recode A` +
`lib/dosage_from_raw.py`; makes `golden/dosage`) and `dosage_v0`
(`variants/dosage_v0/dosage.py`, one Python pass). Knobs: `PTHR` (default
0.01), `BLOCK` (selected SNPs per decode block, 4000). Smoke tests 23–24
cover both against a brute-force `bed-reader` read and `compare_dosage.py`'s
PASS/FAIL behaviour. `dosage_v1` = v0 with each block read as runs of
selected rows no more than `MAX_GAP` (256) rows apart, so the read volume is
about selected + gaps and peak memory is the decode buffer, not the span;
smoke test 25 shows it byte-identical to v0 at gaps 1, 3 and 1000.
**Done, PASS at both sizes** (2 threads, commit 7115489):

| fixture | row | variant | wall | kernel (select / read / decode / write) | peak RSS |
|---|---|---|---|---|---|
| 20k | `20261023_130230` | dosage_v0 | 1.55 s | 0.4 / 0.2 / 0.1 / 0.5 | 1009 MB |
| 20k | `20261023_134137` | dosage_v1 | 1.18 s | 0.4 / 0.1 / 0.1 / 0.5 | 305 MB |
| 100k | `20261023_134424` | ref_dosage | 8.88 s | plink --recode A + convert | 1552 MB |
| 100k | `20261023_134433` | dosage_v1 | 4.71 s | 0.9 / 0.8 / 0.6 / 2.2 | 1270 MB |

v1 reads 64,121 rows in 182 spans (12.6× the 5,078 selected) instead of the
whole chromosome. 20k → 100k scales linearly (kernel 1.1 → 4.6 s; the write
is half of it), so the full ad/eur cohort on chr19 is ~20 s, and the genome
(~22× the SNPs) a few minutes of single-core work across 22 jobs. The
`--recode A` reference scales the same way and is itself only 2× slower, so
this step is not where time goes any more; 12a is.

**`dosage_raw_v0`** (smoke test 26, Bouchet run pending) does without 12a:
it seeks to the selected rows of the raw release chromosome named in the
fixture's `params.json` (`source_bed`, when `source_mode` is `raw`), decodes
all ~488k raw people, gathers the fixture's people (`--keep`), flips the SNPs
the raw `.bim` stores on the other allele to the fixture's A1 (`--a1-allele`),
and writes the same layout. Output columns stay in fixture `.bim` order even
when the raw order differs (then the block write is a column scatter). The
smoke test feeds it a shuffled raw release with extra people and SNPs and half
the shared SNPs on the other allele; it reproduces dosage_v0 cell for cell.
This is the one variant that reads outside the fixture, read-only, by design.
**Done, PASS at both sizes** (commit 8381c8c, 2 threads; rows
`20261023_140043` 20k, `20261023_140053` 100k): identical to golden/dosage
made from the 12a cohort bed. Wall 8.48 s / 8.72 s, RSS 471 / 1094 MB. The
raw chr19 is 624,503 variants × 487,409 people (121,853 bytes per row); the
5,078 wanted rows were read as 5,574 rows in ~3,960 reads, 679 MB, in 4.2 s
cold (0.2 s on the second run, page cache), 2,094 of them flipped to the
fixture A1. Decode of all 487k people 2.1–4.2 s, gather 0.4–1.0 s, write
0.6 / 2.2 s. So the 12a pass for this feature set — 4 h 10 min for the
genome — becomes ~10 s per chromosome, and the read volume genome-wide is
(selected SNPs × 122 KB), tens of GB in total. In production the `snps`
sub-step of 12a (per-gene snplists from the windows, `gwas.QC.txt` and step
04's `.PH.bim`, with the giant-window cap) is still needed; it is Python
and cheap. plink is not needed at all on this path.

**Goal for this path (set 2026-10-23):** training takes ~3 h regardless, so
the feature build should take minutes. The baseline is the ad/eur PRS-Net
build (380,962 people, 22 chromosomes; `claude/prsnet_features.md` §C.7):
12a, one plink pass over the raw UKB release per chromosome (chr1 = 258 GB)
to a cohort bed of window SNPs, 4 h 10 min total and NFS-bound; 12b per-gene
clumping ~2 h; 12c scoring ~2.5 h; 12d stacking minutes. (The "24 h" was
12b's original per-chromosome request, which never backfilled.) A PRSNet-2
build replaces 12b + 12c (~4.5 h) with the dosage step, whose fixture cost is
1.5 s for 20k × 5k cells on one core — the cohort × genome is ~2,000× more
cells, so ~an hour of single-core decode spread over 22 jobs, i.e. minutes.
After that 12a is ~95% of what is left, and for this path it is avoidable:
only SNPs with P < 0.01 are needed (5,078 of 162,703 window SNPs on chr19),
so a step that seeks to just those rows of the raw release — `--keep` as a
column gather after the decode, the effect-allele orientation as a per-SNP
choice of lookup table — reads ~3% of what 12a streams. That is the next
variant once the raw release layout is known (not reproducible on the
fixture; it has to read the real chromosome in an salloc, read-only).
The stored layout: the trainer is Nick's own re-implementation of the paper,
to become one Slurm submission with the feature build, so the layout is ours
to choose. GPU nodes have 1–2.3 TB of RAM, so even the int8 matrix (~190 GB
for the cohort) fits; what the layout decides is load time and disk. The
2-bit sample-major store **`X2.npy`** (`lib/x2bit.py`, §2) is a quarter of
the bytes (~48 GB for the cohort), each person's genome one contiguous row
(~125 KB), a batch of any 512 people a row gather plus `unpack()` — 64 MB,
~0.3 s on one core, hidden by DataLoader workers. **`dosage_pack_v0`** =
dosage_raw_v0 writing X2.npy (blocks start on 4-column boundaries so each
block is a byte slice); smoke test 27 checks the pack/unpack round trip, that
the packed store unpacks to dosage_v0's matrix, and that `compare_dosage.py`
accepts X2 on either side. `lib/x2bit.py` is the file the trainer's loader
imports. **Done, PASS at both sizes** (commit d9d05f7; rows `20261023_141951`
20k: 4.45 s, 479 MB; `20261023_141956` 100k: 6.24 s, 1015 MB; X2.npy 127 MB
for 100k × 5,078). The write fell from 2.2 s (int8) to 1.4 s at 100k; a random
512-person batch off the cluster filesystem took 4 ms cold through
`x2bit.batch`. `dosage_pack.py` also takes production's inputs directly:
`--keep <FID IID list>` and `--a1 <SNP allele>` (12a's `KEEP_LIST` and
`target_data.a1`); smoke test 28 checks both are honoured. `x2bit.Sharded`
reads per-chromosome shards as one matrix (smoke tests 28–29).

## 10. The single submission (`port/`)

Goal: the PRSNet-2 feature build as one submission, minutes instead of
hours; the features and the model stay exactly PRSNet-2's (Nick's trainer is
not touched here). Run on Bouchet 2026-10-23, results below. The pieces:

- `port/prsnet2_features.sbatch` — a 22-task array; task = chromosome. Each
  task runs 12a's `snps` sub-step (per-gene lists from the windows,
  `gwas.QC.txt`, step 04's `.PH.bim`, giant-window cap) and then
  `dosage_pack.py` against the raw release with `KEEP_LIST` and
  `target_data.a1`, writing `$OUT/chr<k>/X2.npy` and sidecars. No plink, no
  cohort bed. Variable names are those 12a uses from `config.sh`; the first
  run will tell whether `PRSNET_PY`/`PRSNET_SCRIPT`/`PRSNET_BIG_SNPS` are
  spelled as assumed.
- `port/merge_shards.py` — genome-wide `snps.tsv`, `pvalues.npy`,
  `gene2snps.{tsv,pkl}` with column offsets, and `shards.tsv`; checks all
  shards share `samples.txt`. Smoke test 29.
- `port/prsnet2_store.py` — `DosageStore(data_dir, load="ram", gene_order=…)`,
  the trainer's view of the merged directory, shaped like `train_prsnet.py`'s
  `FeatureStore`: `.n .m .labels .sample_ids`, `.rows(idx)` → float32 torch
  with missing (3) → 0 as the authors' `src/dataset.py` does, `.rows_int8(idx)`.
  `load="ram"` holds the packed genome as one (n, 21,611) uint8 array (8.2 GB
  for ad/eur) so a batch is a row gather plus 22 shift-or unpacks;
  `load="memmap"` leaves the shards on disk. On `runs/port_timed` (memmap, in
  the salloc, cold NFS): opened in 2 s, three random rows match
  `x2bit.Sharded`, `model_inputs` gives 19,831 nodes, 9,721 genes with SNPs,
  199,019 snp_ids (gene_ids start 2, 5, 7 = A2M, A4GALT, AAAS); a 512-person
  batch took 14.4 s because it is 512 × 22 random row reads off NFS — the
  memmap mode is for evaluation passes, training uses `load="ram"`. Same
  directory with `--load ram` (same salloc, shards now in page cache): the
  7.7 GiB packed genome loaded in 8 s, store open in 10 s, **a 512-person
  batch in 77 ms** (gather + 22 unpacks + 3→0 + float32), so 20,000 training
  steps spend ~26 min in the loader on one core, hidden behind the GPU by
  DataLoader workers. `.model_inputs()` gives
  `PRSNet2(...)` its `n_snps, n_genes, gene_ids, gene2snp_len, snp_ids,
  pvalues` with `gene_ids` = node indices from `gene_order.txt`, ascending,
  exactly as the authors' `train.py` sorts `gene2snps` by key. `--export <dir>`
  writes the authors' own layout (`X.npy` int8 n × m, `Y.npy`, `pvalues.npy`,
  `gene2snps.pkl` with node-index keys; 33 GB for ad/eur) for a trainer that
  wants the whole matrix in memory as their `train.py` does. Smoke test 30;
  the CLI checks three random rows against `x2bit.Sharded` and times a batch.
  Shard paths in `shards.tsv` are resolved as written, then relative to the
  directory, then as `chr<k>/X2.npy`, so a moved directory still opens.

**First full run (2026-10-23, commit 278d4d4, inside a 4-CPU salloc, two
chromosomes at a time, ad/eur 380,962 people).** The chr19 dry run PASSED
against `fixture_20k/golden/dosage` with the fixture's 20,000 people matched
by IID inside the cohort (380,962 × 5,078, 14.6 s kernel). 21 of 22
chromosomes then completed; chr10 stopped at rs2201904: the counted allele
(target A1 = A) is neither raw allele (C/T) — a strand flip. Step 03's
`mismatch.R` complements such target SNPs; the raw release keeps the original
strand, so `dosage_pack.py` now matches the counted allele through its
complement as well (A→T etc.), reports how many SNPs were matched that way,
and still errors on anything else. Smoke test 27's synthetic release stores
every third shared SNP on the other strand. chr10 then ran in 15.2 s (4,287
SNPs, 2 of them matched on the other strand; the other 21 chromosomes had
none, or they would have stopped the same way).

**All 22 shards merged:** 380,962 people × **86,402 SNPs**, 9,721 genes with
≥ 1 selected SNP, 199,019 gene–SNP links, 3,162 cases in `info.csv`; packed
store 8.2 GB (8.9 GB with sidecars). Kernel time for the genome 317 s; wall
about 5–6 min at two chromosomes at a time in a 4-CPU salloc, no plink, no
cohort bed. The baseline for the same inputs was 12a 4 h 10 min + 12b ~2 h +
12c ~2.5 h of compute plus queue waits.

**Headline wall time (fresh directory, four chromosomes at a time, same
4-CPU salloc, commit 80a31de):**

```
time (seq 1 22 | xargs -P 4 ... port/prsnet2_features.sbatch ; merge_shards.py --pheno ...)
real    2m11.046s    user 2m54s    sys 0m49s
```

**2 min 11 s for the whole genome plus the merge**, against 12a 4 h 10 min
+ 12b ~2 h + 12c ~2.5 h (~8.7 h of compute, more with queue waits): about
240×. Kernel times per chromosome: chr1 31.2 s (8,554 SNPs, 17.7 s of
seeks), chr5 23.2, chr11 22.1, chr6 21.6, chr8 20.5, chr7 20.4, chr2 19.4,
chr17 18.3, chr19 16.3, chr10 15.2, chr3 15.8 … chr21 2.5 s; 317 s in total,
reads 60–70 % of it. Selected SNPs: 86,402 (P < 0.01 of the Wightman AD GWAS
inside gene windows), far fewer than the paper's "half a million" on its
GWASs. No task was killed for memory at four at a time in 16 GB.

**Full-cohort check:** `runs/port_full/chr19` (380,962 people) compared
against `fixture_100k/golden/dosage`, the golden plink made from the 12a
cohort bed: `PASS: 5078 snps, 0 bad snps, 0 bad cells, 0 missing, 0 bad
genes` with the 100,000 people matched by IID. Together with the 20k check
that is every selected chr19 SNP verified cell for cell on 100k of the
cohort's people.

**Correctness difference, confirmed from production's own files:** 12a ran
plink `--a1-allele target_data.a1` against the raw release; for a
strand-flipped SNP plink cannot assign that allele and keeps the raw
orientation. `prsnet/chr10/target_chr10.PH.log` holds 594 "Impossible A1
allele assignment" warnings (over every variant of the chr10 PH bed, not
just the selected ones). rs2201904 was then an index SNP in 12b's
`.clumped` output for its gene, and 12c's `load_gene_plan`
(`prod_snapshot/12_prsnet.py:300-306`) skips an index SNP whose gwas A1
matches neither target allele (`n_nomatch`, "skipped, as plink2 does"), so
production scored without it. The new path counts such SNPs on the
complemented strand as step 03's `mismatch.R` intended. On ad/eur this
touches 2 of 86,402 SNPs (both chr10); the two feature sets are otherwise
the same SNPs with the same counted alleles. Nothing in either path needs
changing for this; it is recorded so the difference is not mistaken for a
bug later.

First Bouchet runs (fixture_20k, 2 threads, commit d7909c0): both select the
same 5,078 SNPs (P < 0.01: 5,365 in `SNP.pvalue`, 287 not in the `.bim`),
838 of 1,525 genes, 10,421 gene-SNP links. `dosage_v0` (row `20261023_125956`)
wrote the 20,000 × 5,078 int8 matrix in 1.32 s wall (select 0.4 s, read 0.2,
decode 0.1, write 0.5), peak RSS 1010 MB because a 4,000-SNP block's span
covers almost the whole chromosome (156,359 rows read for 5,078 selected,
30.8×; a later variant should split spans at large row gaps). `ref_dosage`
(row `20261023_125954`) failed its own allele check: plink 1.9 re-labels A1
as the minor allele when writing a new fileset, so `--recode A` counted the
other allele for rs72984410. Fixed with `--keep-allele-order` and an explicit
`--recode-allele` list taken from the `.bim`.

**`dosage_v0` PASSES** (rows `20261023_130227` ref_dosage → golden/dosage,
`20261023_130230` dosage_v0; fixture_20k, 2 threads, commit bf60f7d): all
5,078 × 20,000 cells, alleles, P and 838 gene sets identical. Wall 1.55 s vs
the reference's 2.31 s — on 20k people the reference is already fast, so the
bench question for this path is not speed but memory and shape: peak RSS
1009 MB vs 367 MB (the whole-chromosome span read), and at 411k × 500k the
matrix is ~200 GB, which neither `np.load` nor one memmap per job handles.
Gene load is skewed by the giant window: ERICH4 carries 2,224 of the 5,078
SNPs, the next genes 382 (MARK4), 177 (PVRL2), 154, 147. The `p` column of
`snps.tsv` was written as `np.float64(...)` in these two runs (numpy 2 repr);
fixed to a plain float, so golden/dosage should be remade once.

## 11. Replicating PRS-Net: the 12a / 12b / 12c chain without the hours

Decided 2026-10-23: the goal is the ORIGINAL PRS-Net (the trainer
`train_prsnet.py` in `prod_snapshot/` reads 12d's `feats.f32`, gene × 11
thresholds, and stays as it is), not PRSNet-2 (§9–10, kept but shelved). The
features are the production chain: 12a cohort bed (window-union SNPs, `--keep`
cohort, `--a1-allele target_data.a1`) → 12b plink per-gene clump (r² 0.5,
250 kb, p1 1, index-first) → 12c scores at 11 thresholds → 12d `feats.f32`.
Each stage is replaced separately and checked against the production artefact
of that stage:

| stage | production | replacement | golden on Bouchet |
|---|---|---|---|
| 12a | plink streams the 3.65 TB release, 4 h 10 min | `bed_seek_v0`: seek the window rows, gather the cohort, write the bed | the cohort beds were deleted; `fixture_20k/geno` and `fixture_100k/geno` are plink-made beds of the same kind for 20k/100k cohort people → `golden/bed`; the 12b `.clumped` files are the downstream check |
| 12b | per-gene plink `--clump`, ~2 h | first unchanged on the new bed (identical bed ⇒ identical clumps); later a numpy clump variant | `prsnet/chr<k>/clumps/<gene>.clumped`, all 22 chromosomes |
| 12c | `12_prsnet.py score`, ~2.5 h | `ct_sparse_v3` (§7) | `prsnet/chr19/scores/` chunks; genome-wide `genome/ad_eur/feats.f32` (332 GB, 380,962 × 19,831 × 11) |

Sizes that set the floor (from `window.snplist` / `clump_manifest.tsv`):
5.46 M window SNPs over 22 chromosomes (chr1 545,489 … chr21 48,446), i.e.
666 GB to read from the raw release (487,409 people per row) and a 520 GB
cohort bed (380,962 per row); 2.53 M index-SNP gene links for scoring. The
file system is not the limit: a cold sequential read of a raw bed ran at
3.2 GB/s, four readers at once at ~5.3 GB/s, so the read is 3–4 min of wall.
12a's hours were plink's decode, and the replacement's cost is CPU too:
decode, gather the kept 78 % of people, repack.

**`bed_seek_v0`** (`variants/bed_seek_v0/bed_seek.py`) reproduces
`plink --bfile RAW --keep --extract --a1-allele --make-bed` byte for byte:
variants in raw `.bim` order, people in raw `.fam` order, allele == raw A2 ⇒
`.bim` alleles and homozygous codes swapped, and for an allele that is neither
("Impossible A1 allele assignment", 594 on chr10) plink's 1.9 default applies,
A1 = the minor allele among the kept people (`--unassignable minor`, the
production behaviour; `complement` matches through the strand instead, as
`dosage_pack_v0` does). Blocks of 256 wanted rows: gap-split `np.fromfile`
reads, 256 × 4 table decode, column gather, swap, `x2bit.pack`, write at the
row's offset; `--threads` workers take contiguous row ranges (fork). Smoke
test 31 checks the rules against a brute force and `compare_bed.py`'s verdicts.
Local timing at full width (487,636 raw people, 380,962 kept, one core):
read 0.1 s, decode 0.6 s, gather 1.0 s, pack 0.4 s per 1,000 rows, ~2.5 ms a
row, so ~3.8 core-hours for the genome, about an hour on four cores — v0's
honest estimate before any Bouchet measurement. v1 should gather the 2-bit
fields straight out of the raw bytes and pack in the same pass (the gather is
the cost).

The **bed layout** (third layout of `run_variant.sh`, detected by `geno.bed`):
`geno.bed/.bim/.fam` in the output dir, compared by `compare_bed.py` exactly
(fam fields, bim rows, every genotype byte; differing variants are named with
the number of differing people and whether their `.bim` alleles are swapped).

**First Bouchet runs (2026-10-23, commit 7275089, 4 threads, same salloc).**
`ref_bed` stored `fixture_20k/golden/bed` and `fixture_100k/golden/bed`;
`bed_seek_v0` reproduced both byte for byte: rows `20261023_180041` (20,000
people, wall 48.9 s, cpu 96 s, RSS 1001 MB) and `20261023_180139` (100,000
people, 52.9 s, cpu 116 s, 1048 MB), "all 156485 variants identical". Full
cohort chr19 (`runs/bed_full/chr19`, 156,485 × 380,962 = 14.9 GB): 76 s wall
(read 57.7 s, decode 72.2, gather 60.3, pack 48.9, write 44.5 summed over 4
workers; 78,817 reads, 1.19× over-read, 22.8 GB in) against production 12a's
`ad_eur_pnet_12a_c19` at **11 min 37 s on 4 CPUs / 32 GB** (sacct), ~9×.
Then plink `--clump` with 12b's flags on the new bed for the ten chr19 genes
with the most index SNPs (ERICH4 22,022 lines … KDM4B 154) gave `.clumped`
files identical to production's. chr19's plink log has 0 "Impossible A1"
warnings, so the minor-allele fallback is still untested on real data; chr10
(594 of them) is the test. The 4 h 10 min quoted for 12a was the whole array,
not one chromosome; the per-stage baseline comes from `sacct | grep pnet`.

**chr10 (2026-10-23): the minor-allele fallback on real data.** 342,114 ×
380,962 = 32.6 GB in 186 s wall (4 workers; read 155 s, decode 185, gather
158, pack 132, write 72 summed; 172,221 reads, 1.20× over-read) against 12a's
`ad_eur_pnet_12a_10` at 12 min 21 s. Exactly the 594 SNPs plink had called
impossible were left to the rule, 151 of them swapped to the minor allele,
and plink's clump on the new bed matched production's `.clumped` file for all
nine genes whose windows hold those SNPs (AGAP7 … ZNF518A, 49,728 lines). The
bed stage is therefore reproduced on both chromosomes tested, including the
fallback that `dosage_pack_v0` had treated as a strand flip (§10).

**Per-stage baseline from `sacct | grep pnet`** (ad/eur, all 4 CPUs unless
noted): 12a per chromosome chr1 35:11, chr2 30:58, chr3 28:06, chr4 27:10,
chr5 10:54, chr6 8:59, chr7 13:05, chr8 13:44, chr9 10:31, chr10 12:21,
chr11 11:40, chr12 8:50, chr13 5:14, chr14 9:18, chr15 4:44, chr16 4:45,
chr17 5:40, chr18 3:00, chr19 11:37, chr20 3:14, chr21 1:27, chr22 2:20 —
**~4 h 23 min of job time summed** (the "4 h 10 min" was this array, not a
chromosome). 12b is three kinds of job: `12bt` task arrays (1–2 min each,
2–3 per chromosome), `12bg` giant genes clumped alone on 1 CPU with 14–60 GB
(chr1 SRGAP 26:24, chr8 SCX/JRK 29:50 each, chr5 PCDHB9 23:19, chr7 SPDYE6
18:10 …) and `12bf` finalize (seconds); the giants set 12b's wall. 12c per
chromosome 3–6 min (chr19 5:53, chr2 3:40, chr3 5:15, chr4 2:52). v0's
projection for the genome is ~47 min on 4 cores (0.5 ms a row), ~5.5× 12a
like for like; v1 goes after decode/gather/pack, which are ~65 % of worker time.

**`bed_seek_v1`** (`variants/bed_seek_v1/`): the same rules, reads and files
as v0 with a fused kernel — decode through a 256-entry uint32 table (one
take), the kept people gathered in four strided groups `keep[s::4]` each
shifted into its 2-bit slot and OR-ed straight into the output bytes (no
(k, n) code matrix, no pack pass), and allele swaps applied to the packed
bytes of the swapped rows through a 256-entry table with the padding bits
re-cleared. Locally identical bytes to v0 at full width; kernel 1.5 s per
1,000 rows against v0's 2.0 s (decode 0.5, gather 0.9, pack 0.1). The gather
(381,000 picks a row) is now the floor for plain numpy; beyond it the lever is
cores, which the final submission has (one array task per chromosome).

**v1 on Bouchet (2026-10-23, commit 380b16e, a different node from v0's
rows).** Byte-identical on fixture_20k (row `20261023_184717`, 52.0 s) and
fixture_100k (`20261023_184810`, 64.8 s), and the full chr19 cohort bed from
v1 is byte-identical to v0's 14.9 GB (`compare_bed.py`, all 156,485 ×
380,962). Wall is the same as v0's (76.0 s on chr19; summed read 78 s, decode
79, gather 83, pack 13, write 19): pack fell from 49 s to 13 s but decode and
gather did not move, so **no speed-up is claimed for the v1 kernel**. Locally
a decode-free kernel (gather the raw bytes per slot, shift in place) is only
9 % faster and a per-shift-subgroup scatter 3× slower, so the gather is at
plain numpy's floor and the lever left for the bed stage is cores.
**`--max-gap` sweep** (v1, fixture_20k, same node, 4 threads): gap 2 →
78,817 reads, 1.19× over-read, wall 51.6 s; gap 16 → 3,334 reads, 3.45×,
44.9 s; **gap 64 → 691 reads, 3.82×, 38.1 s**; gap 256 → 640 reads, 39.0 s
(rows `20261023_185150/185236/185316`). Reading 3.8× the bytes in 691 long
reads beats 79,000 short ones by 27 %, so v1's default is 64. Decode (all
487,409 people, ~79 s summed per chromosome whatever the keep list) is now
the largest fixed cost.

**Stage 2 reproduced on chr19 (2026-10-23).** plink `--clump` with 12b's flags
on the v1 cohort bed for all 1,525 chr19 genes with status ok: **1,525
`.clumped` files identical to production's, 0 differ, 0 missing**; 7 min 52 s
wall with four plinks at a time in the 4-CPU salloc (user 8:45, sys 3:59 —
plink's clump is largely I/O and page-cache work on the 14.9 GB bed).
Production's chr19 12b was two `12bt` tasks (1:40, 1:29 at 4 CPUs) plus
ERICH4 alone as `12bg` (9:44 on 1 CPU, 17.7 GB).

**Baseline totals from `sacct` (ad/eur, COMPLETED `pnet` jobs since
2026-09-01, reruns included, so an upper bound):**

| stage | jobs | elapsed (sum) | CPU-hours |
|---|---|---|---|
| 12a cohort beds | 44 | 7.9 h | 31.7 |
| 12bt clump task arrays | 99 | 4.6 h | 18.4 |
| 12bg giant genes, 1 CPU each | 75 | **23.2 h** | 23.2 |
| 12bf finalize | 52 | 0.3 h | 0.3 |
| 12c scoring | 51 | 8.6 h | 34.3 |
| 12d assembly | 2 | 0.4 h | 1.5 |

The giant-gene clumps are half of all 12-stage job time; that is where a
numpy clump pays, not in the task arrays.

**Stage 3 golden:** production's `prsnet/chr19/scores/scores_chunk*.npy` were
removed after 12d (only `scores_manifest.tsv`, `genes_chunk*.txt`,
`gene_counts.tsv`, `samples.txt` remain), so the chr19 scores golden is cut
out of `genome/ad_eur/feats.f32` (row = info.csv order, column = gene_index ×
11 + threshold) for the chr19 genes.

**Stage 3, first full-cohort run (2026-10-23): the scorer is write-bound at
380,962 people.** `ct_sparse_v3` on the v1 chr19 bed with the verified clumps
(79,709 (gene, index SNP) weights, 43,906 unique SNPs, 16,786 columns, 48
chunks of 352 columns at `--chunk-gb 2`): kernel read 15.4 s, decode+impute
66.7 s, scatter 59.1 s — **141 s of compute against production 12c's 5 min
53 s on the same 4 CPUs** — but write 821.6 s, 1016 s in all. Each column
chunk writes 1.4 KB into every row of the 25 GB `(n, m)` memmap, so the file
is rewritten through NFS pages 48 times. Production's 12c never did this: it
writes one contiguous `scores_chunk<k>.npy` per chunk and 12d assembles.
**`ct_sparse_v5`** (`--out-layout chunks`) writes exactly that layout
(`scores_chunk<k>.npy` (n, G_k, T), `genes_chunk<k>.txt`,
`scores_manifest.tsv`, `gene_counts.tsv` with cumulative per-bin counts,
`samples.txt`), one `np.save` per chunk, so 12d consumes it unchanged.
`compare.py` now reads either layout on either side and compares in row
blocks (a 25 GB matrix is read once, sequentially; the old column-by-column
memmap read would touch every page per column); `run_variant.sh` detects the
chunk layout (fourth layout) by `scores_manifest.tsv`. Smoke test 13a: v5's
chunks equal v3's flat output bit for bit through `from_chunks.py`.

**Disk:** the project file system was at 99 % (73 GB free of 4 TB) during
these runs. A genome-wide cohort bed (520 GB) can never exist there; the
submission must build, clump, score and delete one chromosome at a time
(chr1's bed alone is 52 GB).
Full-cohort outputs therefore live in Nick's scratch space
(`/home/naa79/scratch_pi_sz622/naa79/prs-feature-bench/`), symlinked from
`runs/` where a path is needed; fixture-scale runs stay in `runs/` (the
harness deletes their bulk on PASS).

**Stage 3 reproduced on chr19 (2026-10-23, commit cc8065f).** Golden:
`feats_extract.py` cut chr19's 1,526 genes × 11 thresholds out of
`feats.f32` (25.6 GB, 1088 s — the memmap row reads ran at 0.33 GB/s, a
tenth of `dd`; a `fromfile` reader would do it in ~2 min). Candidate:
`ct_sparse_v5` on the v1 bed with the verified clumps, chunk layout to
scratch: kernel read 16.0 s, decode+impute 67.3 s, scatter 62.3 s, write
159.6 s (48 × 536 MB `np.save`, ~160 MB/s), 340 s in all, 5 min 40 s wall
with the weights step, against 12c's `ad_eur_pnet_12c_19` at 5 min 53 s on
the same 4 CPUs. **`compare.py`: PASS, 16,786 cols, 0 bad, max|diff|
0.000e+00** — bit-identical to production's features, read in row blocks in
3 min 13 s. `scores_manifest.tsv`: 1,526 genes both; 59,871 index-SNP reads
against production's 60,084 (different gene chunking); `gene_counts.tsv`
ERICH4 identical (22019 index; 13 28 99 525 1953 3448 6280 8873 11256 13542
22019). On the fixtures v5 PASSes fixture_20k bit for bit (row
`20261023_201704`, 48.5 s at 4 threads, RSS 923 MB; v3's rows are at 2
threads so no wall comparison) and ran fixture_100k (3:22, no `golden/ct`
there).

**Where the chr19 chain stands (4 CPUs, same salloc vs production's jobs):**

| stage | production | bench | note |
|---|---|---|---|
| 12a cohort bed | 11 min 37 s | **76 s** (`bed_seek_v1`) | byte-identical |
| 12b clump, 1,525 genes | ~13 min (2 × 12bt at 4 CPUs + ERICH4 9:44 alone) | 7 min 52 s (same plink, 4 at a time) | 1,525/1,525 `.clumped` identical |
| 12c scores | 5 min 53 s | 5 min 40 s (`ct_sparse_v5`) | bit-identical; compute 146 s, write 160 s |
| chr19 total | ~31 min | ~15 min | |

The bed stage is 9× and done. The clump stage runs production's own plink,
so its cost per gene is unchanged; what the submission removes is the
scheduling — the giant genes (`12bg`, 23 h summed on 1 CPU each, 75 jobs
with reruns) run inside the chromosome's task like every other gene, largest
first, `$SLURM_CPUS_PER_TASK` plinks at a time. A numpy clump was tried and
closed (`clump_np_v0`, below): plink's r² is not the genotype correlation,
and reproducing its EM r² exactly costs more GEMM than plink's own popcounts.
The scoring stage's compute is 2.4× faster than 12c's but the stage is
write-bound: 25.6 GB per chromosome at ~160 MB/s, the same bytes 12c wrote
and 12d writes again when it assembles `feats.f32` (332 GB, 0.4 h). A
scorer that writes `feats.f32` once, in people-row blocks across all
chromosomes, would save one of the two 332 GB writes.

**12b without plink: `clump_np_v0`.** The facts that make it possible: only
30 giant genes exist genome-wide, but their windows hold 300k–469k SNPs each
(SRGAP2B 468,814 of chr1's 545,489; ERICH4's plink log: "22019 clumps formed
from 113666 top variants"), and every gene on a chromosome clumps a subset of
the same window SNPs under the same rule, so 12b recomputes the chromosome's
LD once per gene. `ld_pairs.py` computes it once: every pair within 250 kb
with r² ≥ 0.5, as a sparse graph. r² is plink's --clump r², the squared
Pearson correlation of the allele counts over the people with both calls
present (pairwise deletion); with missing coded 0, Sxy = x·y comes from a
float32 GEMM that is exact because every sum is an integer below 2²⁴
(4 × 380,962), the missing corrections (Σx over y-missing etc.) are sparse
products, and r² is formed in float64 — deterministic, no rounding at the
0.5 boundary. Blocks of 512 SNPs in position order, each against itself and
the previous blocks still within reach. `clump_genes.py` then runs plink's
greedy per gene over the graph restricted to the gene's SNPs (P ascending,
ties by position, a SNP without P skipped) and writes `<gene>.clumped` (SNP in
column 3, TOTAL) plus `clump_manifest.tsv`, so `make_ct_weights.py` and the
scorer run unchanged, and `clumps.tsv` (gene, n_index, index SNPs), the
bench's **clumps layout** (fifth layout; `compare_clumps.py` compares index
SETS exactly; `ref_clump` = plink per gene on the fixture → `golden/clumps`;
`lib/clumped_to_tsv.py` turns production's `.clumped` directories into the
same file). Smoke test 33: pairs and r² equal a brute force, index sets equal
a hand-written greedy.

**clump_np_v0: FAIL, closed (2026-10-23).** fixture_20k: `ref_clump` (plink
per gene, 4 at a time) 0:32.73 → `golden/clumps`; `clump_np_v0` 2:47.09,
**FAIL: 1526 genes, 1316 identical, 210 differ** (79,715 index SNPs vs
79,639; LD pass 161 s of which GEMM 125 s). Full chr19 on the v1 bed: LD pass
51 min (read 701 s, GEMM 2363 s, 4 cores), greedy 9 s, **637 identical, 889
differ** against production's 1,525 `.clumped` (80,542 vs 79,709 index
SNPs). The ordering is right (ties by P then .bim index is exactly plink's
`double_cmp_deref_tiebreak`, its 2021 determinism fix); the r² is not.
plink 1.9's `clump_reports` (plink_ld.c) does **not** use the allele-count
correlation: for each (index, candidate) pair it builds the 3 × 3 genotype
table over founders with both calls present (`genovec_3freq` against the
index SNP's three genotype masks), turns it into the four known haplotype
counts plus the double-heterozygote count, and runs `em_phase_hethet` — the
maximum-likelihood haplotype frequencies from the cubic (`cubic_real_roots`,
boundary and multiple-root rules, `calc_lnlike` to pick the root) — then
r² = D² / (p₁ q₁ p₂ q₂) with D = f₁₁ − p₁ q₁, and clumps when r² is
**strictly greater** than `--clump-r2`. The genotype r² and the EM r² agree
on most pairs, which is why 1,316 genes matched, and disagree around the
threshold; with 380,962 people the disagreements multiplied (889 genes).
Reproducing it exactly is possible — the 3 × 3 table has four degrees of
freedom once the per-SNP marginals are known, so four indicator GEMMs (or
two float64 GEMMs with packed counts) per block pair, then the EM in float64
per candidate pair — but it would cost 2–4× the one GEMM that already took
51 min on chr19, where production's own plink loop clumps all 1,525 genes
(ERICH4 included) in **7 min 52 s on the same 4 CPUs**. plink's per-index
popcounts over 2-bit genotypes, skipping SNPs already clumped, are simply
the cheaper algorithm here; the chromosome-wide LD graph saves nothing. The
numpy clump is closed; 12b stays plink, scheduled better. The variant,
`ref_clump`, `compare_clumps.py`, the clumps layout and `clumped_to_tsv.py`
stay as the exact-comparison machinery for any future 12b change, and smoke
test 33 still checks `clump_np_v0` against its own definition.

**The single submission: `port/prsnet_chain.sbatch`** (draft, not yet run
under sbatch). One array, one task per chromosome, 8 CPUs / 64 GB / 8 h (the
limit is chr1's: 545,489 window SNPs and 8 of the 30 giant genes): 12a
`bed_seek_v1` → 12b production's plink `--clump` per gene, all genes
including the giants, largest first, `$SLURM_CPUS_PER_TASK` at a time with
`--memory` split evenly, `clump_manifest.tsv` by 12b's own awk → 12c
`ct_sparse_v5` in the chunk layout → the bed deleted (`CHAIN_KEEP_BED=1`
keeps it). Everything lands in scratch
(`CHAIN_OUT`, default `…/scratch_pi_sz622/naa79/prs-feature-bench/prsnet_chain/<disease>_<cohort>/chr<k>/`),
production's `genes/`, `gene_manifest.tsv` and `window.snplist` are reused
when they exist (made with `12_prsnet.py snps` otherwise), every stage
resumes if its output exists (`CHAIN_FORCE=1` redoes), and each stage prints
a `== <stage> <s> s` lap line. `STEP=12d` runs production's
`12_prsnet.py assemble` unchanged over the run dir (one small job after the
22 tasks; feats.f32 lands in scratch too). `verify` compares one
chromosome's clumps with production's `.clumped` (exact index sets) and its
scores with a golden dir (`CHAIN_GOLDEN`).

**chr19 through the chain (2026-10-23, 4 CPUs, salloc).** 12a 67 s (bed
14.9 GB, 1.09 GB/s in, 0 SNPs left to the minor-allele rule on chr19), 12b
475 s (1,525 genes, 4 plink at a time, ERICH4 first), 12c weights 1 s +
`ct_sparse_v5` 361 s (96 chunks of 176 columns, write 170 s of it), bed
deleted: **905 s = 15 min 5 s** against production's ~31 min for the same
chromosome. `verify`: scores **PASS, 16,786 cols, max|diff| 0** against
`runs/ct_full/golden_chr19` (production feats.f32).

**Genome-wide (2026-10-23/24, jobs 30499668–30499689).** Submitted with
`port/submit_chain.sh ad eur`: 22 jobs sized per chromosome from its gene
manifest (4 or 8 CPUs, 16 or 32 GB, 20 min to 1 h 45 min). All 22 COMPLETED;
`gene_counts.tsv` rows sum to **19,831** (every graph node); 311 GB of scores
in scratch, every bed deleted by its own job. **Submit → last task done
≈ 55 min**, bounded by chr1 (53:51); the 22 elapsed times sum to ≈ 6.4 h
of job time against production's ≈ 44 h across its 12a/12bt/12bg/12c arrays
(sacct, reruns included).

| chr | elapsed | 12a bed | 12b clump | 12c score | note |
|---|---|---|---|---|---|
| 1 | 53:51 | 164 s | 2153 s | 907 s | 8 giant genes, 545,489 window SNPs; 8 CPUs |
| 8 | 41:32 | 329 s | 1827 s | 332 s | SRGAP2B-class giant 422,869 SNPs |
| 5 | 30:28 | 152 s | 1374 s | 299 s | giant 388,217 |
| 7 / 9 / 11 / 10 | 25–28 min | 200–300 s | 900–1150 s | 230–320 s | one giant each |
| 14 / 17 | 21–22 min | | | | giants 245,965 / 192,278 (17 at 4 CPUs) |
| 19 | 15:05 (salloc) | 67 s | 475 s | 361 s | ERICH4 |
| 2, 4, 12, 13, 15, 16, 18, 20, 21, 22 | 4–9 min | 46–300 s | 36–130 s | 62–285 s | no giant |

The clump stage of a giant-bearing chromosome is the only item above ten
minutes left; it is production's own plink, one process per gene, so the
remaining lever is CPUs per job (the giants of one chromosome run
concurrently). The submitter's raw estimates were within ~15 % for the big
chromosomes and high for the small ones; its 1.5× + 10 min margin made the
requests about twice the actual, so the margin is now 1.3× + 5 min. Each
plink is now capped at `--memory 2500` (CHAIN_PLINK_MB) and memory is
requested by CPU count (16 GB at 4, 24 GB at 8): plink pre-allocates its
whole `--memory` share, so MaxRSS equalled the request on every job and said
nothing about need.

**12d and the verdict (2026-10-24, job 30508884, 4 CPUs / 32 GB).**
Production's `12_prsnet.py assemble --export memmap --sample-block 5000`
over the scratch run dir: **1737 s = 28.9 min**, MaxRSS 32 GB (the memmap's
pages count; the request was 32 GB). `genes_not_in_order.txt` and
`genes_without_features.txt` empty; `info.csv` identical to production's.
A dependent `cmp` job (8 min) over the two 332,413,726,568-byte files:
**`FEATS_IDENTICAL`** — the chain's genome-wide `feats.f32` is byte for byte
production's. `feats.meta.json` differs only in the two absolute paths it
records. So the complete PRS-Net feature chain (12a → 12b → 12c → 12d)
reproduces production exactly in **≈ 55 min of chain + 29 min of 12d ≈ 1 h
25 min submit → feats.f32**, against ~44 h of job time (days of wall) in
production's arrays. 12b is production's plink, 12d is production's
assembler; only 12a (`bed_seek_v1`) and 12c (`ct_sparse_v5`) are new code,
and both are checked against production's own artefacts. Left on the
table: 12d's 332 GB write (0.5 h) could be folded into 12c by writing each
chromosome's columns straight into a preallocated `feats.f32`, and the
clump stage of the giant chromosomes is bounded by plink on one gene.
