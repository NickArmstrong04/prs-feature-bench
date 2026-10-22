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
├── compare.py              fidelity check: golden vs candidate, standard layout
├── results.tsv             append-only log (created by the first run)
├── fixture/                the data (not in git; ~200 MB)
├── golden/<name>/          reference outputs (not in git)
├── runs/<variant>_<stamp>/ per-run output + run.log + compare.txt
├── lib/
│   ├── ref_12_prsnet.py    pinned copy of the production scorer (from make_fixture)
│   └── from_chunks.py      12c chunk layout -> standard layout
├── eqtl/
│   └── weights_to_plink2.py  long-format weights -> plink2 wide score file
├── tests/smoke_test.py     synthetic end-to-end test of the Python pieces
└── variants/
    ├── ref_ct/             reference: production 12b + 12c   -> golden/ct
    ├── ref_eqtl_plink2/    oracle: plink2 multi-column --score -> golden/eqtl_<set>
    └── eqtl_sparse_v0/     first fast eQTL implementation (bed-reader + scipy)
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
`BENCH_FIXTURE=…` (not implemented yet; add the variable when needed).

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
   **Kernel side, pending: `ct_sparse_v4`** = ct_sparse_v3's run.sh (12b
   clumping unchanged) with eqtl_sparse_v3's kernel in `--mode ct`, plus genes
   position-sorted in ct mode too (the manifest order is not positional and
   the GEMM kernel's blocks are runs of whole genes). Smoke test 22 PASSES
   against the ct brute force with the same (gene, channel, n_snps) set as
   v0; column order differs, which the contract allows. Expect the kernel
   line to fall from ~11�13 s toward 7 s; the wall stays dominated by 12b's
   ~1,500 plink processes, which is the A1 item.

## 8. Relationship to the production tree

Production: `$ROOT/scripts/slurm/12_prsnet.{sh,py}` and `submit_12b_split.sh`
(design in project docs `claude/prsnet.md`, `claude/prsnet_features.md`). The
bench never writes there. A variant that wins here is ported back as a new
step (`12e`, say), run on one chromosome, and gated by `12_prsnet.sh check` or
its eQTL equivalent before it touches the multi-disease plan.
