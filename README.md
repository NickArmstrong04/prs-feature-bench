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
2. **Decode dosages from the packed bed bytes directly** (2 bits/genotype →
   int8) instead of float32 through bed-reader; impute from per-SNP allele
   counts computed in the same pass.
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

## 8. Relationship to the production tree

Production: `$ROOT/scripts/slurm/12_prsnet.{sh,py}` and `submit_12b_split.sh`
(design in project docs `claude/prsnet.md`, `claude/prsnet_features.md`). The
bench never writes there. A variant that wins here is ported back as a new
step (`12e`, say), run on one chromosome, and gated by `12_prsnet.sh check` or
its eQTL equivalent before it touches the multi-disease plan.
