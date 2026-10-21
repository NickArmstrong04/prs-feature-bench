#!/bin/bash
# install_bench.sh -- writes the files of prs-feature-bench that were not
# copied over (plus the updated make_fixture.sh) into the directory this
# script sits in. Paste into VS Code, save next to run_variant.sh, then:
#     bash install_bench.sh
# Safe to re-run; it overwrites only the files listed below.
set -euo pipefail
cd "$(dirname "$0")"
[[ -f run_variant.sh ]] || { echo "run this from the prs-feature-bench directory (run_variant.sh not found here)"; exit 1; }

cat > 'README.md' <<'EOF_README_MD'
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
3. **Column chunking**: with ~20k genes × 14 cell types = 280k columns, `S` at
   381k people is 850 GB in float64 — chunk the columns (by gene block) and
   write each chunk's float32 to the memmap as it finishes; the SNP reads are
   then repeated per column chunk, so size the chunk to the memory budget.
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
EOF_README_MD
echo "wrote README.md ($(wc -l < 'README.md') lines)"

cat > 'CLAUDE.md' <<'EOF_CLAUDE_MD'
# prs-feature-bench — rules for this session

This repo is a test harness for versions of the PRS-Net gene-feature scripts.
Read README.md first (sections 2 and 6 are the contract and the method).

## Never
- Edit `variants/ref_ct/`, `variants/ref_eqtl_plink2/`, `lib/ref_12_prsnet.py`,
  anything under `fixture/` or `golden/`. If a reference looks wrong, say so
  and stop; the fix belongs in the production scripts on the cluster.
- Loosen `--rtol`/`--atol` to make a variant pass. A numeric deviation gets a
  new, named golden and a note here, not a wider tolerance.
- Run anything on the login node that takes more than a few seconds of CPU.
  Use the `salloc` the user has open; if none is open, say which request to
  make (`salloc -c 4 --mem=16G -t 2:00:00` is the pattern that starts fast
  on this account; 8 CPUs is refused by the interactive QOS).
- Delete `runs/` entries or `results.tsv` rows. They are the experiment log.

## Always
- Make a change as a new `variants/<name>/` (copy the closest variant), one
  idea per variant, and run it through `bash run_variant.sh <name> --tag "…"`
  — never by calling a variant's scripts directly, because only the harness
  records timing, commit and verdict.
- Commit after each PASSing variant with a message that says what changed and
  the wall time vs the previous best at the same `--threads`.
- Before claiming a speed-up, show two rows of `results.tsv` at equal threads.
- When a run FAILs, read `runs/<v>_<stamp>/compare.txt` and `run.log` before
  theorising; the worst columns are listed first.
- Keep the Python plain numpy/scipy/bed-reader unless a variant's whole point
  is a new dependency; then install it into the bench venv and say so.

## Environment on Bouchet
- `BENCH_PY` = the production PRS-Net venv python (`$PRSNET_VENV/bin/python` in
  `$ROOT/scripts/slurm/config.sh`); it has numpy, scipy, bed-reader.
- plink 1.9 / plink2 are Lmod modules in the same family — loading one unloads
  the other (`PLINK/1.9b_7.11-x86_64`, `PLINK/2_avx2_20250707`). The reference
  scripts load what they need; a variant needing both must resolve absolute
  paths right after each load.
- `set -o pipefail` + `sort | head` kills jobs silently (SIGPIPE); use
  `awk 'NR<=n'`.

## Where things are recorded
- `results.tsv` — every run.
- `README.md` §7 — the optimization list; strike items as they are tried,
  with the results row that settled them.
- Project doc `claude/feature_bench.md` (claude.ai project "Benchmarking")
  mirrors README.md; update it when the README changes materially.
EOF_CLAUDE_MD
echo "wrote CLAUDE.md ($(wc -l < 'CLAUDE.md') lines)"

cat > '.gitignore' <<'EOF__GITIGNORE'
fixture/
golden/
runs/
results.tsv
lib/ref_12_prsnet.py
__pycache__/
EOF__GITIGNORE
echo "wrote .gitignore ($(wc -l < '.gitignore') lines)"

cat > 'make_fixture.sh' <<'EOF_MAKE_FIXTURE_SH'
#!/bin/bash
# =============================================================================
#  make_fixture.sh -- build the fixed test dataset for prs-feature-bench
#
#  Runs ONCE, on Bouchet (inside a small salloc: 2 CPU / 8 GB is plenty).
#  Takes one chromosome of one finished PRS-Net feature build (default
#  ad/eur chr19, which passed `12_prsnet.sh check`), subsamples the cohort
#  to N people (case/control stratified, seeded) and copies every input the
#  feature scripts need into $BENCH/fixture/. Nothing in the production
#  tree is modified. The fixture is ~150-300 MB and can be scp'd anywhere.
#
#  USAGE
#      DISEASE=ad COHORT=eur CHR=19 N=2000 bash make_fixture.sh
#
#  Reads the production config (ROOT/scripts/slurm/config.sh) only to find
#  paths; the fixture carries its own params.json afterwards, so variants
#  never source config.sh.
#
#  Output layout (the harness contract -- see README.md "Fixture"):
#      fixture/geno.{bed,bim,fam}      N samples x window SNPs, A1 = effect allele
#      fixture/gwas.QC.txt             QC'd sumstats, this chromosome
#      fixture/SNP.pvalue              plink2 --q-score-range input (SNP P)
#      fixture/pheno.tab               FID IID PHENO for the N samples
#      fixture/gene_windows.tsv        windows on this chromosome
#      fixture/gene_manifest.tsv       12a's manifest (gene, ..., n_snps, big)
#      fixture/genes/<gene>.{snplist,assoc}
#      fixture/params.json             thresholds, r2, kb, chr, N, provenance
#      lib/ref_12_prsnet.py            pinned copy of the production scorer
# =============================================================================
set -euo pipefail

DISEASE=${DISEASE:-ad}; COHORT=${COHORT:-eur}; CHR=${CHR:-19}; N=${N:-2000}; SEED=${SEED:-7}
ROOT=${ROOT:-/home/naa79/project_pi_sz622/naa79/prs-benchmark}
BENCH=${BENCH:-$(cd "$(dirname "$0")" && pwd)}
FIX=$BENCH/fixture

export DISEASE COHORT CHR
source "$ROOT/scripts/slurm/config.sh"

for v in PNET_CHR GQC PHENO PRSNET_WINDOWS PRSNET_THRESHOLDS PRSNET_CLUMP_R2 PRSNET_CLUMP_KB PLINK_MODULE; do
    [[ -n "${!v:-}" ]] || { echo "config.sh does not define $v"; exit 1; }
done
for f in "$GQC/gwas.QC.txt" "$GQC/SNP.pvalue" "$PHENO" "$PRSNET_WINDOWS" \
         "$PNET_CHR/gene_manifest.tsv" "$PNET_CHR/window.snplist" "$ROOT/scripts/slurm/12_prsnet.py"; do
    [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
done
[[ -d "$PNET_CHR/genes" ]] || { echo "MISSING: $PNET_CHR/genes (run 12a for chr$CHR)"; exit 1; }

# Genotype source. Preferred: 12a's cohort bed (window SNPs, A1 = effect
# allele). It is a regenerable intermediate and may have been deleted; then
# fall back to 12a's own command against the raw release, restricted to the
# fixture's people -- same --extract, same --a1-allele, so the same variants
# and orientation come out. The raw chromosome is streamed once (minutes).
BED=$PNET_CHR/target_chr${CHR}.PH
if [[ -f "$BED.bed" && -f "$BED.bim" && -f "$BED.fam" ]]; then
    MODE=cohort
else
    MODE=raw
    for v in GENO_PREFIX MIS; do [[ -n "${!v:-}" ]] || { echo "config.sh does not define $v (needed for the raw fallback)"; exit 1; }; done
    for f in "$GENO_PREFIX.bed" "$GENO_PREFIX.bim" "$GENO_PREFIX.fam" "$MIS/target_data.a1"; do
        [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
    done
fi

module load "$PLINK_MODULE"
command -v plink >/dev/null || { echo "plink not on PATH after module load $PLINK_MODULE"; exit 1; }

if [[ $MODE == cohort ]]; then
    echo "source:   $DISEASE/$COHORT chr$CHR cohort bed  ($(wc -l < "$BED.fam") samples, $(wc -l < "$BED.bim") window SNPs)"
    FAM=$BED.fam
else
    echo "source:   $DISEASE/$COHORT chr$CHR -- cohort bed absent, building from the RAW release"
    echo "          $GENO_PREFIX  (--keep fixture people, --extract $(wc -l < "$PNET_CHR/window.snplist") window SNPs, --a1-allele)"
    FAM=$GENO_PREFIX.fam
fi
echo "fixture:  $FIX  (N=$N, seed $SEED)"
mkdir -p "$FIX/genes" "$BENCH/lib"

# --- 1. stratified sample of N people present in both the genotypes and the phenotype
#     pheno.tab: FID IID PHENO (1=control, 2=case), no header
export LC_ALL=C
join -j 2 <(awk '{print $1, $2}' "$FAM" | sort -k2,2) \
          <(awk '{print $1, $2, $3}' "$PHENO" | sort -k2,2) \
    | awk '{print $2, $1, $4}' > "$FIX/.pool.txt"        # FID IID PHENO
n_case=$(awk '$3==2' "$FIX/.pool.txt" | wc -l); n_ctrl=$(awk '$3==1' "$FIX/.pool.txt" | wc -l)
k_case=$(( N * n_case / (n_case + n_ctrl) )); k_ctrl=$(( N - k_case ))
{ awk '$3==2' "$FIX/.pool.txt" | shuf --random-source=<(yes "$SEED") | head -n "$k_case"
  awk '$3==1' "$FIX/.pool.txt" | shuf --random-source=<(yes "$SEED") | head -n "$k_ctrl"; } \
    | sort -k2,2 > "$FIX/pheno.tab"
awk '{print $1, $2}' "$FIX/pheno.tab" > "$FIX/.keep.txt"
rm -f "$FIX/.pool.txt"
echo "sampled:  $(wc -l < "$FIX/pheno.tab") people ($k_case cases / $k_ctrl controls; pool $n_case / $n_ctrl)"

# --- 2. the genotype fixture: same variants, same A1, N people
if [[ $MODE == cohort ]]; then
    plink --bfile "$BED" --keep "$FIX/.keep.txt" --make-bed --out "$FIX/geno" \
          --memory 4000 --threads 2 > "$FIX/.plink.log" 2>&1 || { cat "$FIX/.plink.log"; exit 1; }
    rm -f "$FIX/.keep.txt" "$FIX/geno.nosex"
    cmp -s <(awk '{print $2, $5}' "$BED.bim") <(awk '{print $2, $5}' "$FIX/geno.bim") \
        || { echo "ERROR: fixture .bim A1 differs from the source .bim (plink flipped something)"; exit 1; }
else
    # 12a's command (12_prsnet.sh step_12a), keep list = the fixture's people
    plink --bfile "$GENO_PREFIX" --keep "$FIX/.keep.txt" --extract "$PNET_CHR/window.snplist" \
          --a1-allele "$MIS/target_data.a1" --make-bed --out "$FIX/geno" \
          --memory 4000 --threads 2 > "$FIX/.plink.log" 2>&1 || { cat "$FIX/.plink.log"; exit 1; }
    rm -f "$FIX/.keep.txt" "$FIX/geno.nosex"
    # orientation check, as 12a does it: .bim A1 (col 5) must equal gwas A1 (col 4) for every variant
    join -1 2 -2 3 <(sort -k2,2 "$FIX/geno.bim") <(tail -n +2 "$GQC/gwas.QC.txt" | sort -k3,3) \
      | awk '{if($5==$9) m++; else x++} END{printf "A1 vs gwas effect allele: %d match, %d mismatch\n", m+0, x+0; exit (x>0)}' \
      || { echo "ERROR: fixture A1 is not the GWAS effect allele for some variants"; exit 1; }
    n_win=$(wc -l < "$PNET_CHR/window.snplist"); n_got=$(wc -l < "$FIX/geno.bim")
    (( n_got == n_win )) || echo "NOTE: $n_got of $n_win window SNPs written (12a's own bed had the same property if 03 dropped variants)"
fi
echo "geno:     $(wc -l < "$FIX/geno.bim") variants x $(wc -l < "$FIX/geno.fam") samples, $(du -h "$FIX/geno.bed" | cut -f1)"

# --- 3. sumstats, p-values, windows, manifest, per-gene files
cp "$GQC/gwas.QC.txt" "$FIX/gwas.QC.txt"
cp "$GQC/SNP.pvalue"  "$FIX/SNP.pvalue"
head -n 1 "$PRSNET_WINDOWS" > "$FIX/gene_windows.tsv"
awk -v c="$CHR" 'NR>1 && ($1==c || $1=="chr"c || $2==c || $2=="chr"c)' "$PRSNET_WINDOWS" >> "$FIX/gene_windows.tsv"
cp "$PNET_CHR/gene_manifest.tsv" "$FIX/gene_manifest.tsv"
cp "$PNET_CHR"/genes/*.snplist "$PNET_CHR"/genes/*.assoc "$FIX/genes/" 2>/dev/null
echo "genes:    $(ls "$FIX/genes"/*.snplist | wc -l) snplists, $(awk 'NR>1' "$FIX/gene_windows.tsv" | wc -l) window rows"

# --- 4. pinned reference scorer + params
cp "$ROOT/scripts/slurm/12_prsnet.py" "$BENCH/lib/ref_12_prsnet.py"
cat > "$FIX/params.json" <<EOF
{
  "disease": "$DISEASE", "cohort": "$COHORT", "chr": $CHR,
  "n_samples": $(wc -l < "$FIX/geno.fam"), "n_variants": $(wc -l < "$FIX/geno.bim"),
  "seed": $SEED,
  "thresholds": "$PRSNET_THRESHOLDS",
  "clump_r2": $PRSNET_CLUMP_R2, "clump_kb": $PRSNET_CLUMP_KB,
  "giant_window_mb": ${PRSNET_GIANT_MB:-20},
  "source_mode": "$MODE", "source_bed": "$([[ $MODE == cohort ]] && echo "$BED" || echo "$GENO_PREFIX")",
  "source_scorer": "$ROOT/scripts/slurm/12_prsnet.py",
  "built": "$(date -Is)"
}
EOF
cat "$FIX/params.json"
echo
echo "fixture size: $(du -sh "$FIX" | cut -f1)"
echo "Next:  bash run_variant.sh ref_ct --as-golden ct     (first golden, ~minutes)"
EOF_MAKE_FIXTURE_SH
echo "wrote make_fixture.sh ($(wc -l < 'make_fixture.sh') lines)"

mkdir -p "variants/ref_ct"
cat > 'variants/ref_ct/run.sh' <<'EOF_VARIANTS_REF_CT_RUN_SH'
#!/bin/bash
# =============================================================================
#  variants/ref_ct/run.sh -- the REFERENCE C+T variant
#
#  Exactly the production path on the fixture: the authors' per-gene
#  `plink --clump` (12b's clump_one, same flags) followed by the pinned
#  production scorer (lib/ref_12_prsnet.py score = 12c), then converted to
#  the harness layout. This is what golden/ct is made from. Do not optimise
#  this file -- copy it to variants/<new>/ and change the copy.
#
#  Called by run_variant.sh as:  run.sh <fixture_dir> <out_dir>
# =============================================================================
set -euo pipefail
FIX=$1; OUT=$2
W=$OUT/work; mkdir -p "$W/clumps" "$W/chunks"
PY=${BENCH_PY:-python3}; LIB=${BENCH_LIB:?}; NT=${BENCH_THREADS:-4}

# fixture params
eval "$("$PY" - "$FIX/params.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f'CHR={d["chr"]}; R2={d["clump_r2"]}; KB={d["clump_kb"]}; THR="{d["thresholds"]}"; GIANT={d["giant_window_mb"]}')
EOF
)"

# plink 1.9: module on Bouchet, PATH elsewhere
if ! command -v plink >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load PLINK/1.9b_7.11-x86_64 2>/dev/null || true
fi
command -v plink >/dev/null || { echo "plink (1.9) not found"; exit 1; }

# --- 12b: the authors' per-gene clump, same flags as production clump_one
clump_one() {
    local g=$1
    plink --bfile "$FIX/geno" --extract "$FIX/genes/$g.snplist" --chr "$CHR" \
          --clump-p1 1 --clump-r2 "$R2" --clump-kb "$KB" \
          --clump "$FIX/genes/$g.assoc" "$FIX/gwas.QC.txt" --clump-index-first \
          --clump-snp-field SNP --clump-field P \
          --out "$W/clumps/$g" --memory 1500 --threads 1 > /dev/null 2>&1 || true
}
export -f clump_one; export FIX W CHR R2 KB
awk 'NR>1 && $4>0 {print $1}' "$FIX/gene_manifest.tsv" > "$W/genes.txt"
echo "clumping $(wc -l < "$W/genes.txt") genes, $NT at a time  ($(date +%T))"
xargs -a "$W/genes.txt" -P "$NT" -I{} bash -c 'clump_one "$1"' _ {}

# clump_manifest.tsv: identical awk to 12_prsnet.sh write_12b_results
awk -v D="$W/clumps" 'BEGIN{OFS="\t"}
     NR==1 {print $0, "n_index", "status"; next}
     { n=0; st="nosnp"
       if ($4>0) { f=D"/"$1".clumped"; st="failed"
           if ((getline l < f) > 0) { st="ok"; while ((getline l < f) > 0) { split(l,a," "); if (length(a)>=3) n++ } }
           close(f) }
       print $0, n, st }' "$FIX/gene_manifest.tsv" > "$W/clump_manifest.tsv"
nfail=$(awk 'NR>1 && $7=="failed"' "$W/clump_manifest.tsv" | wc -l)
echo "clumped: $(awk 'NR>1 && $7=="ok"' "$W/clump_manifest.tsv" | wc -l) ok, $nfail failed  ($(date +%T))"
(( nfail == 0 )) || { awk 'NR>1 && $7=="failed" {print "  failed:", $1}' "$W/clump_manifest.tsv"; exit 1; }

# --- 12c: the pinned production scorer
"$PY" "$LIB/ref_12_prsnet.py" score --bed "$FIX/geno" --gwas "$FIX/gwas.QC.txt" \
    --clumps "$W/clumps" --manifest "$W/clump_manifest.tsv" \
    --out-dir "$W/chunks" --thresholds "$THR" \
    --chunk-genes 100 --block-snps 2000 --giant-window-mb "$GIANT" --threads "$NT"

# --- harness layout
"$PY" "$LIB/from_chunks.py" "$W/chunks" "$OUT"
EOF_VARIANTS_REF_CT_RUN_SH
echo "wrote variants/ref_ct/run.sh ($(wc -l < 'variants/ref_ct/run.sh') lines)"

mkdir -p "variants/ref_eqtl_plink2"
cat > 'variants/ref_eqtl_plink2/run.sh' <<'EOF_VARIANTS_REF_EQTL_PLINK2_RUN_SH'
#!/bin/bash
# =============================================================================
#  variants/ref_eqtl_plink2/run.sh -- the ORACLE for eQTL-weighted features
#
#  Feature (i, gene g, channel c) = sum_j  w_{j,g,c} * dosage_i(a1_j)
#  i.e. predicted expression of g in cell type c for person i, with missing
#  calls mean-imputed (plink2's default; the same rule 12c uses).
#
#  This variant asks plink2 to compute every column at once from one wide
#  score file (multi-column --score, scoresums). It is slow-ish and exact;
#  it is what golden/eqtl_<set> is made from. Fast variants are compared
#  against it. Do not optimise this file.
#
#  Needs fixture/weights/$WEIGHTS.tsv (contract in eqtl/weights_to_plink2.py).
#  Called by run_variant.sh as:  WEIGHTS=<set> run.sh <fixture_dir> <out_dir>
# =============================================================================
set -euo pipefail
FIX=$1; OUT=$2
W=$OUT/work; mkdir -p "$W"
PY=${BENCH_PY:-python3}; NT=${BENCH_THREADS:-4}
WEIGHTS=${WEIGHTS:-onek1k}
WFILE=$FIX/weights/$WEIGHTS.tsv
[[ -f "$WFILE" ]] || { echo "no weight set: $WFILE  (see eqtl/README section of README.md)"; exit 1; }
BENCH=$(cd "$(dirname "$0")/../.." && pwd)

if ! command -v plink2 >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load PLINK/2_avx2_20250707 2>/dev/null || true
fi
command -v plink2 >/dev/null || { echo "plink2 not found"; exit 1; }

"$PY" "$BENCH/eqtl/weights_to_plink2.py" "$WFILE" "$FIX/geno.bim" "$W/wide.tsv" --min-snps 1
NCOL=$(awk 'NR>1' "$W/wide.tsv.columns.tsv" | wc -l)
(( NCOL > 0 )) || { echo "no usable columns"; exit 1; }

# plink2 --score: col 1 = SNP, col 2 = A1, score columns 3..(NCOL+2);
# header-read names the scores; scoresums instead of averages.
# plink2's cols= syntax and the exact SUM column names changed between
# releases -- the first run on Bouchet should confirm the header printed
# below (the production `check` had the same kind of surprise, C.6 (3)).
plink2 --bfile "$FIX/geno" \
       --score "$W/wide.tsv" 1 2 header-read cols=+scoresums,-scoreavgs \
       --score-col-nums 3-$((NCOL + 2)) \
       --out "$W/oracle" --threads "$NT" --memory 8000 > "$W/plink2.log" 2>&1 \
    || { tail -n 20 "$W/plink2.log"; exit 1; }
head -c 400 "$W/oracle.sscore" | head -n 1 | cut -c1-200; echo

# .sscore -> harness layout (columns in the wide file's order)
"$PY" - "$W/oracle.sscore" "$W/wide.tsv.columns.tsv" "$OUT" <<'EOF'
import sys, numpy as np
ss, colf, out = sys.argv[1:4]
with open(ss) as fh:
    hdr = fh.readline().lstrip("#").rstrip("\n").split("\t")
    rows = [l.rstrip("\n").split("\t") for l in fh if l.strip()]
# score columns = everything ending in _SUM, in file order
sc = [i for i, h in enumerate(hdr) if h.endswith("_SUM")]
names = [hdr[i][:-4] for i in sc]
cols = [l.rstrip("\n").split("\t") for l in open(colf)][1:]
want = [f"{g}|{c}" for _, g, c, _ in cols]
assert names == want, f"plink2 column order differs from the wide file: {names[:3]} vs {want[:3]}"
fid_i, iid_i = hdr.index("FID") if "FID" in hdr else 0, hdr.index("IID")
n, m = len(rows), len(sc)
S = np.memmap(f"{out}/scores.f32", dtype=np.float32, mode="w+", shape=(n, m))
for r, f in enumerate(rows):
    S[r, :] = np.array([f[i] for i in sc], dtype=np.float64)
S.flush()
with open(f"{out}/samples.txt", "w") as w:
    for f in rows:
        w.write(f"{f[fid_i]}\t{f[iid_i]}\n")
with open(f"{out}/columns.tsv", "w") as w:
    w.write("col\tgene\tchannel\tn_snps\n")
    for j, (_, g, c, k) in enumerate(cols):
        w.write(f"{j}\t{g}\t{c}\t{k}\n")
print(f"oracle: {n} samples x {m} columns -> {out}")
EOF
EOF_VARIANTS_REF_EQTL_PLINK2_RUN_SH
echo "wrote variants/ref_eqtl_plink2/run.sh ($(wc -l < 'variants/ref_eqtl_plink2/run.sh') lines)"

mkdir -p "variants/eqtl_sparse_v0"
cat > 'variants/eqtl_sparse_v0/run.sh' <<'EOF_VARIANTS_EQTL_SPARSE_V0_RUN_SH'
#!/bin/bash
# variants/eqtl_sparse_v0 -- numpy/scipy sparse-matmul implementation of the
# eQTL-weighted features; compared with golden/eqtl_<set> (ref_eqtl_plink2).
set -euo pipefail
FIX=$1; OUT=$2
PY=${BENCH_PY:-python3}; NT=${BENCH_THREADS:-4}
WEIGHTS=${WEIGHTS:-onek1k}
D=$(cd "$(dirname "$0")" && pwd)
"$PY" "$D/score.py" --bed "$FIX/geno" --weights "$FIX/weights/$WEIGHTS.tsv" --out "$OUT" \
      --block "${BLOCK:-4000}" --threads "$NT"
EOF_VARIANTS_EQTL_SPARSE_V0_RUN_SH
echo "wrote variants/eqtl_sparse_v0/run.sh ($(wc -l < 'variants/eqtl_sparse_v0/run.sh') lines)"

mkdir -p "variants/eqtl_sparse_v0"
cat > 'variants/eqtl_sparse_v0/score.py' <<'EOF_VARIANTS_EQTL_SPARSE_V0_SCORE_PY'
#!/usr/bin/env python3
"""eqtl_sparse_v0 -- first fast implementation of eQTL-weighted gene x cell-type
features: dosage matrix (people x SNPs) times a sparse weight matrix
(SNPs x columns), in SNP blocks.

    python score.py --bed <prefix> --weights <long.tsv> --out <dir> [--block 4000] [--threads 4]

Semantics (must equal variants/ref_eqtl_plink2): dosage of the weight's a1,
missing calls imputed to the mean dosage of called people, score = sum of
beta * dosage. No averaging, no thresholds.

Intentionally plain: read a block of SNPs as int8, impute, orient, cast,
multiply by the block's slice of the sparse weight matrix, accumulate.
Everything that could be optimised later -- bit-level dosage decode, fewer
casts, column chunking for memory, multithreaded reads, keeping only SNPs
that carry weight -- is left visible so each change is a separate variant.
"""
import argparse, os, sys, time
import numpy as np
import scipy.sparse as sp
from bed_reader import open_bed

MISSING_INT8 = -127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=4000); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    bed = open_bed(a.bed + ".bed", num_threads=a.threads)
    n = bed.iid_count
    sid = {s: i for i, s in enumerate(bed.sid)}
    bim_a1 = bed.allele_1          # bed-reader: allele_1 = .bim column 5
    bim_a2 = bed.allele_2

    # --- weights -> sparse (n_snps_used x n_cols), oriented to .bim A1
    rows, cols, vals = [], [], []
    colidx, colmeta = {}, []
    n_in = n_drop = 0
    with open(a.weights) as fh:
        hdr = fh.readline().rstrip("\n").split("\t"); ix = {h: i for i, h in enumerate(hdr)}
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if len(f) < len(hdr):
                continue
            n_in += 1
            r = sid.get(f[ix["snp"]])
            if r is None:
                n_drop += 1; continue
            a1 = f[ix["a1"]]
            if a1 == bim_a1[r]:
                beta = float(f[ix["beta"]])
            elif a1 == bim_a2[r]:
                beta = -float(f[ix["beta"]])      # dosage(a2) = 2 - dosage(a1): constant handled below
            else:
                n_drop += 1; continue
            key = (f[ix["gene"]], f[ix["channel"]])
            if key not in colidx:
                colidx[key] = len(colmeta); colmeta.append([key[0], key[1], 0, 0.0])
            c = colidx[key]
            rows.append(r); cols.append(c); vals.append(beta)
            colmeta[c][2] += 1
            if a1 == bim_a2[r]:
                colmeta[c][3] += 2.0 * float(f[ix["beta"]])   # the constant from the flip
    m = len(colmeta)
    Wmat = sp.csr_matrix((vals, (rows, cols)), shape=(len(bed.sid), m))
    used = np.unique(np.asarray(rows))
    print(f"weights: {n_in} rows, {n_drop} dropped; {len(used)} SNPs x {m} columns; "
          f"nnz {Wmat.nnz}  ({time.time() - t0:.1f}s)", flush=True)

    # --- the product, in blocks over the SNPs that carry weight
    S = np.zeros((n, m), dtype=np.float64)
    for b0 in range(0, len(used), a.block):
        blk = used[b0: b0 + a.block]
        X = bed.read(index=np.s_[:, blk], dtype="float32", order="F", num_threads=a.threads)  # NaN = missing
        if np.isnan(X).any():
            mean = np.nanmean(X, axis=0, dtype=np.float64)   # double accumulation, as plink2
            mean = np.where(np.isnan(mean), 0.0, mean)
            ii, jj = np.where(np.isnan(X))
            X[ii, jj] = mean[jj]
        Wb = Wmat[blk, :]                     # (block x m), sparse
        S += X.astype(np.float64) @ Wb        # dense x sparse -> dense
    const = np.array([cm[3] for cm in colmeta])
    S += const[None, :]
    print(f"product done ({time.time() - t0:.1f}s)", flush=True)

    os.makedirs(a.out, exist_ok=True)
    out = np.memmap(os.path.join(a.out, "scores.f32"), dtype=np.float32, mode="w+", shape=(n, m))
    out[:] = S; out.flush()
    with open(os.path.join(a.out, "samples.txt"), "w") as w:
        for f, i in zip(bed.fid, bed.iid):
            w.write(f"{f}\t{i}\n")
    with open(os.path.join(a.out, "columns.tsv"), "w") as w:
        w.write("col\tgene\tchannel\tn_snps\n")
        for j, (g, c, k, _) in enumerate(colmeta):
            w.write(f"{j}\t{g}\t{c}\t{k}\n")
    print(f"eqtl_sparse_v0: {n} x {m} -> {a.out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
EOF_VARIANTS_EQTL_SPARSE_V0_SCORE_PY
echo "wrote variants/eqtl_sparse_v0/score.py ($(wc -l < 'variants/eqtl_sparse_v0/score.py') lines)"

chmod +x make_fixture.sh run_variant.sh variants/*/run.sh
echo "exec bits set on make_fixture.sh run_variant.sh variants/*/run.sh"
echo "done -- now: git init -q && git add -A && git commit -qm \"prs-feature-bench scaffold\""
