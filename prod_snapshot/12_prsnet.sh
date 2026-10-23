#!/bin/bash
# --- conservative defaults; run_prsnet.sh sets the real values per step ---
#SBATCH --job-name=prsnet
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nick.armstrong@yale.edu
#SBATCH --output=/home/naa79/project_pi_sz622/naa79/prs-benchmark/logs/prsnet_%A_%a.out
#SBATCH --error=/home/naa79/project_pi_sz622/naa79/prs-benchmark/logs/prsnet_%A_%a.err
# =============================================================================
#  12_prsnet.sh -- PRS-Net input features (gene-level C+T), fourth method
#
#  Builds what PRS-Net's train.py reads -- for every person, eleven C+T scores
#  for each of the 19,836 genes of the authors' GGI network -- from the SAME
#  QC'd GWAS, cohort and allele alignment as the C+T / LDpred2 / PRSice runs
#  (steps 01-04 and 08), so the PRS-Net AUROC is comparable with the other
#  bars. The reasoning, the fidelity argument and the run order are in
#  prsnet.md; the arithmetic is in 12_prsnet.py.
#
#  USAGE
#      sbatch --cpus-per-task=2 --mem=8G --time=1:00:00 12_prsnet.sh setup
#      DISEASE=ad COHORT=eur bash run_prsnet.sh --dry-run     (the chain)
#      DISEASE=ad COHORT=eur bash submit_12b_split.sh --chr 19 (12b as small jobs)
#      CHR=19 bash 12_prsnet.sh 12a                            (inside salloc)
#      CHR=19 sbatch --cpus-per-task=4 --mem=32G --time=6:00:00 12_prsnet.sh check
#
#  STEPS
#      setup  once: clone the PRS-Net repo (gene windows + GGI graph), build a
#             Python venv (numpy, scipy, bed-reader), collapse the 19,831 gene
#             BED files into gene_windows.tsv, write the --q-score-range file.
#             Needs the network: compute nodes reach GitHub and PyPI, so an
#             sbatch job is fine; fall back to the login node if pip fails.
#      12a    per chromosome: SNP -> gene-window assignment (the authors'
#             bedtools step) and ONE cohort .bed restricted to the SNPs that
#             fall in any window, A1 = GWAS effect allele (04's command with
#             --make-bed instead of --make-just-bim). The only step that reads
#             the raw UKB release.
#      12b    per chromosome: the authors' per-gene PLINK --clump, run in
#             parallel against that small .bed. Index SNPs per gene. ONE job
#             per chromosome (24 h, serial giant windows at the end).
#      12b-pack / 12b-task / 12b-big / 12b-finish
#             the same 12b as MANY SMALL JOBS, for a queue that will not start
#             a 24 h / 64 GB job: pack splits the chromosome's pending genes
#             into a few balanced tasks (small genes) plus one task per giant
#             window; task and big do the clumping; finish writes
#             clump_manifest.tsv. Submitted by submit_12b_split.sh. Output is
#             identical to 12b (same clump_one, same files).
#      12c    per chromosome: one pass over the .bed -> every gene x threshold
#             score (12_prsnet.py score). Replaces ~900 plink2 runs per
#             chromosome and the 2.6 TB of .sscore text they would write.
#      12d    genome-wide: stack the 22 chromosomes in GRAPH NODE ORDER into
#             feats.f32 + info.csv (and, optionally, the authors' one .npy per
#             person). Needs ref/prsnet/gene_order.txt -- see prsnet.md.
#      check  one chromosome: run the authors' own per-gene plink/plink2
#             commands on a sample of genes and compare, cell by cell, with
#             12c's output. The proof that the fast path changes no number.
#
#  CHR comes from SLURM_ARRAY_TASK_ID under sbatch, else $CHR, else config's
#  default -- except for 12b-task, where the array index is the TASK and CHR
#  must be in the environment. Everything is scoped by DISEASE/COHORT through
#  config.sh.
# =============================================================================

set -euo pipefail

STEP="${1:-}"
if [[ -z "$STEP" ]]; then
    sed -n '/^#  STEPS/,/^#             12c.s output/p' "$0" | sed 's/^#//'
    echo; echo "usage: bash 12_prsnet.sh <setup|12a|12b|12b-pack|12b-task|12b-big|12b-finish|12c|12d|check> [args]"; exit 1
fi
shift
STEP_ARGS=("$@")

if [[ "$STEP" == 12b-task ]]; then
    # split mode: the array index is the task, the chromosome comes from --export
    TASK="${SLURM_ARRAY_TASK_ID:-${TASK:-}}"
    [[ -n "${CHR:-}" && -n "$TASK" ]] || { echo "12b-task needs CHR in the environment and a task id (array index, or TASK=<k>)"; exit 1; }
    CHR_SOURCE="CHR in the environment"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then CHR_SOURCE="array task"; CHR="$SLURM_ARRAY_TASK_ID"
elif [[ -n "${CHR:-}" ]];              then CHR_SOURCE="CHR in the environment"
else                                        CHR_SOURCE="default"; CHR=10; fi
export CHR

SDIR="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$SDIR/config.sh" ]] || SDIR="${SLURM_SUBMIT_DIR:-$SDIR}"
source "$SDIR/config.sh"

for _v in PRSNET_REPO PRSNET_VENV PRSNET_WINDOWS PRSNET_RANGES PRSNET_THRESHOLDS PNET PNET_CHR PNET_GEN; do
    [[ -n "${!_v:-}" ]] || { echo "config.sh does not define $_v -- append the PRS-Net block (prsnet.md)"; exit 1; }
done
PRSNET_PY=$PRSNET_VENV/bin/python
PRSNET_SCRIPT=$SDIR/12_prsnet.py
BED=$PNET_CHR/target_chr${CHR}.PH

if [[ "$CHR_SOURCE" == default && "$STEP" != setup && "$STEP" != 12d ]]; then
    echo; echo "  NOTE: no chromosome given -- running chr${CHR} (the default). Set CHR=<n> or use --array."; echo
fi

require_py() {
    [[ -x "$PRSNET_PY" ]] || { echo "MISSING: $PRSNET_PY -- run: sbatch --cpus-per-task=2 --mem=8G --time=1:00:00 12_prsnet.sh setup"; exit 1; }
    require_file "$PRSNET_SCRIPT" "12_prsnet.py must sit next to 12_prsnet.sh"
}

# =============================================================================
#  setup
# =============================================================================
step_setup() {
    say "prsnet setup -- repo"
    mkdir -p "$(dirname "$PRSNET_REPO")" "$PRSNET_REF"
    if [[ -d "$PRSNET_REPO/.git" ]]; then
        echo "have    $PRSNET_REPO"
    else
        require_cmd git "git missing"
        git clone --depth 1 "$PRSNET_REPO_URL" "$PRSNET_REPO"
    fi
    require_file "$PRSNET_GRAPH" "the clone has no data/ggi_graph.bin"
    [[ -d "$PRSNET_GENE_BED/chr22" ]] || { echo "MISSING: $PRSNET_GENE_BED/chr22"; exit 1; }
    printf '%-44s %12s\n' "gene BED files in the repo" "$(find "$PRSNET_GENE_BED" -name '*.bed' | wc -l)"
    printf '%-44s %12s\n' "ggi_graph.bin bytes" "$(stat -c %s "$PRSNET_GRAPH")"

    say "prsnet setup -- python venv ($PRSNET_VENV)"
    if [[ -n "${PRSNET_PY_MODULE:-}" ]]; then load_modules "$PRSNET_PY_MODULE"; fi
    require_cmd python3 "set PRSNET_PY_MODULE in config.sh to a Python >= 3.9 module (module avail Python)"
    python3 -c 'import sys; assert sys.version_info >= (3, 9), sys.version' 2>/dev/null || {
        echo "python3 is $(python3 --version 2>&1) -- bed-reader needs >= 3.9."
        echo "  Set PRSNET_PY_MODULE in config.sh (try: module avail Python miniconda) and re-run setup."; exit 1; }
    echo "python3:    $(command -v python3)  ($(python3 --version 2>&1))"
    [[ -x "$PRSNET_PY" ]] || python3 -m venv "$PRSNET_VENV"
    "$PRSNET_VENV/bin/pip" install --quiet --upgrade pip
    "$PRSNET_VENV/bin/pip" install --quiet numpy scipy bed-reader
    "$PRSNET_PY" - <<'PYEOF'
import numpy, scipy, bed_reader, sys
print("numpy", numpy.__version__, " scipy", scipy.__version__, " bed_reader ok  (python", sys.version.split()[0] + ")")
PYEOF

    say "prsnet setup -- gene windows -> $PRSNET_WINDOWS"
    "$PRSNET_PY" "$PRSNET_SCRIPT" windows --bed-dir "$PRSNET_GENE_BED" --out "$PRSNET_WINDOWS" \
        --ext-kb "$PRSNET_EXT_KB" --big-window-mb "$PRSNET_GIANT_MB"

    say "prsnet setup -- --q-score-range file -> $PRSNET_RANGES"
    : > "$PRSNET_RANGES"
    for t in $PRSNET_THRESHOLDS; do echo "$t 0 $t" >> "$PRSNET_RANGES"; done
    cat "$PRSNET_RANGES" | tr '\n' ';'; echo
    echo "  (labels are the literal strings generate_prsnet_data.py looks for in the .sscore names)"

    say "prsnet setup -- gene order (graph node order)"
    if [[ -f "$PRSNET_GENE_ORDER" ]]; then
        printf '%-44s %12s\n' "entries in $PRSNET_GENE_ORDER" "$(grep -c . "$PRSNET_GENE_ORDER")"
        echo "  12d checks this against PRSNET_N_GENES=$PRSNET_N_GENES (the graph's node count)."
    else
        echo "  NOT FOUND: $PRSNET_GENE_ORDER"
        echo "  12a-12c do not need it; 12d (assembly) refuses to run without it."
        echo "  It is the list that maps gene -> node index of ggi_graph.bin, which the"
        echo "  repo does not ship: ask Yurui / the authors for the gene list used to"
        echo "  build the graph (prsnet.md, 'The gene-order blocker')."
    fi
    say "setup done. Next: DISEASE=<d> COHORT=eur bash run_prsnet.sh --dry-run"
}

# =============================================================================
#  12a -- SNP -> gene windows, and the cohort .bed (window SNPs, A1 = effect)
# =============================================================================
step_12a() {
    load_modules "$PLINK_MODULE"
    require_py
    say "12a prsnet bed -- chr${CHR} -- inputs"
    require_bfile "$GENO_PREFIX"        "Check GENO_PREFIX in config.sh"
    require_file  "$GQC/gwas.QC.txt"    "Run step 03 for chr${CHR} first"
    require_file  "${TARGET}.PH.bim"    "Run step 04 for chr${CHR} first"
    require_file  "$MIS/target_data.a1" "Run step 03 for chr${CHR} first"
    require_file  "$PRSNET_WINDOWS"     "Run 12_prsnet.sh setup first"
    require_cmd   plink "module load $PLINK_MODULE"
    [[ -n "$KEEP_LIST" ]] || { echo "PRS-Net needs the cohort keep list (08_make_pheno.sh) -- KEEP_LIST is empty"; exit 1; }
    require_file "$KEEP_LIST" "KEEP_LIST is set but missing"
    mkdir -p "$PNET_CHR/genes"
    echo "genotypes: $GENO_PREFIX  (raw; QC via --extract, cohort via --keep)"
    echo "gwas:      $GQC/gwas.QC.txt  ($(count "$GQC/gwas.QC.txt") variants)"
    echo "windows:   $PRSNET_WINDOWS  (+/- $PRSNET_EXT_KB kb already included)"
    echo "keep list: $KEEP_LIST ($(wc -l < "$KEEP_LIST") samples)"
    echo "output ->  $PNET_CHR/{genes/,gene_manifest.tsv,window.snplist}, ${BED}.{bed,bim,fam}"

    say "SNPs per gene window  ($(date))"
    "$PRSNET_PY" "$PRSNET_SCRIPT" snps --chr "$CHR" --windows "$PRSNET_WINDOWS" \
        --gwas "$GQC/gwas.QC.txt" --bim "${TARGET}.PH.bim" \
        --out-dir "$PNET_CHR/genes" --manifest "$PNET_CHR/gene_manifest.tsv" \
        --union "$PNET_CHR/window.snplist" --big-snps "$PRSNET_BIG_SNPS"

    local EXTRACT=$PNET_CHR/window.snplist
    if [[ "${PRSNET_BED_SCOPE:-window}" == all ]]; then EXTRACT=$GQC/snplist; fi
    [[ -s "$EXTRACT" ]] || { echo "ERROR: $EXTRACT is empty -- no window SNP on chr${CHR}?"; exit 1; }

    say "Building the cohort .bed  ($(date))"
    echo "  (04's command with --make-bed: the one pass over the raw release this method needs)"
    plink --bfile "$GENO_PREFIX" \
          --keep "$KEEP_LIST" \
          --extract "$EXTRACT" \
          --a1-allele "$MIS/target_data.a1" \
          --make-bed \
          --out "$BED" \
          --memory "$PLINK_MEM" --threads "$PLINK_THREADS"
    say "PLINK finished  ($(date))"

    say "12a prsnet bed -- chr${CHR} -- results"
    printf '%-44s %12s\n' "SNPs in gwas.QC.txt"          "$(count "$GQC/gwas.QC.txt")"
    printf '%-44s %12s\n' "SNPs in some gene window"     "$(wc -l < "$PNET_CHR/window.snplist")"
    printf '%-44s %12s\n' "variants written"             "$(wc -l < "${BED}.bim")"
    printf '%-44s %12s\n' "samples written"              "$(wc -l < "${BED}.fam")"
    printf '%-44s %12s\n' "bed size"                     "$(du -h "${BED}.bed" | cut -f1)"
    printf '%-44s %12s\n' "genes on chr${CHR}"           "$(count "$PNET_CHR/gene_manifest.tsv")"
    printf '%-44s %12s\n' "  with >= 1 SNP"              "$(awk 'NR>1 && $4>0' "$PNET_CHR/gene_manifest.tsv" | wc -l)"
    printf '%-44s %12s\n' "  'big' (> $PRSNET_BIG_SNPS SNPs, clumped alone)" "$(awk 'NR>1 && $5==1' "$PNET_CHR/gene_manifest.tsv" | wc -l)"
    echo
    echo "Allele orientation -- .bim A1 (col 5) must equal gwas A1 (col 4) for every variant:"
    export LC_ALL=C
    join -1 2 -2 3 <(sort -k2,2 "${BED}.bim") <(tail -n +2 "$GQC/gwas.QC.txt" | sort -k3,3) 2>/dev/null \
      | awk '{if($5==$9) m++; else x++} END{printf "  matching: %d\n  mismatched: %d  (must be 0)\n", m+0, x+0}' || true

    say "Canary check"
    if [[ "$CHR" != "$CANARY_CHR" ]]; then
        echo "SKIP  canary $CANARY_RSID lives on chr${CANARY_CHR}"
    elif grep -qw "$CANARY_RSID" "${BED}.bim"; then
        echo "PASS  $CANARY_RSID is in the cohort .bed; gene windows containing it:"
        grep -lw "$CANARY_RSID" "$PNET_CHR"/genes/*.snplist 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.snplist$//; s/^/      /' || echo "      (none -- it lies outside every gene window +/- $PRSNET_EXT_KB kb)"
    else
        echo "WARN  $CANARY_RSID is not in the cohort .bed -- outside every gene window, or lost upstream"
    fi
    say "Next: 12_prsnet.sh 12b (same chromosome), or submit_12b_split.sh --chr ${CHR}"
}

# =============================================================================
#  12b -- the authors' per-gene PLINK --clump, in parallel
# =============================================================================
clump_one() {   # <gene> <memory MB>   -- never fails the caller; a missing .clumped is the signal
    local g=$1 mem=$2
    local out=$PNET_CHR/clumps/$g      # separate `local`: a builtin expands ALL its
                                       # arguments before assigning any, so $g would be empty
    [[ -f "$out.clumped" && "${PRSNET_FORCE:-0}" != 1 ]] && return 0      # resume
    plink --bfile "$PNET_CHR/target_chr${CHR}.PH" \
          --extract "$PNET_CHR/genes/$g.snplist" \
          --chr "$CHR" \
          --clump-p1 1 \
          --clump-r2 "$PRSNET_CLUMP_R2" \
          --clump-kb "$PRSNET_CLUMP_KB" \
          --clump "$PNET_CHR/genes/$g.assoc" "$GQC/gwas.QC.txt" \
          --clump-index-first \
          --clump-snp-field SNP \
          --clump-field P \
          --out "$out" \
          --memory "$mem" --threads 1 > /dev/null 2>&1 || true
    return 0
}
export -f clump_one

clump_inputs() {   # shared by 12b, 12b-pack, 12b-task, 12b-big
    load_modules "$PLINK_MODULE"
    require_bfile "$BED"                        "Run 12a for chr${CHR} first"
    require_file  "$PNET_CHR/gene_manifest.tsv" "Run 12a for chr${CHR} first"
    require_file  "$GQC/gwas.QC.txt"            "Run step 03 for chr${CHR} first"
    require_cmd   plink "module load $PLINK_MODULE"
    mkdir -p "$PNET_CHR/clumps"
    export PNET_CHR GQC CHR PRSNET_CLUMP_R2 PRSNET_CLUMP_KB PRSNET_FORCE="${PRSNET_FORCE:-0}"
}

write_12b_results() {   # clump_manifest.tsv + the summary + the canary; shared by 12b and 12b-finish
    say "Writing clump_manifest.tsv"
    awk -v D="$PNET_CHR/clumps" 'BEGIN{OFS="\t"}
         NR==1 {print $0, "n_index", "status"; next}
         { n=0; st="nosnp"
           if ($4>0) { f=D"/"$1".clumped"; st="failed"
               if ((getline l < f) > 0) { st="ok"; while ((getline l < f) > 0) { split(l,a," "); if (length(a)>=3) n++ } }
               close(f) }
           print $0, n, st }' "$PNET_CHR/gene_manifest.tsv" > "$PNET_CHR/clump_manifest.tsv"

    say "12b prsnet clump -- chr${CHR} -- results"
    printf '%-44s %12s\n' "genes"                     "$(count "$PNET_CHR/clump_manifest.tsv")"
    printf '%-44s %12s\n' "  clumped (ok)"            "$(awk 'NR>1 && $7=="ok"' "$PNET_CHR/clump_manifest.tsv" | wc -l)"
    printf '%-44s %12s\n' "  no SNP in window"        "$(awk 'NR>1 && $7=="nosnp"' "$PNET_CHR/clump_manifest.tsv" | wc -l)"
    printf '%-44s %12s\n' "  FAILED"                  "$(awk 'NR>1 && $7=="failed"' "$PNET_CHR/clump_manifest.tsv" | wc -l)"
    printf '%-44s %12s\n' "window SNPs"               "$(awk 'NR>1 {s+=$4} END{print s+0}' "$PNET_CHR/clump_manifest.tsv")"
    printf '%-44s %12s\n' "index SNPs (sum over genes)" "$(awk 'NR>1 {s+=$6} END{print s+0}' "$PNET_CHR/clump_manifest.tsv")"
    if awk 'NR>1 && $7=="failed"' "$PNET_CHR/clump_manifest.tsv" | grep -q .; then
        echo; echo "Failed genes (look at $PNET_CHR/clumps/<gene>.log; rerun 12b or submit_12b_split.sh -- done genes are skipped):"
        awk 'NR>1 && $7=="failed" && ++n<=30 {print "  "$1, "("$4" SNPs)"}' "$PNET_CHR/clump_manifest.tsv"
        echo; echo "12c refuses to run while any gene is 'failed'."
    fi
    echo
    echo "Sanity: on dense data at r2 0.5 / 250 kb, roughly 30-60% of a gene's window SNPs"
    echo "survive as index SNPs (windows are short, so most pairs are within 250 kb)."

    say "Canary check"
    if [[ "$CHR" != "$CANARY_CHR" ]]; then
        echo "SKIP  canary $CANARY_RSID lives on chr${CANARY_CHR}"
    else
        local hit=0 g f
        for f in $(grep -lw "$CANARY_RSID" "$PNET_CHR"/genes/*.snplist 2>/dev/null); do
            g=$(basename "$f" .snplist); hit=1
            if [[ -f "$PNET_CHR/clumps/$g.clumped" ]] && awk 'NR>1 && NF>=3 {print $3}' "$PNET_CHR/clumps/$g.clumped" | grep -qxw "$CANARY_RSID"; then
                echo "PASS  $CANARY_RSID is an index SNP of $g"
            else
                echo "WARN  $CANARY_RSID is in $g's window but absorbed into another clump (benign if its P ties with a neighbour's)"
            fi
        done
        (( hit )) || echo "SKIP  $CANARY_RSID is in no gene window on chr${CHR}"
    fi
    say "Next: 12_prsnet.sh 12c (same chromosome)"
}

step_12b() {
    clump_inputs
    say "12b prsnet clump -- chr${CHR} -- inputs"
    local NPAR=${PRSNET_PARALLEL:-$PLINK_THREADS}
    local PER_MEM=$(( PLINK_MEM / NPAR ))
    awk 'NR>1 && $4>0 && $5==0 {print $1}' "$PNET_CHR/gene_manifest.tsv" > "$PNET_CHR/genes.small"
    awk 'NR>1 && $4>0 && $5==1 {print $1}' "$PNET_CHR/gene_manifest.tsv" > "$PNET_CHR/genes.big"
    echo "bed:       $BED  ($(wc -l < "${BED}.bim") variants, $(wc -l < "${BED}.fam") samples)"
    echo "params:    r2 $PRSNET_CLUMP_R2 | window $PRSNET_CLUMP_KB kb | p1 1 | p2 PLINK default (the authors' flags)"
    echo "genes:     $(wc -l < "$PNET_CHR/genes.small") small, $NPAR at a time with ${PER_MEM} MB each;"
    echo "           $(wc -l < "$PNET_CHR/genes.big") big (> $PRSNET_BIG_SNPS SNPs), one at a time with ${PLINK_MEM} MB"
    echo "resume:    genes with an existing .clumped are skipped (PRSNET_FORCE=1 redoes them)"
    echo "output ->  $PNET_CHR/clumps/<gene>.clumped, clump_manifest.tsv"

    say "Clumping small genes  ($(date))"
    xargs -a "$PNET_CHR/genes.small" -P "$NPAR" -I{} bash -c 'clump_one "$1" "$2"' _ {} "$PER_MEM"
    say "Clumping big genes  ($(date))"
    local g
    while read -r g; do
        echo "  $g  ($(wc -l < "$PNET_CHR/genes/$g.snplist") SNPs)  $(date +%H:%M:%S)"
        clump_one "$g" "$PLINK_MEM"
    done < "$PNET_CHR/genes.big"
    say "PLINK finished  ($(date))"
    write_12b_results
}

# =============================================================================
#  12b as many small jobs:  12b-pack  ->  12b-task (array) + 12b-big (one job
#  per giant window)  ->  12b-finish.  Submitted by submit_12b_split.sh.
# =============================================================================
#  Knobs (config.sh or the environment):
#    PRSNET_TASK_SNPS     window SNPs per small task, ~150k = a few minutes of
#                         PLINK per task at 4 CPUs (default 150000)
#    PRSNET_SMALL_TASKS   cap on small tasks per chromosome (default 8) -- the
#                         queue counts jobs, so fewer, fatter tasks are better
#    PRSNET_BIG_MEM_MAX   MB ceiling for a giant-window job (default 122880)
#    PRSNET_BIG_TIME      time limit for a giant-window job (default 4:00:00 -- a
#                         24 h request never backfills on this account; chr19's
#                         ERICH4, 113,666 SNPs, took 10 min)
#
#  Giant-window memory: PLINK --clump holds the extracted genotypes, ~ samples/4
#  bytes per SNP (381k samples -> ~95 KB per SNP), plus the LD workspace; the
#  pack step budgets 0.12 MB per window SNP + 4 GB, floor 8 GB, cap
#  PRSNET_BIG_MEM_MAX. --clump is single-threaded, so these jobs ask for 1 CPU.
# =============================================================================
step_12b_pack() {
    clump_inputs
    say "12b-pack -- chr${CHR} -- split the pending genes into tasks"
    local T=$PNET_CHR/tasks
    local BUDGET=${PRSNET_TASK_SNPS:-150000} MAXSMALL=${PRSNET_SMALL_TASKS:-8} MEMMAX=${PRSNET_BIG_MEM_MAX:-122880}
    mkdir -p "$T"; rm -f "$T"/small_*.genes "$T/small.tsv" "$T/big.tsv"
    # pending = has SNPs and no .clumped yet (PRSNET_FORCE=1 ignores existing .clumped)
    awk -v D="$PNET_CHR/clumps" -v F="$PRSNET_FORCE" 'NR>1 && $4>0 { f=D"/"$1".clumped"
         if (F==1 || (getline l < f) <= 0) print $1, $4, $5; close(f) }' "$PNET_CHR/gene_manifest.tsv" \
         | sort -k2,2nr > "$T/pending.txt"
    local n_all n_pending n_small n_big
    n_all=$(awk 'NR>1 && $4>0' "$PNET_CHR/gene_manifest.tsv" | wc -l)
    n_pending=$(wc -l < "$T/pending.txt")
    n_small=$(awk '$3==0' "$T/pending.txt" | wc -l)
    n_big=$(awk '$3==1' "$T/pending.txt" | wc -l)
    echo "genes with SNPs: $n_all | pending: $n_pending ($n_small small, $n_big big) | done: $((n_all - n_pending))"

    # small genes: largest first into the least-loaded of K bins (K = ceil(total/BUDGET), capped)
    awk -v B="$BUDGET" -v K="$MAXSMALL" -v T="$T" '
        $3==0 { g[++n]=$1; s[n]=$2; tot+=$2 }
        END {
            if (n==0) exit
            k=int((tot+B-1)/B); if (k<1) k=1; if (k>K) k=K
            for (i=1;i<=n;i++) { best=0; for (j=0;j<k;j++) if (j==0 || load[j]<load[best]) best=j
                load[best]+=s[i]; cnt[best]++; print g[i] >> (T"/small_"best".genes") }
            for (j=0;j<k;j++) printf "%d\t%d\t%d\n", j, cnt[j], load[j] > (T"/small.tsv")
        }' "$T/pending.txt"
    # big genes: one task each, with a memory estimate from the window size
    awk -v MAX="$MEMMAX" '$3==1 { m=int($2*0.12+4096); if (m<8192) m=8192; if (m>MAX) m=MAX
         printf "%s\t%d\t%d\n", $1, $2, m }' "$T/pending.txt" > "$T/big.tsv"

    [[ -f "$T/small.tsv" ]] && { echo; echo "small tasks (task  genes  windowSNPs):"; sed 's/^/  /' "$T/small.tsv"; }
    [[ -s "$T/big.tsv"   ]] && { echo; echo "big tasks (gene  windowSNPs  memMB):";   sed 's/^/  /' "$T/big.tsv"; }
    echo
    echo "tasks -> $T/{small.tsv,small_<k>.genes,big.tsv}"
    say "Next: submit_12b_split.sh does the sbatch calls"
}

step_12b_task() {
    clump_inputs
    local T=$PNET_CHR/tasks
    require_file "$T/small_${TASK}.genes" "no task $TASK on chr${CHR} -- run 12b-pack first"
    local NPAR=${PRSNET_PARALLEL:-${SLURM_CPUS_PER_TASK:-$PLINK_THREADS}}
    local MEM=${SLURM_MEM_PER_NODE:-$PLINK_MEM}
    local PER_MEM=$(( (MEM - 1024) / NPAR ))
    say "12b-task -- chr${CHR} task ${TASK}: $(wc -l < "$T/small_${TASK}.genes") genes, $NPAR at a time with ${PER_MEM} MB each  ($(date))"
    xargs -a "$T/small_${TASK}.genes" -P "$NPAR" -I{} bash -c 'clump_one "$1" "$2"' _ {} "$PER_MEM"
    local done_n
    done_n=$(while read -r g; do [[ -f "$PNET_CHR/clumps/$g.clumped" ]] && echo x; done < "$T/small_${TASK}.genes" | wc -l)
    say "task ${TASK} done: $done_n of $(wc -l < "$T/small_${TASK}.genes") genes have a .clumped  ($(date))"
}

step_12b_big() {
    clump_inputs
    local g=${STEP_ARGS[0]:?12b-big needs a gene name}
    require_file "$PNET_CHR/genes/$g.snplist" "no window SNP list for $g on chr${CHR}"
    local MEM=${SLURM_MEM_PER_NODE:-$PLINK_MEM}
    say "12b-big -- chr${CHR} $g: $(wc -l < "$PNET_CHR/genes/$g.snplist") window SNPs, $((MEM - 1024)) MB  ($(date))"
    clump_one "$g" "$((MEM - 1024))"
    if [[ -f "$PNET_CHR/clumps/$g.clumped" ]]; then
        say "$g done: $(awk 'NR>1 && NF>=3' "$PNET_CHR/clumps/$g.clumped" | wc -l) index SNPs  ($(date))"
    else
        say "$g FAILED -- see $PNET_CHR/clumps/$g.log (out of memory? raise PRSNET_BIG_MEM_MAX; out of time? PRSNET_BIG_TIME)"; exit 1
    fi
}

step_12b_finish() {
    clump_inputs
    say "12b-finish -- chr${CHR}"
    write_12b_results
    if awk 'NR>1 && $7=="failed"' "$PNET_CHR/clump_manifest.tsv" | grep -q .; then
        echo "Some genes failed -- resubmit: DISEASE=$DISEASE COHORT=$COHORT bash submit_12b_split.sh --chr $CHR  (packs only the pending genes)"
        exit 2
    fi
}

# =============================================================================
#  12c -- one pass: every gene x threshold score for this chromosome
# =============================================================================
step_12c() {
    require_py
    say "12c prsnet score -- chr${CHR} -- inputs"
    require_bfile "$BED"                         "Run 12a for chr${CHR} first"
    require_file  "$PNET_CHR/clump_manifest.tsv" "Run 12b for chr${CHR} first"
    require_file  "$GQC/gwas.QC.txt"             "Run step 03 for chr${CHR} first"
    echo "bed:        $BED"
    echo "clumps:     $PNET_CHR/clumps  ($(awk 'NR>1 && $7=="ok"' "$PNET_CHR/clump_manifest.tsv" | wc -l) genes)"
    echo "thresholds: $PRSNET_THRESHOLDS"
    echo "output ->   $PNET_CHR/scores/scores_chunk<k>.npy  (samples x genes x 11, float32)"
    export OPENBLAS_NUM_THREADS=$PLINK_THREADS OMP_NUM_THREADS=$PLINK_THREADS
    say "Scoring  ($(date))"
    "$PRSNET_PY" "$PRSNET_SCRIPT" score --bed "$BED" --gwas "$GQC/gwas.QC.txt" \
        --clumps "$PNET_CHR/clumps" --manifest "$PNET_CHR/clump_manifest.tsv" \
        --out-dir "$PNET_CHR/scores" --thresholds "$PRSNET_THRESHOLDS" \
        --chunk-genes "$PRSNET_CHUNK_GENES" --block-snps "$PRSNET_BLOCK_SNPS" \
        --giant-window-mb "$PRSNET_GIANT_MB" --threads "$PLINK_THREADS"
    say "12c prsnet score -- chr${CHR} -- results"
    printf '%-44s %12s\n' "chunk files"   "$(count "$PNET_CHR/scores/scores_manifest.tsv")"
    printf '%-44s %12s\n' "genes scored"  "$(count "$PNET_CHR/scores/gene_counts.tsv")"
    printf '%-44s %12s\n' "disk"          "$(du -sh "$PNET_CHR/scores" | cut -f1)"
    echo
    echo "Index SNPs per threshold, summed over genes (must be non-decreasing left to right):"
    awk -F'\t' 'NR==1 {for(i=3;i<=NF;i++) h[i]=$i; next} {for(i=3;i<=NF;i++) s[i]+=$i}
         END {for(i=3;i<=NF;i++) printf "  P<=%-8s %10d\n", substr(h[i],6), s[i]}' "$PNET_CHR/scores/gene_counts.tsv"
    if [[ "$CHR" == "$CANARY_CHR" ]]; then
        echo; echo "Canary gene rows of gene_counts.tsv (index SNPs cumulative per threshold):"
        for f in $(grep -lw "$CANARY_RSID" "$PNET_CHR"/genes/*.snplist 2>/dev/null); do
            grep -w "^$(basename "$f" .snplist)" "$PNET_CHR/scores/gene_counts.tsv" | sed 's/^/  /' || true
        done
    fi
    say "Next: after ALL chromosomes, 12_prsnet.sh 12d (no array); or 12_prsnet.sh check on this one"
}

# =============================================================================
#  12d -- assemble the genome-wide feature store in graph node order
# =============================================================================
step_12d() {
    require_py
    say "12d prsnet assemble -- $RUN -- inputs"
    local c
    for c in $(seq 1 22); do
        require_file "$PNET/chr${c}/scores/scores_manifest.tsv" "12c has not completed for chr${c}"
    done
    [[ -n "$PHENO" ]] || { echo "no phenotype for $RUN -- run 08_make_pheno.sh"; exit 1; }
    require_file "$PHENO" "PHENO is set but missing"
    if [[ ! -f "$PRSNET_GENE_ORDER" ]]; then
        echo "MISSING: $PRSNET_GENE_ORDER"
        echo "  The feature vector must list genes in the node order of ggi_graph.bin, and"
        echo "  the repo does not say what that order is (prsnet.md, 'The gene-order blocker')."
        echo "  Put the authors' gene list there (one gene per line, $PRSNET_N_GENES lines) and re-run 12d."
        exit 1
    fi
    local ANC=UNK; [[ "$COHORT" == eur || "$COHORT" == caucasian ]] && ANC=EUR
    local OUT=$PNET_GEN/${DISEASE}_${COHORT}
    echo "chromosomes: $PNET/chr{1..22}/scores"
    echo "gene order:  $PRSNET_GENE_ORDER ($(grep -c . "$PRSNET_GENE_ORDER") genes; graph has $PRSNET_N_GENES nodes)"
    echo "phenotype:   $PHENO"
    echo "export:      ${PRSNET_EXPORT:-memmap}"
    echo "output ->    $OUT/{feats.f32,feats.meta.json,info.csv}  (+ feats/<id>.npy if exported)"
    echo "             train.py reads it as: --data_path $PNET_GEN --dataset ${DISEASE}_${COHORT}"
    mkdir -p "$OUT"
    say "Assembling  ($(date))"
    "$PRSNET_PY" "$PRSNET_SCRIPT" assemble --run-dir "$PNET" --gene-order "$PRSNET_GENE_ORDER" \
        --pheno "$PHENO" --ancestry "$ANC" --out-dir "$OUT" --thresholds "$PRSNET_THRESHOLDS" \
        --export "${PRSNET_EXPORT:-memmap}" --expect-n-genes "$PRSNET_N_GENES" --sample-block 5000
    say "12d prsnet assemble -- results"
    printf '%-44s %12s\n' "samples (info.csv)"  "$(count "$OUT/info.csv")"
    printf '%-44s %12s\n' "cases"               "$(awk -F, 'NR>1 && $2==1' "$OUT/info.csv" | wc -l)"
    printf '%-44s %12s\n' "genes without features (zero columns)" "$(grep -c . "$OUT/genes_without_features.txt" || true)"
    printf '%-44s %12s\n' "scored genes not in the order (dropped)" "$(grep -c . "$OUT/genes_not_in_order.txt" || true)"
    printf '%-44s %12s\n' "feature store"       "$(du -sh "$OUT" | cut -f1)"
    cat "$OUT/feats.meta.json"
    say "Done. The ML part (train.py on a Blackwell node) reads $OUT -- see prsnet.md, 'Training'."
}

# =============================================================================
#  check -- the authors' per-gene plink/plink2 commands vs 12c, one chromosome
# =============================================================================
step_check() {
    # plink 1.9 and plink2 are the same Lmod family on Bouchet: loading the
    # second module unloads the first. Resolve each binary to its absolute
    # path right after its own module load; the files stay on disk.
    load_modules "$PLINK_MODULE";  require_cmd plink  "module load $PLINK_MODULE";  local PLINK1;  PLINK1=$(command -v plink)
    load_modules "$PLINK2_MODULE"; require_cmd plink2 "module load $PLINK2_MODULE"; local PLINK2;  PLINK2=$(command -v plink2)
    echo "plink 1.9: $PLINK1"; echo "plink2:    $PLINK2"
    require_py
    local N=${STEP_ARGS[0]:-40}
    say "prsnet check -- chr${CHR} -- inputs"
    require_bfile "$BED"                              "Run 12a for chr${CHR} first"
    require_file  "$PNET_CHR/clump_manifest.tsv"      "Run 12b for chr${CHR} first"
    require_file  "$PNET_CHR/scores/scores_manifest.tsv" "Run 12c for chr${CHR} first"
    require_file  "$GQC/SNP.pvalue"                   "Run step 04 for chr${CHR} first"
    require_file  "$PRSNET_RANGES"                    "Run setup first"
    local CK=$PNET_CHR/check; mkdir -p "$CK/clumps" "$CK/sscore"
    # Sample: the N/2 small genes with the most index SNPs, N/2 at random
    # (seeded), plus every gene whose window holds the canary.
    # (no `head` here: under pipefail, head closing the pipe early makes sort
    #  exit 141 and set -e kills the job silently -- awk reads everything)
    { awk 'NR>1 && $7=="ok" && $5==0 {print $6, $1}' "$PNET_CHR/clump_manifest.tsv" | sort -k1,1nr | awk -v n=$((N / 2)) 'NR<=n {print $2}'
      awk 'NR>1 && $7=="ok" && $5==0 {print $1}' "$PNET_CHR/clump_manifest.tsv" | awk 'BEGIN{srand(22)} {print rand(), $0}' | sort -k1,1n | awk -v n=$((N - N / 2)) 'NR<=n {print $2}'
      if [[ "$CHR" == "$CANARY_CHR" ]]; then
          grep -lw "$CANARY_RSID" "$PNET_CHR"/genes/*.snplist 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.snplist$//' || true
      fi
    } | sort -u > "$CK/genes.txt"
    echo "genes:     $(wc -l < "$CK/genes.txt")  ($CK/genes.txt)"
    echo "clump:     the authors' command verbatim, incl. --extract <gene>.assoc"
    echo "score:     plink2 --score gwas.QC.txt 3 4 12 header --q-score-range $PRSNET_RANGES SNP.pvalue --extract <gene>.assoc.clumped"
    echo "output ->  $CK/{clumps,sscore}, then 12_prsnet.py compare"

    say "Running the authors' per-gene commands  ($(date))"
    local g q_hdr="" x_mode=assoc n_clump_diff=0
    while read -r g; do
        # --- clump, their way (whole .assoc file handed to --extract) ------
        "$PLINK1" --bfile "$BED" --extract "$PNET_CHR/genes/$g.assoc" --chr "$CHR" \
              --clump-p1 1 --clump-r2 "$PRSNET_CLUMP_R2" --clump-kb "$PRSNET_CLUMP_KB" \
              --clump "$PNET_CHR/genes/$g.assoc" "$GQC/gwas.QC.txt" --clump-index-first \
              --clump-snp-field SNP --clump-field P \
              --out "$CK/clumps/$g.assoc" --memory "$PLINK_MEM" --threads "$PLINK_THREADS" > /dev/null 2>&1 || true
        if ! cmp -s <(awk 'NR>1 && NF>=3 {print $3}' "$CK/clumps/$g.assoc.clumped" 2>/dev/null | sort) \
                    <(awk 'NR>1 && NF>=3 {print $3}' "$PNET_CHR/clumps/$g.clumped" | sort); then
            echo "  CLUMP DIFF: $g -- index set with --extract .assoc differs from 12b's (--extract .snplist)"
            n_clump_diff=$((n_clump_diff + 1))
        fi
        # --- score, their way --------------------------------------------
        local ex=$CK/clumps/$g.assoc.clumped
        [[ "$x_mode" == ids ]] && { awk 'NR>1 && NF>=3 {print $3}' "$ex" > "$ex.ids"; ex=$ex.ids; }
        if ! "$PLINK2" --bfile "$BED" --chr "$CHR" \
                    --score "$GQC/gwas.QC.txt" 3 4 12 header \
                    --q-score-range "$PRSNET_RANGES" "$GQC/SNP.pvalue" $q_hdr \
                    --extract "$ex" \
                    --out "$CK/sscore/$g.assoc.clumped" --memory "$PLINK_MEM" --threads "$PLINK_THREADS" > "$CK/sscore/$g.stdout" 2>&1; then
            if grep -qi "q-score-range" "$CK/sscore/$g.stdout" && [[ -z "$q_hdr" ]]; then
                echo "  NOTE: plink2 rejected SNP.pvalue's header line without 'header'; adding it (plink2 $("$PLINK2" --version 2>&1 | head -1))"
                q_hdr=header
            elif grep -qi "No variants remaining after --extract" "$CK/sscore/$g.stdout" && [[ "$x_mode" == assoc ]]; then
                echo "  NOTE: this plink2 does not read variant IDs out of a whole .clumped file; passing column 3 instead"
                x_mode=ids
            else
                echo "  plink2 FAILED on $g -- see $CK/sscore/$g.stdout"; continue
            fi
            awk 'NR>1 && NF>=3 {print $3}' "$CK/clumps/$g.assoc.clumped" > "$CK/clumps/$g.assoc.clumped.ids"
            [[ "$x_mode" == ids ]] && ex=$CK/clumps/$g.assoc.clumped.ids
            "$PLINK2" --bfile "$BED" --chr "$CHR" --score "$GQC/gwas.QC.txt" 3 4 12 header \
                   --q-score-range "$PRSNET_RANGES" "$GQC/SNP.pvalue" $q_hdr --extract "$ex" \
                   --out "$CK/sscore/$g.assoc.clumped" --memory "$PLINK_MEM" --threads "$PLINK_THREADS" > "$CK/sscore/$g.stdout" 2>&1 \
                || { echo "  plink2 FAILED on $g -- see $CK/sscore/$g.stdout"; continue; }
        fi
    done < "$CK/genes.txt"
    say "Comparing  ($(date))"
    echo "clump index sets differing between the two --extract forms: $n_clump_diff (must be 0)"
    echo
    "$PRSNET_PY" "$PRSNET_SCRIPT" compare --scores-dir "$PNET_CHR/scores" --sscore-dir "$CK/sscore" \
        --genes "$CK/genes.txt" --thresholds "$PRSNET_THRESHOLDS" --rtol "${PRSNET_CHECK_RTOL:-1e-5}" \
        | tee "$CK/compare.txt"
    local rc=${PIPESTATUS[0]}
    (( rc == 0 && n_clump_diff == 0 )) && say "CHECK PASSED on chr${CHR}: the one-pass features equal the authors' pipeline output" \
                                        || { say "CHECK FAILED on chr${CHR} -- see $CK/compare.txt"; exit 2; }
}

case "$STEP" in
    setup)      step_setup ;;
    12a)        step_12a ;;
    12b)        step_12b ;;
    12b-pack)   step_12b_pack ;;
    12b-task)   step_12b_task ;;
    12b-big)    step_12b_big ;;
    12b-finish) step_12b_finish ;;
    12c)        step_12c ;;
    12d)        step_12d ;;
    check)      step_check ;;
    *) echo "12_prsnet.sh: unknown step '$STEP' (setup|12a|12b|12b-pack|12b-task|12b-big|12b-finish|12c|12d|check)"; exit 1 ;;
esac
