#!/bin/bash
#SBATCH --job-name=prs09_eval
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nick.armstrong@yale.edu
#SBATCH --time=1:00:00
#SBATCH --output=/home/naa79/project_pi_sz622/naa79/prs-benchmark/logs/09_eval_%j.out
#SBATCH --error=/home/naa79/project_pi_sz622/naa79/prs-benchmark/logs/09_eval_%j.err
# =============================================================================
#  09_evaluate.sh -- AUROC of a genome-wide PRS (paper l. 165-172)
#
#  Reads a score matrix and the run's phenotype, does the stratified 8:1:1
#  split for three seeds, picks the model (p-value threshold for C+T and
#  PRSice-2, grid point for LDpred2-grid) on validation, reports test AUROC.
#
#  METHOD selects which matrix and where the results go:
#      ct            (default)  scores/genome/prs.all_thresholds.txt -> eval/
#      ldpred2_inf              ldpred2/genome/prs.ldpred2_inf.txt   -> eval_ldpred2_inf/
#      ldpred2_grid             ldpred2/genome/prs.ldpred2_grid.txt  -> eval_ldpred2_grid/
#      ldpred2_auto             ldpred2/genome/prs.ldpred2_auto.txt  -> eval_ldpred2_auto/
#      prsice                   prsice/genome/prs.prsice.txt         -> eval_prsice/
#  The C+T paths are unchanged, so every existing result stays where it is.
#
#  RUN:  bash 09_evaluate.sh                               (~1 min at 400k rows)
#        METHOD=ldpred2_auto bash 09_evaluate.sh
#        METHOD=prsice sbatch --mem=32G --time=4:00:00 09_evaluate.sh   (~1,000 columns)
#        DISEASE=t2d COHORT=all bash 09_evaluate.sh        (the all-ancestry run)
#        sbatch 09_evaluate.sh
# =============================================================================

set -euo pipefail
export CHR=1

# Where config.sh lives. Under sbatch, $0 is Slurm's spool copy and has no
# config.sh beside it, so SLURM_SUBMIT_DIR is the way home. Under salloc,
# SLURM_SUBMIT_DIR is merely wherever salloc was run and has nothing to do with
# this script -- so the script's own directory wins when it holds config.sh.
SDIR="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$SDIR/config.sh" ]] || SDIR="${SLURM_SUBMIT_DIR:-$SDIR}"
source "$SDIR/config.sh"

METHOD="${METHOD:-ct}"
EVAL_SEEDS="${EVAL_SEEDS:-0 1 2}"
# SPLITS=author: score on the PRS-Net authors' six folds (make_author_splits.py
# -> $PROC/author_splits) so every method shares test sets with PRS-Net.
# Results go to eval_<method>_author and outputs/<method>_author/; the
# harness-seed results are untouched.
SPLITS="${SPLITS:-harness}"
SPLITS_ARGS=(); TAG=""; LABEL_SUFFIX=""
if [[ "$SPLITS" == author ]]; then
    SPLITS_DIR="${SPLITS_DIR:-$PROC/author_splits}"
    [[ -f "$SPLITS_DIR/seed42.tsv" ]] || { echo "MISSING: $SPLITS_DIR/seed42.tsv -- run make_author_splits.py first (run_author_splits.sh does)"; exit 1; }
    SPLITS_ARGS=(--splits-dir "$SPLITS_DIR"); TAG="_author"; LABEL_SUFFIX=" (authors' splits)"
elif [[ "$SPLITS" != harness ]]; then
    echo "09_evaluate.sh: SPLITS must be harness or author"; exit 1
fi
case "$METHOD" in
    ct)
        SCORES_MATRIX=$PROC/scores/genome/prs.all_thresholds.txt
        EVAL_DIR="${EVAL_DIR:-$PROC/eval${TAG:+_ct$TAG}}"
        METHOD_LABEL="PLINK C+T"
        PREREQ="Run 07 (run_step.sh 07) first" ;;
    ldpred2_inf|ldpred2_grid|ldpred2_auto)
        SCORES_MATRIX=$PROC/ldpred2/genome/prs.${METHOD}.txt
        EVAL_DIR="${EVAL_DIR:-$PROC/eval_$METHOD$TAG}"
        METHOD_LABEL="LDpred2-${METHOD#ldpred2_}"
        PREREQ="Run 10_ldpred2.sh 10d first (and 10b ${METHOD#ldpred2_} before it)" ;;
    prsice)
        SCORES_MATRIX=$PROC/prsice/genome/prs.prsice.txt
        EVAL_DIR="${EVAL_DIR:-$PROC/eval_prsice$TAG}"
        METHOD_LABEL="PRSice-2"
        PREREQ="Run 11_prsice.sh 11b first (11a for all 22 chromosomes before it)" ;;
    *)  echo "09_evaluate.sh: unknown METHOD '$METHOD' (ct | ldpred2_inf | ldpred2_grid | ldpred2_auto | prsice)"; exit 1 ;;
esac

say "09_evaluate -- $RUN -- $METHOD_LABEL -- inputs"
require_file "$SCORES_MATRIX" "$PREREQ"
[[ -n "$PHENO" ]] || {
    echo "ERROR: no phenotype for this run."
    if [[ "$COHORT" == nopheno ]]; then
        echo "  COHORT=nopheno has no labels by definition, so there is nothing"
        echo "  to compute an AUROC against. Re-run the pipeline with"
        echo "  COHORT=eur (or all / caucasian) to evaluate it."
    else
        echo "  Expected: $PHENO_DEFAULT"
        echo "  Build it:  DISEASE=$DISEASE COHORT=$COHORT bash 08_make_pheno.sh"
    fi
    exit 1; }
require_cmd python3 "python3 missing"
mkdir -p "$EVAL_DIR"
echo "method:     $METHOD ($METHOD_LABEL)"
echo "scores:     $SCORES_MATRIX  ($(( $(head -1 "$SCORES_MATRIX" | awk '{print NF}') - 2 )) score columns)"
echo "phenotype:  $PHENO"
echo "splits:     $SPLITS${SPLITS_DIR:+ ($SPLITS_DIR)}"
[[ "$SPLITS" == author ]] || echo "seeds:      $EVAL_SEEDS"
echo "output ->   $EVAL_DIR"

say "Evaluating  ($(date))"
python3 "$SDIR/09_evaluate.py" \
    --scores "$SCORES_MATRIX" \
    --pheno "$PHENO" \
    --seeds $EVAL_SEEDS \
    --method "$METHOD_LABEL$LABEL_SUFFIX" "${SPLITS_ARGS[@]}" \
    --out-dir "$EVAL_DIR"
# 09c reads this to label the row; an eval/ directory without it is C+T.
printf '%s\t%s\n' "$METHOD$TAG" "$METHOD_LABEL$LABEL_SUFFIX" > "$EVAL_DIR/method.txt"
# For the grid, the chosen label (g037) means nothing without the parameters.
if [[ "$METHOD" == ldpred2_grid && -f "$PROC/ldpred2/genome/grid_params.tsv" ]]; then
    cp -f "$PROC/ldpred2/genome/grid_params.tsv" "$EVAL_DIR/"
    echo
    echo "Chosen grid points (label -> p, h2, sparse):"
    awk -F'\t' 'NR==FNR { if (NR>1) c[$2]=1; next }
                FNR>1 && ($4 in c) { printf "  %-6s p %-8s h2 %-8s sparse %s\n", $4, $1, $2, $3 }' \
        "$EVAL_DIR/summary.tsv" "$EVAL_DIR/grid_params.tsv"
fi
# For PRSice, the chosen threshold means more next to its SNP count.
if [[ "$METHOD" == prsice && -f "$PROC/prsice/genome/thresholds.tsv" ]]; then
    cp -f "$PROC/prsice/genome/thresholds.tsv" "$EVAL_DIR/"
    echo
    echo "Chosen thresholds (seed -> P <= threshold, SNPs genome-wide):"
    awk -F'\t' 'NR==FNR { if (FNR>1) n[$1]=$2; next }
                FNR>1 { printf "  seed %s  P <= %-10s %10s SNPs\n", $1, $2, ($2 in n ? n[$2] : "?") }' \
        "$EVAL_DIR/thresholds.tsv" "$EVAL_DIR/summary.tsv"
fi
say "Done  ($(date)) -- see $EVAL_DIR/README.md and summary.tsv"

# ---------------------------------------------------------------------------
#  Publish the result into $ROOT/outputs: a readable per-run report plus the
#  cross-run index, so every benchmark lands in one place instead of being
#  buried in its own data/processed subtree. Reads what 09_evaluate.py just
#  wrote and computes nothing, so a failure here cannot cost you the
#  evaluation -- it is reported and the step still succeeds.
# ---------------------------------------------------------------------------
if [[ -x "$SDIR/09c_publish.sh" || -f "$SDIR/09c_publish.sh" ]]; then
    EVAL_DIR="$EVAL_DIR" METHOD="$METHOD$TAG" bash "$SDIR/09c_publish.sh" || {
        echo "WARNING: 09c_publish.sh failed. The evaluation itself is fine --"
        echo "  results are in $EVAL_DIR. Re-publish with: METHOD=$METHOD bash 09c_publish.sh"
    }
else
    echo "(09c_publish.sh not found in $SDIR -- results not copied to outputs/)"
fi
