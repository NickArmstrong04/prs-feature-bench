#!/usr/bin/env bash
# =============================================================================
#  run_author_splits.sh -- score every method of a run on the PRS-Net authors'
#  six folds, and file PRS-Net's own result next to them.
#
#    DISEASE=ad COHORT=eur bash run_author_splits.sh [--methods "ct ldpred2_grid ..."] [--dry-run]
#
#  Steps (each skipped if its output exists, so it is safe to rerun):
#    1. make_author_splits.py   info.csv (12d) -> $PROC/author_splits/seed42..47.tsv
#    2. SPLITS=author 09_evaluate.sh for each method whose score matrix exists
#       (ct, ldpred2_inf, ldpred2_grid, ldpred2_auto, prsice) -> eval_<m>_author/
#    3. prsnet_to_eval.py        outputs/prsnet/<d>/<c>/train -> eval_prsnet_author/
#    4. 09c_publish.sh for prsnet_author, then the index
#  Everything is light (minutes) except PRSice's ~1,000-column matrix, which
#  09 handles in ~1 h / 32 GB: run this under sbatch or salloc, not on a login
#  node (cluster rule), e.g.
#    sbatch --cpus-per-task=1 --mem=32G --time=3:00:00 --export=ALL,DISEASE=ad,COHORT=eur \
#           --output=../../logs/ad/eur/author_splits_%j.out --wrap "bash run_author_splits.sh"
# =============================================================================
set -euo pipefail
export CHR="${CHR:-1}"
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$SDIR/config.sh" ]] || SDIR="${SLURM_SUBMIT_DIR:-$SDIR}"
source "$SDIR/config.sh"

METHODS="ct ldpred2_inf ldpred2_grid ldpred2_auto prsice"; DRY=0
while (( $# )); do
    case "$1" in
        --methods) METHODS="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) echo "unknown argument $1"; exit 1 ;;
    esac
done

INFO="$PROC/prsnet/genome/${DISEASE}_${COHORT}/info.csv"
SPLITS_DIR="$PROC/author_splits"
TRAIN_DIR="$ROOT/outputs/prsnet/$DISEASE/$COHORT/train"
PY="${PRSNET_PY:-$ROOT/tools/prsnet_venv/bin/python}"; [[ -x "$PY" ]] || PY=python3

say "author-splits evaluation -- $RUN"
echo "info.csv:    $INFO"
echo "splits ->    $SPLITS_DIR"
echo "methods:     $METHODS"
echo "prsnet run:  $TRAIN_DIR"
require_file "$INFO" "12d (feature assembly) has not run for this disease; the authors' folds are defined on its info.csv"
(( DRY )) && { echo "DRY RUN -- nothing executed"; exit 0; }

# 1. folds
if [[ -f "$SPLITS_DIR/seed47.tsv" ]]; then
    echo "have      $SPLITS_DIR (seed42-47)"
else
    "$PY" "$SDIR/make_author_splits.py" --info "$INFO" --out-dir "$SPLITS_DIR"
fi

# 2. each method
for m in $METHODS; do
    case "$m" in
        ct)        mat=$PROC/scores/genome/prs.all_thresholds.txt; ev=$PROC/eval_ct_author ;;
        ldpred2_*) mat=$PROC/ldpred2/genome/prs.$m.txt;           ev=$PROC/eval_${m}_author ;;
        prsice)    mat=$PROC/prsice/genome/prs.prsice.txt;         ev=$PROC/eval_prsice_author ;;
        *) echo "unknown method $m"; exit 1 ;;
    esac
    if [[ ! -f "$mat" ]]; then echo "skip      $m (no score matrix $mat)"; continue; fi
    if [[ -f "$ev/summary.tsv" ]]; then echo "have      $m -> $ev"; continue; fi
    say "09 on authors' folds -- $m"
    SPLITS=author METHOD=$m bash "$SDIR/09_evaluate.sh"
done

# 3. PRS-Net itself
if [[ -f "$TRAIN_DIR/results.tsv" ]]; then
    n=$(( $(wc -l < "$TRAIN_DIR/results.tsv") - 1 ))
    if (( n < 6 )); then
        echo "note: PRS-Net results.tsv has $n of 6 splits -- filing what exists"
    fi
    say "PRS-Net -> eval_prsnet_author"
    "$PY" "$SDIR/prsnet_to_eval.py" --train-dir "$TRAIN_DIR" --splits-dir "$SPLITS_DIR" --info "$INFO" \
        --out-dir "$PROC/eval_prsnet_author"
    METHOD=prsnet_author EVAL_DIR="$PROC/eval_prsnet_author" bash "$SDIR/09c_publish.sh"
else
    echo "skip      prsnet (no $TRAIN_DIR/results.tsv)"
    bash "$SDIR/09c_publish.sh" --index-only
fi

say "paired results for $RUN (authors' folds)"
awk -F'\t' -v r="$RUN" 'NR==1 || ($1==r && $4 ~ /_author$/)' "$ROOT/outputs/benchmark_summary.tsv" | awk -F'\t' '{printf "%-10s %-20s %-8s %-6s %-8s %-8s %-10s\n", $1, $4, $5, $6, $9, $10, $11}'
