#!/bin/bash
# =============================================================================
#  variants/ldp_sum_v0/run.sh -- LDpred2-posterior gene features: ct_sparse_v5's
#  kernel in --mode sum on a weight file of LDpred2 posterior effects, no 12b.
#
#  Feature (i, gene g, channel c) = sum_j beta_post[j, c] * dosage_i(a1_j) over the
#  HapMap3+ SNPs in g's window (README s14). No clumping, no P thresholds, no
#  SCORE_AVG division. The channel is an LDpred2 model (auto, inf, or a grid point
#  g###); the output is 12c's chunk layout with T = number of channels, so 12d and
#  train_prsnet.py take it as they take the 11 C+T thresholds.
#
#  The weight file is made ONCE per (disease, channel set, assignment) by
#  lib/make_ldpred_weights.py from production's betas.txt and placed in
#  fixture/weights/<set>.tsv; the oracle for it is ref_eqtl_plink2 with the same
#  WEIGHTS (plink2 multi-column --score, scoresums):
#
#      WEIGHTS=ldp_auto bash run_variant.sh ref_eqtl_plink2 --as-golden eqtl_ldp_auto --tag "oracle"
#      WEIGHTS=ldp_auto bash run_variant.sh ldp_sum_v0 --golden eqtl_ldp_auto --tag "..."
#
#  Columns: every gene of fixture/gene_manifest.tsv x every channel in the weight
#  file (zero columns for genes without a weighted SNP); the oracle drops columns
#  with no usable SNP, so those show up as "extra, not compared".
#
#  Called by run_variant.sh as:  WEIGHTS=<set> run.sh <fixture_dir> <out_dir>
# =============================================================================
set -euo pipefail
FIX=$1; OUT=$2
W=$OUT/work; mkdir -p "$W"
PY=${BENCH_PY:-python3}; NT=${BENCH_THREADS:-4}
D=$(cd "$(dirname "$0")" && pwd)
WEIGHTS=${WEIGHTS:-ldp_auto}
WFILE=$FIX/weights/$WEIGHTS.tsv
[[ -f "$WFILE" ]] || { echo "no weight set: $WFILE  (make it with lib/make_ldpred_weights.py, README s14)"; exit 1; }

# every manifest gene gets columns; channels = the weight file's, in first-seen order
awk -F'\t' 'NR>1 {print $1}' "$FIX/gene_manifest.tsv" > "$W/genes_all.txt"
CHANNELS=$(awk -F'\t' 'NR>1 && !seen[$4]++ {printf "%s%s", (n++ ? " " : ""), $4}' "$WFILE")
[[ -n "$CHANNELS" ]] || { echo "no channel column values in $WFILE"; exit 1; }
echo "weights: $WFILE ($(awk 'NR>1' "$WFILE" | wc -l) rows); channels: $CHANNELS; genes: $(wc -l < "$W/genes_all.txt")"
echo "scoring, chunk budget ${CHUNK_GB:-1.0} GiB  ($(date +%T))"
"$PY" "$D/sparse_score.py" --bed "$FIX/geno" --weights "$WFILE" --out "$OUT" \
    --mode sum --all-genes "$W/genes_all.txt" --channels "$CHANNELS" --block "${BLOCK:-4000}" --threads "$NT" \
    --chunk-gb "${CHUNK_GB:-1.0}" ${CHUNK_COLS:+--chunk-cols "$CHUNK_COLS"} --out-layout chunks
