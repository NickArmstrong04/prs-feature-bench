#!/bin/bash
# =============================================================================
#  variants/ldpred2_v0/run.sh -- LD-ablation row "ldpred2" (README s12)
#
#  No clumping, no thresholds. For every gene, every window SNP that carries an
#  LDpred2 posterior beta (the fixture's betas.txt, one chosen model column) is
#  scored with that beta: feature(i, g) = sum_{j in W_g} beta_hat_j x_ij, missing
#  calls mean-imputed. One channel per gene (T = 1). The kernel is this
#  directory's copy of ct_sparse_v5's with sum mode given the same fixed
#  gene-major layout and chunk writer ct mode has (--all-genes + --channels),
#  so 12d assembles it unchanged and the trainer sees d_input = 1.
#
#  LDPRED_MODE=ldpred2-unique applies the nearest-window-centre rule as well.
#  Fixture inputs: fixture/betas.txt (SNP CHR A1 <models...>, the 10b `write`
#  output cut to this chromosome) and BETA_COL (default from params.json
#  "ldpred2_col", else g001). There is no golden: the oracle is plink2
#  --score on the same weights (ref_eqtl_plink2's path, --mode sum), which the
#  bench already validated the kernel against to 5e-6.
#
#  Called by run_variant.sh as:  run.sh <fixture_dir> <out_dir>
# =============================================================================
set -euo pipefail
FIX=$1; OUT=$2
W=$OUT/work; mkdir -p "$W"
PY=${BENCH_PY:-python3}; LIB=${BENCH_LIB:?}; NT=${BENCH_THREADS:-4}
D=$(cd "$(dirname "$0")" && pwd)
MODE=${LDPRED_MODE:-ldpred2}

eval "$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(f"CHR={d[\"chr\"]}; BETA_COL_DEFAULT={d.get(\"ldpred2_col\",\"g001\")}")' "$FIX/params.json")"
BETA_COL=${BETA_COL:-$BETA_COL_DEFAULT}
[[ -f "$FIX/betas.txt" ]] || { echo "MISSING: $FIX/betas.txt (LDpred2 10b write output for this chromosome)"; exit 1; }

"$PY" "$LIB/make_ld_weights.py" --mode "$MODE" --windows "$FIX/gene_windows.tsv" --chr "$CHR" \
    --genes-dir "$FIX/genes" --bim "$FIX/geno.bim" --betas "$FIX/betas.txt" --beta-col "$BETA_COL" \
    --out "$W/ld_weights.tsv" --genes-out "$W/genes_all.txt" --report "$W/assign_report.tsv"
echo "scoring [$MODE, $BETA_COL], chunk budget ${CHUNK_GB:-1.0} GiB  ($(date +%T))"
"$PY" "$D/sparse_score.py" --bed "$FIX/geno" --weights "$W/ld_weights.tsv" --out "$OUT" \
    --mode sum --all-genes "$W/genes_all.txt" --channels "$BETA_COL" --block "${BLOCK:-4000}" --threads "$NT" \
    --chunk-gb "${CHUNK_GB:-1.0}" ${CHUNK_COLS:+--chunk-cols "$CHUNK_COLS"} --out-layout "${OUT_LAYOUT:-flat}"
