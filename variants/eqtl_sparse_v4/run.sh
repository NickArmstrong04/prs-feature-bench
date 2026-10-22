#!/bin/bash
# variants/eqtl_sparse_v4 -- eqtl_sparse_v3 with a SAMPLE-MAJOR chunk result:
# each gene's GEMM writes G_g^T @ W_g into S[:, cols] (n x c), so the float32
# memmap write is a contiguous copy instead of a 1 GiB transpose per chunk.
# Columns come out gene-major (order is free under the contract). Compared
# with golden/eqtl_<set> (ref_eqtl_plink2). Knobs: WEIGHTS, CHUNK_GB,
# CHUNK_COLS, BLOCK.
set -euo pipefail
FIX=$1; OUT=$2
PY=${BENCH_PY:-python3}; NT=${BENCH_THREADS:-4}
WEIGHTS=${WEIGHTS:-onek1k}
D=$(cd "$(dirname "$0")" && pwd)
[[ -f "$FIX/weights/$WEIGHTS.tsv" ]] || { echo "no weight set: $FIX/weights/$WEIGHTS.tsv"; exit 1; }
echo "scoring $WEIGHTS, chunk budget ${CHUNK_GB:-1.0} GiB  ($(date +%T))"
"$PY" "$D/sparse_score.py" --bed "$FIX/geno" --weights "$FIX/weights/$WEIGHTS.tsv" --out "$OUT" \
      --mode sum --block "${BLOCK:-4000}" --threads "$NT" \
      --chunk-gb "${CHUNK_GB:-1.0}" ${CHUNK_COLS:+--chunk-cols "$CHUNK_COLS"}
