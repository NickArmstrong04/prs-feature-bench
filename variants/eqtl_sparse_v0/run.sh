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
