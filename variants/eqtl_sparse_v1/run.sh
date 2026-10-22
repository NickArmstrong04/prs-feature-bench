#!/bin/bash
# variants/eqtl_sparse_v1 -- the eQTL-weighted features through the ct_sparse_v3
# kernel in --mode sum (column chunking, packed-bed decode, row-streamed
# scatter). sparse_score.py here is a verbatim copy of ct_sparse_v3's
# (tests/smoke_test.py asserts that). Compared with golden/eqtl_<set>
# (ref_eqtl_plink2). Knobs: WEIGHTS (fixture/weights/<set>.tsv), CHUNK_GB,
# CHUNK_COLS, BLOCK. The eQTL case has many weights per SNP (one per cell
# type), which is what the kernel's per-nonzero inner loop must be judged on.
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
