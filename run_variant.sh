#!/bin/bash
# =============================================================================
#  run_variant.sh -- run one variant of the feature scripts on the fixture,
#  time it, compare it with a golden, append one row to results.tsv
#
#  USAGE
#      bash run_variant.sh <variant> [--golden <name>] [--as-golden <name>]
#                                    [--threads N] [--tag text] [--keep]
#
#      <variant>        a directory variants/<variant>/ containing run.sh
#      --golden ct      compare with golden/ct (default: golden/ct)
#      --as-golden X    after a successful run, store the output as golden/X
#                       (refuses to overwrite an existing golden)
#      --threads N      BENCH_THREADS for the variant (default 4). Recorded.
#      --tag text       free text for the results row (what changed)
#      --keep           keep runs/<variant>_<stamp>/ even on PASS
#
#  THE VARIANT CONTRACT
#      variants/<v>/run.sh  <fixture_dir>  <out_dir>
#  must leave in <out_dir> (nothing else is read):
#      samples.txt     FID<TAB>IID, one per row, row order of scores.f32
#      columns.tsv     header: col gene channel n_snps
#                      one row per column of scores.f32; channel is a label
#                      such as P<=1e-5 (C+T) or a cell type (eQTL)
#      scores.f32      float32, C order, shape (n_samples, n_columns)
#  It gets BENCH_THREADS, BENCH_LIB (lib/) and BENCH_PY (python) in the
#  environment and should use them. It must not write outside <out_dir>.
#
#  Fidelity rule: a variant PASSES when, on every (gene, channel) column
#  present in the golden, |ours - golden| <= ATOL + RTOL*|golden| for every
#  sample (compare.py). Missing columns FAIL; extra columns are reported.
# =============================================================================
set -euo pipefail

BENCH=$(cd "$(dirname "$0")" && pwd)
FIX=$BENCH/fixture
VAR=${1:-}; shift || true
[[ -n "$VAR" && -f "$BENCH/variants/$VAR/run.sh" ]] || { echo "usage: run_variant.sh <variant> ...  (variants/: $(ls "$BENCH/variants" 2>/dev/null | tr '\n' ' '))"; exit 1; }
[[ -f "$FIX/params.json" ]] || { echo "no fixture -- run make_fixture.sh first"; exit 1; }

GOLDEN=ct; AS_GOLDEN=""; THREADS=${BENCH_THREADS:-4}; TAG=""; KEEP=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --golden)    GOLDEN=$2; shift 2 ;;
        --as-golden) AS_GOLDEN=$2; shift 2 ;;
        --threads)   THREADS=$2; shift 2 ;;
        --tag)       TAG=$2; shift 2 ;;
        --keep)      KEEP=1; shift ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
done

export BENCH_THREADS=$THREADS BENCH_LIB=$BENCH/lib BENCH_PY=${BENCH_PY:-python3}
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$BENCH/runs/${VAR}_$STAMP
mkdir -p "$OUT" "$BENCH/golden"
COMMIT=$(git -C "$BENCH" rev-parse --short HEAD 2>/dev/null || echo nogit)
DIRTY=$(git -C "$BENCH" status --porcelain "variants/$VAR" 2>/dev/null | grep -q . && echo "+dirty" || true)
HOST=$(hostname -s)

echo "variant:  $VAR  ($COMMIT$DIRTY)   threads $THREADS   host $HOST"
echo "fixture:  $(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(f"{d[\"disease\"]}/{d[\"cohort\"]} chr{d[\"chr\"]}  {d[\"n_samples\"]} x {d[\"n_variants\"]}")' "$FIX/params.json")"
echo "out:      $OUT"

# --- run, timed; /usr/bin/time -v gives wall time + MaxRSS (KB)
TIMEF=$OUT/.time.txt
set +e
/usr/bin/time -v -o "$TIMEF" bash "$BENCH/variants/$VAR/run.sh" "$FIX" "$OUT" > "$OUT/run.log" 2>&1
RC=$?
set -e
WALL=$(awk -F': ' '/Elapsed \(wall clock\)/{print $2}' "$TIMEF")
MAXRSS_MB=$(awk -F': ' '/Maximum resident set size/{printf "%d", $2/1024}' "$TIMEF")
CPU_S=$(awk -F': ' '/User time/{u=$2} /System time/{s=$2} END{printf "%.0f", u+s}' "$TIMEF")
if (( RC != 0 )); then
    echo "RUN FAILED (rc $RC) -- tail of $OUT/run.log:"; tail -n 20 "$OUT/run.log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$STAMP" "$VAR" "$COMMIT$DIRTY" "$THREADS" "$WALL" "$CPU_S" "$MAXRSS_MB" "RUNFAIL" "-" "$TAG" >> "$BENCH/results.tsv"
    exit "$RC"
fi
for f in samples.txt columns.tsv scores.f32; do
    [[ -s "$OUT/$f" ]] || { echo "CONTRACT: $OUT/$f missing or empty"; exit 2; }
done
NCOL=$(awk 'NR>1' "$OUT/columns.tsv" | wc -l)
echo "done:     wall $WALL   cpu ${CPU_S}s   maxrss ${MAXRSS_MB} MB   columns $NCOL"

# --- compare
VERDICT=NOGOLDEN; DETAIL="-"
if [[ -d "$BENCH/golden/$GOLDEN" ]]; then
    set +e
    "$BENCH_PY" "$BENCH/compare.py" "$BENCH/golden/$GOLDEN" "$OUT" --rtol "${BENCH_RTOL:-1e-5}" --atol "${BENCH_ATOL:-1e-7}" | tee "$OUT/compare.txt"
    CRC=${PIPESTATUS[0]}
    set -e
    VERDICT=$([[ $CRC -eq 0 ]] && echo PASS || echo FAIL)
    DETAIL=$(tail -n 1 "$OUT/compare.txt")
else
    echo "no golden/$GOLDEN -- fidelity not checked"
fi

[[ -f "$BENCH/results.tsv" ]] || printf 'stamp\tvariant\tcommit\tthreads\twall\tcpu_s\tmaxrss_mb\tverdict\tdetail\ttag\n' > "$BENCH/results.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$STAMP" "$VAR" "$COMMIT$DIRTY" "$THREADS" "$WALL" "$CPU_S" "$MAXRSS_MB" "$VERDICT" "$DETAIL" "$TAG" >> "$BENCH/results.tsv"
echo "results:  $VERDICT  -> results.tsv"

# --- promote to golden
if [[ -n "$AS_GOLDEN" ]]; then
    [[ -d "$BENCH/golden/$AS_GOLDEN" ]] && { echo "golden/$AS_GOLDEN exists -- remove it by hand if you really mean to replace it"; exit 3; }
    mkdir -p "$BENCH/golden/$AS_GOLDEN"
    cp "$OUT"/{samples.txt,columns.tsv,scores.f32,run.log} "$BENCH/golden/$AS_GOLDEN/"
    printf 'variant\t%s\ncommit\t%s\nstamp\t%s\nthreads\t%s\nwall\t%s\n' "$VAR" "$COMMIT$DIRTY" "$STAMP" "$THREADS" "$WALL" > "$BENCH/golden/$AS_GOLDEN/provenance.tsv"
    echo "stored:   golden/$AS_GOLDEN  (from $VAR @ $COMMIT$DIRTY)"
fi

# --- clean up bulky run output unless asked to keep it
if (( KEEP == 0 )) && [[ "$VERDICT" == PASS ]]; then
    rm -f "$OUT/scores.f32"; rm -rf "$OUT/work"
    echo "          (scores.f32 and work/ removed; --keep to retain)"
fi
