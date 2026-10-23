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
#  Second layout (PRSNet-2 dosage features, README s2/s9): samples.txt,
#  snps.tsv, X.npy (int8, 3 = missing), pvalues.npy, gene2snps.{tsv,pkl};
#  detected by X.npy and compared exactly with compare_dosage.py.
#
#  Fidelity rule: a variant PASSES when, on every (gene, channel) column
#  present in the golden, |ours - golden| <= ATOL + RTOL*|golden| for every
#  sample (compare.py). Missing columns FAIL; extra columns are reported.
# =============================================================================
set -euo pipefail

BENCH=$(cd "$(dirname "$0")" && pwd)
FIX=${BENCH_FIXTURE:-$BENCH/fixture}            # another fixture dir (a timing fixture, say)
FIX=$(cd "$FIX" && pwd)
if [[ "$FIX" == "$BENCH/fixture" ]]; then GOLD=$BENCH/golden; else GOLD=$FIX/golden; fi   # goldens belong to a fixture
FIXNAME=$(basename "$FIX")
VAR=${1:-}; shift || true
[[ -n "$VAR" && -f "$BENCH/variants/$VAR/run.sh" ]] || { echo "usage: run_variant.sh <variant> ...  (variants/: $(ls "$BENCH/variants" 2>/dev/null | tr '\n' ' '))"; exit 1; }
[[ -f "$FIX/params.json" ]] || { echo "no fixture -- run make_fixture.sh first"; exit 1; }

GOLDEN=ct; GOLDEN_SET=0; AS_GOLDEN=""; THREADS=${BENCH_THREADS:-4}; TAG=""; KEEP=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --golden)    GOLDEN=$2; GOLDEN_SET=1; shift 2 ;;
        --as-golden) AS_GOLDEN=$2; shift 2 ;;
        --threads)   THREADS=$2; shift 2 ;;
        --tag)       TAG=$2; shift 2 ;;
        --keep)      KEEP=1; shift ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
done

[[ "$FIXNAME" == fixture ]] || TAG="[$FIXNAME] $TAG"
export BENCH_THREADS=$THREADS BENCH_LIB=$BENCH/lib BENCH_PY=${BENCH_PY:-python3}
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$BENCH/runs/${VAR}_$STAMP
mkdir -p "$OUT" "$GOLD"
COMMIT=$(git -C "$BENCH" rev-parse --short HEAD 2>/dev/null || echo nogit)
DIRTY=$(git -C "$BENCH" status --porcelain "variants/$VAR" 2>/dev/null | grep -q . && echo "+dirty" || true)
HOST=$(hostname -s)

echo "variant:  $VAR  ($COMMIT$DIRTY)   threads $THREADS   host $HOST"
echo "fixture:  $FIXNAME  $(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["disease"]+"/"+d["cohort"]+" chr"+str(d["chr"])+"  "+str(d["n_samples"])+" x "+str(d["n_variants"]))' "$FIX/params.json")"
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
# two layouts (README s2): scores (scores.f32, compared with tolerance) or
# dosage (X.npy int8 + sidecars, compared exactly); told apart by X.npy
if [[ -f "$OUT/X.npy" || -f "$OUT/X2.npy" ]]; then
    BULK=X.npy; [[ -f "$OUT/X.npy" ]] || BULK=X2.npy             # int8, or the 2-bit store (lib/x2bit.py)
    LAYOUT=dosage; FILES="samples.txt snps.tsv $BULK pvalues.npy gene2snps.tsv gene2snps.pkl"
    CMP=compare_dosage.py; CMP_ARGS=(); COLF=snps.tsv
elif [[ -f "$OUT/geno.bed" ]]; then                             # third layout: a PLINK fileset (12a's cohort bed, README s11)
    LAYOUT=bed; FILES="geno.bed geno.bim geno.fam"
    CMP=compare_bed.py; CMP_ARGS=(); BULK=geno.bed; COLF=geno.bim
elif [[ -f "$OUT/scores_manifest.tsv" ]]; then                  # fourth: 12c's chunk layout (ct_sparse_v5, README s11)
    LAYOUT=chunks; FILES="samples.txt scores_manifest.tsv gene_counts.tsv"
    CMP=compare.py; CMP_ARGS=(--rtol "${BENCH_RTOL:-1e-5}" --atol "${BENCH_ATOL:-1e-7}"); BULK="scores_chunk*.npy"; COLF=gene_counts.tsv
else
    LAYOUT=scores; FILES="samples.txt columns.tsv scores.f32"
    CMP=compare.py; CMP_ARGS=(--rtol "${BENCH_RTOL:-1e-5}" --atol "${BENCH_ATOL:-1e-7}"); BULK=scores.f32; COLF=columns.tsv
fi
for f in $FILES; do
    [[ -s "$OUT/$f" ]] || { echo "CONTRACT ($LAYOUT layout): $OUT/$f missing or empty"; exit 2; }
done
if [[ "$LAYOUT" == bed ]]; then NCOL=$(wc -l < "$OUT/$COLF"); else NCOL=$(awk 'NR>1' "$OUT/$COLF" | wc -l); fi   # .bim has no header; chunks: genes
echo "done:     wall $WALL   cpu ${CPU_S}s   maxrss ${MAXRSS_MB} MB   columns $NCOL   ($LAYOUT layout)"

# --- compare
VERDICT=NOGOLDEN; DETAIL="-"
if [[ -n "$AS_GOLDEN" && $GOLDEN_SET -eq 0 ]]; then
    echo "making golden/$AS_GOLDEN -- no comparison (pass --golden <name> to compare as well)"
elif [[ -d "$GOLD/$GOLDEN" ]]; then
    set +e
    "$BENCH_PY" "$BENCH/$CMP" "$GOLD/$GOLDEN" "$OUT" "${CMP_ARGS[@]}" | tee "$OUT/compare.txt"
    CRC=${PIPESTATUS[0]}
    set -e
    VERDICT=$([[ $CRC -eq 0 ]] && echo PASS || echo FAIL)
    DETAIL=$(tail -n 1 "$OUT/compare.txt")
else
    echo "no $GOLD/$GOLDEN -- fidelity not checked"
fi

[[ -f "$BENCH/results.tsv" ]] || printf 'stamp\tvariant\tcommit\tthreads\twall\tcpu_s\tmaxrss_mb\tverdict\tdetail\ttag\n' > "$BENCH/results.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$STAMP" "$VAR" "$COMMIT$DIRTY" "$THREADS" "$WALL" "$CPU_S" "$MAXRSS_MB" "$VERDICT" "$DETAIL" "$TAG" >> "$BENCH/results.tsv"
echo "results:  $VERDICT  -> results.tsv"

# --- promote to golden
if [[ -n "$AS_GOLDEN" ]]; then
    [[ -d "$GOLD/$AS_GOLDEN" ]] && { echo "$GOLD/$AS_GOLDEN exists -- remove it by hand if you really mean to replace it"; exit 3; }
    mkdir -p "$GOLD/$AS_GOLDEN"
    for f in $FILES run.log; do cp "$OUT/$f" "$GOLD/$AS_GOLDEN/"; done
    [[ "$LAYOUT" == chunks ]] && cp "$OUT"/scores_chunk*.npy "$OUT"/genes_chunk*.txt "$GOLD/$AS_GOLDEN/"
    printf 'variant\t%s\ncommit\t%s\nstamp\t%s\nthreads\t%s\nwall\t%s\nfixture\t%s\n' "$VAR" "$COMMIT$DIRTY" "$STAMP" "$THREADS" "$WALL" "$FIX" > "$GOLD/$AS_GOLDEN/provenance.tsv"
    echo "stored:   $GOLD/$AS_GOLDEN  (from $VAR @ $COMMIT$DIRTY)"
fi

# --- clean up bulky run output unless asked to keep it
if (( KEEP == 0 )) && [[ "$VERDICT" == PASS ]]; then
    rm -f $OUT/$BULK; rm -rf "$OUT/work"                      # unquoted: BULK may be a glob (chunks layout)
    echo "          ($BULK and work/ removed; --keep to retain)"
fi
