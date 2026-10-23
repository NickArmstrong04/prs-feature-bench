#!/bin/bash
# =============================================================================
#  variants/bed_seek_v1/run.sh -- 12a's cohort .bed by seeking the raw release
#
#  The fixture is a plink-made bed of window SNPs for a subset of the cohort
#  (make_fixture: --keep --extract --a1-allele --make-bed on the raw chromosome
#  named in params.json), so it is itself the golden for this layout
#  (variants/ref_bed copies it into golden/bed). This variant rebuilds it from
#  the raw release: --extract = the fixture .bim ids, --keep = the fixture .fam,
#  counted allele = the fixture .bim A1, and must match byte for byte.
#  Full-cohort use (one chromosome, inside the salloc):
#      bed_seek.py --raw $GENO_PREFIX --extract $PNET_CHR/window.snplist \
#                  --keep $KEEP_LIST --a1 $MIS/target_data.a1 --out <prefix> --threads 4
#  Called by run_variant.sh as:  run.sh <fixture_dir> <out_dir>
# =============================================================================
set -euo pipefail
FIX=$1; OUT=$2
PY=${BENCH_PY:-python3}; NT=${BENCH_THREADS:-4}
D=$(cd "$(dirname "$0")" && pwd)
RAW=${RAW_BED:-$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("source_bed", ""))' "$FIX/params.json")}
[[ -n "$RAW" && -f "$RAW.bed" ]] || { echo "raw release bed not found: '$RAW' (params.json source_bed, or RAW_BED=...)"; exit 1; }
mkdir -p "$OUT/work"
awk '{print $2}' "$FIX/geno.bim" > "$OUT/work/extract.txt"
"$PY" "$D/bed_seek.py" --raw "$RAW" --extract "$OUT/work/extract.txt" --keep "$FIX/geno.fam" \
    --a1-from-bim "$FIX/geno" --out "$OUT/geno" --threads "$NT" --block "${BLOCK:-256}" --max-gap "${MAX_GAP:-64}" \
    --unassignable "${UNASSIGNABLE:-minor}"
