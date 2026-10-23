#!/bin/bash
# =============================================================================
#  variants/dosage_pack_v0/run.sh -- dosage_raw_v0 writing the 2-bit sample-major
#  store X2.npy (lib/x2bit.py) instead of int8 X.npy
#
#  Same selection as dosage_v1 and the same raw-release seek path as
#  dosage_raw_v0; the output matrix is X2.npy, a quarter of the bytes, which
#  compare_dosage.py unpacks against the int8 golden. Genotypes are read from the
#  raw chromosome file named in the fixture's params.json (source_bed, when
#  source_mode is raw; override with RAW_BED=<prefix>) -- read-only, ~3% of
#  the rows 12a streams. The fixture's .fam is the --keep list, its .bim A1 the
#  --a1-allele. Compared with golden/dosage. Knobs: PTHR, BLOCK_MB, SUB, MAX_GAP.
#
#  NOTE: this is the one variant that reads outside the fixture, by design;
#  the fixture names the file and it is only ever read.
#
#  Called by run_variant.sh as:  run.sh <fixture_dir> <out_dir>
# =============================================================================
set -euo pipefail
FIX=$1; OUT=$2
mkdir -p "$OUT/work"
PY=${BENCH_PY:-python3}
D=$(cd "$(dirname "$0")" && pwd)

RAW=${RAW_BED:-$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("source_bed","") if d.get("source_mode")=="raw" else "")' "$FIX/params.json")}
[[ -n "$RAW" ]] || { echo "fixture was not built from the raw release (source_mode != raw); set RAW_BED=<prefix>"; exit 1; }
for e in bed bim fam; do [[ -r "$RAW.$e" ]] || { echo "cannot read $RAW.$e"; exit 1; }; done
echo "raw release: $RAW  ($(du -h "$RAW.bed" | cut -f1) .bed)  P < ${PTHR:-0.01}  ($(date +%T))"
BENCH_LIB=${BENCH_LIB:?} "$PY" "$D/dosage_pack.py" --raw "$RAW" --bed "$FIX/geno" --pvalue "$FIX/SNP.pvalue" --genes-dir "$FIX/genes" \
    --manifest "$FIX/gene_manifest.tsv" --out "$OUT" --pthr "${PTHR:-0.01}" \
    --block-mb "${BLOCK_MB:-256}" --sub "${SUB:-64}" --max-gap "${MAX_GAP:-2}"
