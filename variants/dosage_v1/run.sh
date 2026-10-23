#!/bin/bash
# =============================================================================
#  variants/dosage_v1/run.sh -- dosage_v0 with the .bed read in gap-split spans
#
#  No clumping, no scoring: X.npy int8 (n x selected SNPs) of counted-allele
#  copies with 3 = missing, plus pvalues.npy and gene2snps (the layout
#  github.com/lihan97/PRSNet-2 trains from). SNPs: P < PTHR (paper default
#  0.01) and inside at least one gene window (fixture/genes/<gene>.snplist).
#  One Python pass over the packed .bed (this directory's dosage.py); each
#  block is read as runs of selected rows (MAX_GAP), so RSS is bounded by the
#  decode buffer, not by the chromosome. Compared with golden/dosage
#  (ref_dosage: plink --recode A). Knobs: PTHR, BLOCK, MAX_GAP.
#
#  Called by run_variant.sh as:  run.sh <fixture_dir> <out_dir>
# =============================================================================
set -euo pipefail
FIX=$1; OUT=$2
mkdir -p "$OUT/work"
PY=${BENCH_PY:-python3}
D=$(cd "$(dirname "$0")" && pwd)

echo "dosage features: P < ${PTHR:-0.01}, block ${BLOCK:-4000}, max gap ${MAX_GAP:-256} rows  ($(date +%T))"
"$PY" "$D/dosage.py" --bed "$FIX/geno" --pvalue "$FIX/SNP.pvalue" --genes-dir "$FIX/genes" \
    --manifest "$FIX/gene_manifest.tsv" --out "$OUT" --pthr "${PTHR:-0.01}" --block "${BLOCK:-4000}" --max-gap "${MAX_GAP:-256}"
