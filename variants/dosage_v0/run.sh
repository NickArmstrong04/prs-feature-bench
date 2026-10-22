#!/bin/bash
# =============================================================================
#  variants/dosage_v0/run.sh -- PRSNet-2 dosage features, first fast version
#
#  No clumping, no scoring: X.npy int8 (n x selected SNPs) of counted-allele
#  copies with 3 = missing, plus pvalues.npy and gene2snps (the layout
#  github.com/lihan97/PRSNet-2 trains from). SNPs: P < PTHR (paper default
#  0.01) and inside at least one gene window (fixture/genes/<gene>.snplist).
#  One Python pass over the packed .bed (this directory's dosage.py).
#  Compared with golden/dosage (ref_dosage: plink --recode A). Knobs: PTHR, BLOCK.
#
#  Called by run_variant.sh as:  run.sh <fixture_dir> <out_dir>
# =============================================================================
set -euo pipefail
FIX=$1; OUT=$2
mkdir -p "$OUT/work"
PY=${BENCH_PY:-python3}
D=$(cd "$(dirname "$0")" && pwd)

echo "dosage features: P < ${PTHR:-0.01}, block ${BLOCK:-4000}  ($(date +%T))"
"$PY" "$D/dosage.py" --bed "$FIX/geno" --pvalue "$FIX/SNP.pvalue" --genes-dir "$FIX/genes" \
    --manifest "$FIX/gene_manifest.tsv" --out "$OUT" --pthr "${PTHR:-0.01}" --block "${BLOCK:-4000}"
