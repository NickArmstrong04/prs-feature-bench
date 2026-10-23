#!/bin/bash
# =============================================================================
#  variants/ref_bed/run.sh -- reference for the bed layout: the fixture's own
#  plink-made geno.{bed,bim,fam} (make_fixture ran 12a's plink command on the
#  raw chromosome for the fixture people). Run once with --as-golden bed.
#  Called by run_variant.sh as:  run.sh <fixture_dir> <out_dir>
# =============================================================================
set -euo pipefail
FIX=$1; OUT=$2
for e in bed bim fam; do cp "$FIX/geno.$e" "$OUT/geno.$e"; done
echo "ref_bed: copied $FIX/geno.{bed,bim,fam} ($(wc -l < "$FIX/geno.bim") variants x $(wc -l < "$FIX/geno.fam") people)"
