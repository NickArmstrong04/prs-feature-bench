#!/bin/bash
# =============================================================================
#  variants/clump_np_v0/run.sh -- 12b without plink: one LD pass per chromosome
#  (ld_pairs.py: every pair within clump_kb with r^2 >= clump_r2) and a greedy
#  per-gene index selection over that graph (clump_genes.py). Output: the
#  clumps layout, clumps.tsv (gene, n_index, index SNPs), compared exactly with
#  golden/clumps made by variants/ref_clump (plink per gene).
#  Full-cohort use (one chromosome, inside the salloc):
#      ld_pairs.py --bed <cohort bed> --out pairs.npz --kb 250 --r2 0.5 --threads 4
#      clump_genes.py --pairs pairs.npz --genes-dir $PNET_CHR/genes --manifest $PNET_CHR/gene_manifest.tsv \
#                     --pvalue $GQC/SNP.pvalue --out <clumps dir> --clumps-tsv clumps.tsv --chr $CHR
#  Called by run_variant.sh as:  run.sh <fixture_dir> <out_dir>
# =============================================================================
set -euo pipefail
FIX=$1; OUT=$2
W=$OUT/work; mkdir -p "$W"
PY=${BENCH_PY:-python3}; NT=${BENCH_THREADS:-4}
D=$(cd "$(dirname "$0")" && pwd)
eval "$("$PY" - "$FIX/params.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f'CHR={d["chr"]}; R2={d["clump_r2"]}; KB={d["clump_kb"]}')
EOF
)"
echo "LD pass  ($(date +%T))"
"$PY" "$D/ld_pairs.py" --bed "$FIX/geno" --out "$W/pairs.npz" --kb "$KB" --r2 "$R2" --block "${BLOCK:-512}" --threads "$NT"
echo "clumps  ($(date +%T))"
"$PY" "$D/clump_genes.py" --pairs "$W/pairs.npz" --genes-dir "$FIX/genes" --manifest "$FIX/gene_manifest.tsv" \
    --pvalue "$FIX/SNP.pvalue" --out "$W/clumps" --clumps-tsv "$OUT/clumps.tsv" --chr "$CHR"
