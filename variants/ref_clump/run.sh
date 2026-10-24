#!/bin/bash
# =============================================================================
#  variants/ref_clump/run.sh -- reference for the clumps layout: 12b's per-gene
#  plink --clump on the fixture (the same loop ct_sparse_v3 runs), written as
#  clumps.tsv (gene, n_index, index SNPs). Run once with --as-golden clumps.
#  Called by run_variant.sh as:  run.sh <fixture_dir> <out_dir>
# =============================================================================
set -euo pipefail
FIX=$1; OUT=$2
W=$OUT/work; mkdir -p "$W/clumps"
PY=${BENCH_PY:-python3}; LIB=${BENCH_LIB:?}; NT=${BENCH_THREADS:-4}
eval "$("$PY" - "$FIX/params.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f'CHR={d["chr"]}; R2={d["clump_r2"]}; KB={d["clump_kb"]}')
EOF
)"
if ! command -v plink >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load PLINK/1.9b_7.11-x86_64 2>/dev/null || true
fi
command -v plink >/dev/null || { echo "plink (1.9) not found"; exit 1; }
clump_one() {
    local g=$1
    plink --bfile "$FIX/geno" --extract "$FIX/genes/$g.snplist" --chr "$CHR" \
          --clump-p1 1 --clump-r2 "$R2" --clump-kb "$KB" \
          --clump "$FIX/genes/$g.assoc" "$FIX/gwas.QC.txt" --clump-index-first \
          --clump-snp-field SNP --clump-field P \
          --out "$W/clumps/$g" --memory 1500 --threads 1 > /dev/null 2>&1 || true
}
export -f clump_one; export FIX W CHR R2 KB
awk 'NR>1 && $4>0 {print $1}' "$FIX/gene_manifest.tsv" > "$W/genes.txt"
echo "clumping $(wc -l < "$W/genes.txt") genes with plink, $NT at a time  ($(date +%T))"
xargs -a "$W/genes.txt" -P "$NT" -I{} bash -c 'clump_one "$1"' _ {}
awk -v D="$W/clumps" 'BEGIN{OFS="\t"}
     NR==1 {print $0, "n_index", "status"; next}
     { n=0; st="nosnp"
       if ($4>0) { f=D"/"$1".clumped"; st="failed"
           if ((getline l < f) > 0) { st="ok"; while ((getline l < f) > 0) { split(l,a," "); if (length(a)>=3) n++ } }
           close(f) }
       print $0, n, st }' "$FIX/gene_manifest.tsv" > "$W/clump_manifest.tsv"
nfail=$(awk 'NR>1 && $NF=="failed"' "$W/clump_manifest.tsv" | wc -l)
(( nfail == 0 )) || { awk 'NR>1 && $NF=="failed" {print "  failed:", $1}' "$W/clump_manifest.tsv"; exit 1; }
"$PY" "$LIB/clumped_to_tsv.py" "$W/clumps" "$W/clump_manifest.tsv" "$OUT/clumps.tsv"
echo "ref_clump done  ($(date +%T))"
