#!/bin/bash
# =============================================================================
#  variants/ct_sparse_v0/run.sh -- C+T features through the SHARED KERNEL
#
#  Same clumping as ref_ct (the authors' per-gene plink --clump), then the
#  index SNPs become a weight file (lib/make_ct_weights.py: one row per gene
#  x index SNP, channel = P-bin) and lib/sparse_score.py --mode ct computes
#  every gene x threshold average in one sparse matmul + cumsum. Compared
#  with golden/ct this proves the kernel reproduces 12c; the same kernel in
#  --mode sum is the eQTL path.
#
#  Called by run_variant.sh as:  run.sh <fixture_dir> <out_dir>
# =============================================================================
set -euo pipefail
FIX=$1; OUT=$2
W=$OUT/work; mkdir -p "$W/clumps" "$W/chunks"
PY=${BENCH_PY:-python3}; LIB=${BENCH_LIB:?}; NT=${BENCH_THREADS:-4}

# fixture params
eval "$("$PY" - "$FIX/params.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f'CHR={d["chr"]}; R2={d["clump_r2"]}; KB={d["clump_kb"]}; THR="{d["thresholds"]}"; GIANT={d["giant_window_mb"]}')
EOF
)"

# plink 1.9: module on Bouchet, PATH elsewhere
if ! command -v plink >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load PLINK/1.9b_7.11-x86_64 2>/dev/null || true
fi
command -v plink >/dev/null || { echo "plink (1.9) not found"; exit 1; }

# --- 12b: the authors' per-gene clump, same flags as production clump_one
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
echo "clumping $(wc -l < "$W/genes.txt") genes, $NT at a time  ($(date +%T))"
xargs -a "$W/genes.txt" -P "$NT" -I{} bash -c 'clump_one "$1"' _ {}

# clump_manifest.tsv: identical awk to 12_prsnet.sh write_12b_results
awk -v D="$W/clumps" 'BEGIN{OFS="\t"}
     NR==1 {print $0, "n_index", "status"; next}
     { n=0; st="nosnp"
       if ($4>0) { f=D"/"$1".clumped"; st="failed"
           if ((getline l < f) > 0) { st="ok"; while ((getline l < f) > 0) { split(l,a," "); if (length(a)>=3) n++ } }
           close(f) }
       print $0, n, st }' "$FIX/gene_manifest.tsv" > "$W/clump_manifest.tsv"
nfail=$(awk 'NR>1 && $7=="failed"' "$W/clump_manifest.tsv" | wc -l)
echo "clumped: $(awk 'NR>1 && $7=="ok"' "$W/clump_manifest.tsv" | wc -l) ok, $nfail failed  ($(date +%T))"
(( nfail == 0 )) || { awk 'NR>1 && $7=="failed" {print "  failed:", $1}' "$W/clump_manifest.tsv"; exit 1; }

# --- the shared kernel instead of 12c
"$PY" "$LIB/make_ct_weights.py" --clumps "$W/clumps" --manifest "$W/clump_manifest.tsv" \
    --gwas "$FIX/gwas.QC.txt" --pvalue "$FIX/SNP.pvalue" --thresholds "$THR" \
    --out "$W/ct_weights.tsv" --genes-out "$W/genes_all.txt"
"$PY" "$LIB/sparse_score.py" --bed "$FIX/geno" --weights "$W/ct_weights.tsv" --out "$OUT" \
    --mode ct --thresholds "$THR" --all-genes "$W/genes_all.txt" --block "${BLOCK:-4000}" --threads "$NT"
