#!/bin/bash
# =============================================================================
#  variants/ref_dosage/run.sh -- the REFERENCE for PRSNet-2 dosage features
#
#  X[i, j] = copies of .bim A1 (the GWAS effect allele) of SNP j in person i,
#  0/1/2, 3 = missing; columns = SNPs with GWAS P < PTHR that lie in at least
#  one gene window (fixture/genes/<gene>.snplist); gene2snps = the links.
#  Selection is done in awk, the genotypes come from plink 1.9 `--recode A`
#  (which counts A1 and writes NA for missing), converted by
#  lib/dosage_from_raw.py. Slow and exact; it is what golden/dosage is made
#  from. Do not optimise this file -- copy it to variants/<new>/.
#
#  Called by run_variant.sh as:  run.sh <fixture_dir> <out_dir>
# =============================================================================
set -euo pipefail
FIX=$1; OUT=$2
W=$OUT/work; mkdir -p "$W"
PY=${BENCH_PY:-python3}; LIB=${BENCH_LIB:?}; NT=${BENCH_THREADS:-4}
PTHR=${PTHR:-0.01}

if ! command -v plink >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load PLINK/1.9b_7.11-x86_64 2>/dev/null || true
fi
command -v plink >/dev/null || { echo "plink (1.9) not found"; exit 1; }

# --- selection: P < PTHR (numeric P only; header skipped), in the .bim
awk -v t="$PTHR" '$2 ~ /^[0-9.eE+-]+$/ && ($2+0) < (t+0) {print $1}' "$FIX/SNP.pvalue" | sort -u > "$W/sig.txt"
awk 'NR==FNR {s[$1]; next} ($2 in s) {print $2}' "$W/sig.txt" "$FIX/geno.bim" > "$W/sig_bim.txt"   # .bim order
echo "P < $PTHR: $(wc -l < "$W/sig.txt") SNPs, $(wc -l < "$W/sig_bim.txt") of them in the .bim  ($(date +%T))"

# gene-SNP links: manifest genes with n_snps > 0, each snplist intersected with the significant set
awk -F'\t' 'NR==1 {c=4; for (i=1; i<=NF; i++) if ($i=="n_snps") c=i; next} $c>0 {print $1}' "$FIX/gene_manifest.tsv" > "$W/genes.txt"
awk 'FILENAME==sigf {s[$1]; next}
     FNR==1 {g=FILENAME; sub(/.*\//,"",g); sub(/\.snplist$/,"",g)}
     ($1 in s) {print g "\t" $1}' sigf="$W/sig_bim.txt" "$W/sig_bim.txt" $(sed "s|^|$FIX/genes/|; s|$|.snplist|" "$W/genes.txt") > "$W/pairs.tsv"
awk 'NR==FNR {u[$2]; next} ($2 in u) {print $2}' "$W/pairs.tsv" "$FIX/geno.bim" > "$W/sel.snplist"
echo "linked: $(cut -f1 "$W/pairs.tsv" | sort -u | wc -l) genes, $(wc -l < "$W/pairs.tsv") gene-SNP links, $(wc -l < "$W/sel.snplist") SNPs selected  ($(date +%T))"
[[ -s "$W/sel.snplist" ]] || { echo "no SNPs selected"; exit 1; }

# --- genotypes as text: one column per SNP, counts of the .bim A1, NA = missing.
# plink 1.9 re-labels A1 as the minor allele when it writes a new fileset
# (first Bouchet run: "column rs72984410 counts C, .bim A1 is T"), so the
# counted allele is given explicitly (--recode-allele: SNP allele per line)
# and the .bim order kept; dosage_from_raw.py still checks every column.
awk '{print $2, $5}' "$FIX/geno.bim" > "$W/a1.txt"
plink --bfile "$FIX/geno" --extract "$W/sel.snplist" --keep-allele-order \
      --recode A --recode-allele "$W/a1.txt" --out "$W/raw" \
      --memory 8000 --threads "$NT" > "$W/plink.log" 2>&1 || { tail -n 20 "$W/plink.log"; exit 1; }
echo "plink --recode A done  ($(date +%T))"

"$PY" "$LIB/dosage_from_raw.py" --raw "$W/raw.raw" --bim "$FIX/geno.bim" --pvalue "$FIX/SNP.pvalue" \
    --pairs "$W/pairs.tsv" --out "$OUT"
rm -f "$W/raw.raw"
