#!/bin/bash
# =============================================================================
#  make_fixture.sh -- build the fixed test dataset for prs-feature-bench
#
#  Runs ONCE, on Bouchet (inside a small salloc: 2 CPU / 8 GB is plenty).
#  Takes one chromosome of one finished PRS-Net feature build (default
#  ad/eur chr19, which passed `12_prsnet.sh check`), subsamples the cohort
#  to N people (case/control stratified, seeded) and copies every input the
#  feature scripts need into $BENCH/fixture/. Nothing in the production
#  tree is modified. The fixture is ~150-300 MB and can be scp'd anywhere.
#
#  USAGE
#      DISEASE=ad COHORT=eur CHR=19 N=2000 bash make_fixture.sh
#
#  Reads the production config (ROOT/scripts/slurm/config.sh) only to find
#  paths; the fixture carries its own params.json afterwards, so variants
#  never source config.sh.
#
#  Output layout (the harness contract -- see README.md "Fixture"):
#      fixture/geno.{bed,bim,fam}      N samples x window SNPs, A1 = effect allele
#      fixture/gwas.QC.txt             QC'd sumstats, this chromosome
#      fixture/SNP.pvalue              plink2 --q-score-range input (SNP P)
#      fixture/pheno.tab               FID IID PHENO for the N samples
#      fixture/gene_windows.tsv        windows on this chromosome
#      fixture/gene_manifest.tsv       12a's manifest (gene, ..., n_snps, big)
#      fixture/genes/<gene>.{snplist,assoc}
#      fixture/params.json             thresholds, r2, kb, chr, N, provenance
#      lib/ref_12_prsnet.py            pinned copy of the production scorer
# =============================================================================
set -euo pipefail

DISEASE=${DISEASE:-ad}; COHORT=${COHORT:-eur}; CHR=${CHR:-19}; N=${N:-2000}; SEED=${SEED:-7}
ROOT=${ROOT:-/home/naa79/project_pi_sz622/naa79/prs-benchmark}
BENCH=${BENCH:-$(cd "$(dirname "$0")" && pwd)}
FIX=$BENCH/fixture

export DISEASE COHORT CHR
source "$ROOT/scripts/slurm/config.sh"

for v in PNET_CHR GQC PHENO PRSNET_WINDOWS PRSNET_THRESHOLDS PRSNET_CLUMP_R2 PRSNET_CLUMP_KB PLINK_MODULE; do
    [[ -n "${!v:-}" ]] || { echo "config.sh does not define $v"; exit 1; }
done
BED=$PNET_CHR/target_chr${CHR}.PH
for f in "$BED.bed" "$BED.bim" "$BED.fam" "$GQC/gwas.QC.txt" "$GQC/SNP.pvalue" "$PHENO" \
         "$PRSNET_WINDOWS" "$PNET_CHR/gene_manifest.tsv" "$ROOT/scripts/slurm/12_prsnet.py"; do
    [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
done
[[ -d "$PNET_CHR/genes" ]] || { echo "MISSING: $PNET_CHR/genes (run 12a for chr$CHR)"; exit 1; }

module load "$PLINK_MODULE"
command -v plink >/dev/null || { echo "plink not on PATH after module load $PLINK_MODULE"; exit 1; }

echo "source:   $DISEASE/$COHORT chr$CHR  ($(wc -l < "$BED.fam") samples, $(wc -l < "$BED.bim") window SNPs)"
echo "fixture:  $FIX  (N=$N, seed $SEED)"
mkdir -p "$FIX/genes" "$BENCH/lib"

# --- 1. stratified sample of N people present in both the bed and the phenotype
#     pheno.tab: FID IID PHENO (1=control, 2=case), no header
export LC_ALL=C
join -j 2 <(awk '{print $1, $2}' "$BED.fam" | sort -k2,2) \
          <(awk '{print $1, $2, $3}' "$PHENO" | sort -k2,2) \
    | awk '{print $2, $1, $4}' > "$FIX/.pool.txt"        # FID IID PHENO
n_case=$(awk '$3==2' "$FIX/.pool.txt" | wc -l); n_ctrl=$(awk '$3==1' "$FIX/.pool.txt" | wc -l)
k_case=$(( N * n_case / (n_case + n_ctrl) )); k_ctrl=$(( N - k_case ))
{ awk '$3==2' "$FIX/.pool.txt" | shuf --random-source=<(yes "$SEED") | head -n "$k_case"
  awk '$3==1' "$FIX/.pool.txt" | shuf --random-source=<(yes "$SEED") | head -n "$k_ctrl"; } \
    | sort -k2,2 > "$FIX/pheno.tab"
awk '{print $1, $2}' "$FIX/pheno.tab" > "$FIX/.keep.txt"
rm -f "$FIX/.pool.txt"
echo "sampled:  $(wc -l < "$FIX/pheno.tab") people ($k_case cases / $k_ctrl controls; pool $n_case / $n_ctrl)"

# --- 2. the genotype fixture: same variants, same A1, N people
plink --bfile "$BED" --keep "$FIX/.keep.txt" --make-bed --out "$FIX/geno" --memory 4000 --threads 2 > "$FIX/.plink.log" 2>&1 \
    || { cat "$FIX/.plink.log"; exit 1; }
rm -f "$FIX/.keep.txt" "$FIX/geno.nosex"
cmp -s <(awk '{print $2, $5}' "$BED.bim") <(awk '{print $2, $5}' "$FIX/geno.bim") \
    || { echo "ERROR: fixture .bim A1 differs from the source .bim (plink flipped something)"; exit 1; }
echo "geno:     $(wc -l < "$FIX/geno.bim") variants x $(wc -l < "$FIX/geno.fam") samples, $(du -h "$FIX/geno.bed" | cut -f1)"

# --- 3. sumstats, p-values, windows, manifest, per-gene files
cp "$GQC/gwas.QC.txt" "$FIX/gwas.QC.txt"
cp "$GQC/SNP.pvalue"  "$FIX/SNP.pvalue"
head -n 1 "$PRSNET_WINDOWS" > "$FIX/gene_windows.tsv"
awk -v c="$CHR" 'NR>1 && ($1==c || $1=="chr"c || $2==c || $2=="chr"c)' "$PRSNET_WINDOWS" >> "$FIX/gene_windows.tsv"
cp "$PNET_CHR/gene_manifest.tsv" "$FIX/gene_manifest.tsv"
cp "$PNET_CHR"/genes/*.snplist "$PNET_CHR"/genes/*.assoc "$FIX/genes/" 2>/dev/null
echo "genes:    $(ls "$FIX/genes"/*.snplist | wc -l) snplists, $(awk 'NR>1' "$FIX/gene_windows.tsv" | wc -l) window rows"

# --- 4. pinned reference scorer + params
cp "$ROOT/scripts/slurm/12_prsnet.py" "$BENCH/lib/ref_12_prsnet.py"
cat > "$FIX/params.json" <<EOF
{
  "disease": "$DISEASE", "cohort": "$COHORT", "chr": $CHR,
  "n_samples": $(wc -l < "$FIX/geno.fam"), "n_variants": $(wc -l < "$FIX/geno.bim"),
  "seed": $SEED,
  "thresholds": "$PRSNET_THRESHOLDS",
  "clump_r2": $PRSNET_CLUMP_R2, "clump_kb": $PRSNET_CLUMP_KB,
  "giant_window_mb": ${PRSNET_GIANT_MB:-20},
  "source_bed": "$BED",
  "source_scorer": "$ROOT/scripts/slurm/12_prsnet.py",
  "built": "$(date -Is)"
}
EOF
cat "$FIX/params.json"
echo
echo "fixture size: $(du -sh "$FIX" | cut -f1)"
echo "Next:  bash run_variant.sh ref_ct --as-golden ct     (first golden, ~minutes)"
