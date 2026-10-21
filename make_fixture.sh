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
FIX=${BENCH_FIXTURE:-$BENCH/fixture}        # a second fixture: BENCH_FIXTURE=$BENCH/fixture_20k N=20000 ...

export DISEASE COHORT CHR
source "$ROOT/scripts/slurm/config.sh"

for v in PNET_CHR GQC PHENO PRSNET_WINDOWS PRSNET_THRESHOLDS PRSNET_CLUMP_R2 PRSNET_CLUMP_KB PLINK_MODULE; do
    [[ -n "${!v:-}" ]] || { echo "config.sh does not define $v"; exit 1; }
done
for f in "$GQC/gwas.QC.txt" "$GQC/SNP.pvalue" "$PHENO" "$PRSNET_WINDOWS" \
         "$PNET_CHR/gene_manifest.tsv" "$PNET_CHR/window.snplist" "$ROOT/scripts/slurm/12_prsnet.py"; do
    [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
done
[[ -d "$PNET_CHR/genes" ]] || { echo "MISSING: $PNET_CHR/genes (run 12a for chr$CHR)"; exit 1; }

# Genotype source. Preferred: 12a's cohort bed (window SNPs, A1 = effect
# allele). It is a regenerable intermediate and may have been deleted; then
# fall back to 12a's own command against the raw release, restricted to the
# fixture's people -- same --extract, same --a1-allele, so the same variants
# and orientation come out. The raw chromosome is streamed once (minutes).
BED=$PNET_CHR/target_chr${CHR}.PH
if [[ -f "$BED.bed" && -f "$BED.bim" && -f "$BED.fam" ]]; then
    MODE=cohort
else
    MODE=raw
    for v in GENO_PREFIX MIS; do [[ -n "${!v:-}" ]] || { echo "config.sh does not define $v (needed for the raw fallback)"; exit 1; }; done
    for f in "$GENO_PREFIX.bed" "$GENO_PREFIX.bim" "$GENO_PREFIX.fam" "$MIS/target_data.a1"; do
        [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
    done
fi

module load "$PLINK_MODULE"
command -v plink >/dev/null || { echo "plink not on PATH after module load $PLINK_MODULE"; exit 1; }

if [[ $MODE == cohort ]]; then
    echo "source:   $DISEASE/$COHORT chr$CHR cohort bed  ($(wc -l < "$BED.fam") samples, $(wc -l < "$BED.bim") window SNPs)"
    FAM=$BED.fam
else
    echo "source:   $DISEASE/$COHORT chr$CHR -- cohort bed absent, building from the RAW release"
    echo "          $GENO_PREFIX  (--keep fixture people, --extract $(wc -l < "$PNET_CHR/window.snplist") window SNPs, --a1-allele)"
    FAM=$GENO_PREFIX.fam
fi
echo "fixture:  $FIX  (N=$N, seed $SEED)"
mkdir -p "$FIX/genes" "$BENCH/lib"

# --- 1. stratified sample of N people present in both the genotypes and the phenotype
#     pheno.tab: FID IID PHENO (1=control, 2=case), no header
export LC_ALL=C
join -j 2 <(awk '{print $1, $2}' "$FAM" | sort -k2,2) \
          <(awk '{print $1, $2, $3}' "$PHENO" | sort -k2,2) \
    | awk '{print $2, $1, $4}' > "$FIX/.pool.txt"        # FID IID PHENO
n_case=$(awk '$3==2' "$FIX/.pool.txt" | wc -l); n_ctrl=$(awk '$3==1' "$FIX/.pool.txt" | wc -l)
k_case=$(( N * n_case / (n_case + n_ctrl) )); k_ctrl=$(( N - k_case ))
{ awk '$3==2' "$FIX/.pool.txt" | shuf -n "$k_case" --random-source=<(yes "$SEED")
  awk '$3==1' "$FIX/.pool.txt" | shuf -n "$k_ctrl" --random-source=<(yes "$SEED"); } \
    | sort -k2,2 > "$FIX/pheno.tab"
awk '{print $1, $2}' "$FIX/pheno.tab" > "$FIX/.keep.txt"
rm -f "$FIX/.pool.txt"
echo "sampled:  $(wc -l < "$FIX/pheno.tab") people ($k_case cases / $k_ctrl controls; pool $n_case / $n_ctrl)"

# --- 2. the genotype fixture: same variants, same A1, N people
if [[ $MODE == cohort ]]; then
    plink --bfile "$BED" --keep "$FIX/.keep.txt" --make-bed --out "$FIX/geno" \
          --memory 4000 --threads 2 > "$FIX/.plink.log" 2>&1 || { cat "$FIX/.plink.log"; exit 1; }
    rm -f "$FIX/.keep.txt" "$FIX/geno.nosex"
    cmp -s <(awk '{print $2, $5}' "$BED.bim") <(awk '{print $2, $5}' "$FIX/geno.bim") \
        || { echo "ERROR: fixture .bim A1 differs from the source .bim (plink flipped something)"; exit 1; }
else
    # 12a's command (12_prsnet.sh step_12a), keep list = the fixture's people
    plink --bfile "$GENO_PREFIX" --keep "$FIX/.keep.txt" --extract "$PNET_CHR/window.snplist" \
          --a1-allele "$MIS/target_data.a1" --make-bed --out "$FIX/geno" \
          --memory 4000 --threads 2 > "$FIX/.plink.log" 2>&1 || { cat "$FIX/.plink.log"; exit 1; }
    rm -f "$FIX/.keep.txt" "$FIX/geno.nosex"
    # orientation check, as 12a does it: .bim A1 (col 5) must equal gwas A1 (col 4) for every variant
    join -1 2 -2 3 <(sort -k2,2 "$FIX/geno.bim") <(tail -n +2 "$GQC/gwas.QC.txt" | sort -k3,3) \
      | awk '{if($5==$9) m++; else x++} END{printf "A1 vs gwas effect allele: %d match, %d mismatch\n", m+0, x+0; exit (x>0)}' \
      || { echo "ERROR: fixture A1 is not the GWAS effect allele for some variants"; exit 1; }
    n_win=$(wc -l < "$PNET_CHR/window.snplist"); n_got=$(wc -l < "$FIX/geno.bim")
    (( n_got == n_win )) || echo "NOTE: $n_got of $n_win window SNPs written (12a's own bed had the same property if 03 dropped variants)"
fi
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
  "source_mode": "$MODE", "source_bed": "$([[ $MODE == cohort ]] && echo "$BED" || echo "$GENO_PREFIX")",
  "source_scorer": "$ROOT/scripts/slurm/12_prsnet.py",
  "built": "$(date -Is)"
}
EOF
cat "$FIX/params.json"
echo
echo "fixture size: $(du -sh "$FIX" | cut -f1)"
echo "Next:  bash run_variant.sh ref_ct --as-golden ct     (first golden, ~minutes)"
