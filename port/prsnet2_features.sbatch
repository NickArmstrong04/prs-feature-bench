#!/bin/bash
# =============================================================================
#  port/prsnet2_features.sbatch -- PRSNet-2 input features for one disease/cohort,
#  one Slurm array task per chromosome, straight from the raw release.
#
#  DRAFT (2026-10-23): written from the bench's dosage_pack_v0 and the 12a
#  excerpt of 12_prsnet.sh; NOT yet run on Bouchet. Variable names follow
#  $ROOT/scripts/slurm/config.sh as 12a uses them (GENO_PREFIX, GQC, TARGET,
#  MIS, PRSNET_WINDOWS, KEEP_LIST, PRSNET_PY, PRSNET_SCRIPT, PRSNET_BIG_SNPS).
#
#      sbatch --export=DISEASE=ad,COHORT=eur port/prsnet2_features.sbatch
#
#  Per task (chromosome $SLURM_ARRAY_TASK_ID):
#    1. 12a's `snps` sub-step: per-gene snplists + gene_manifest.tsv from the
#       windows, gwas.QC.txt and step 04's ${TARGET}.PH.bim (giant-window cap)
#    2. dosage_pack.py: P < PTHR SNPs inside gene windows, seeked out of the
#       raw chromosome, --keep KEEP_LIST, counted allele from target_data.a1,
#       written as the 2-bit sample-major shard $OUT/chr$CHR/X2.npy (+ sidecars)
#  Afterwards (one small job, or the training job's prologue):
#    3. port/merge_shards.py $OUT -> genome-wide snps.tsv, pvalues.npy,
#       gene2snps.{tsv,pkl} with column offsets, shards.tsv for x2bit.Sharded
#
#  Resources: the chr19 bench run took ~6 s and 1 GB at 100k people; at
#  381k people and the largest chromosomes expect ~1-3 min and <8 GB per task,
#  dominated by ~3,000-40,000 seeks of 122 KB into the raw .bed.
# =============================================================================
#SBATCH --job-name=prsnet2_feat
#SBATCH --array=1-22
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=0:30:00
#SBATCH --output=logs/prsnet2_feat_%A_%a.out
#SBATCH --error=logs/prsnet2_feat_%A_%a.err
set -euo pipefail

: "${DISEASE:?}" "${COHORT:?}"
export DISEASE COHORT CHR=${CHR:-$SLURM_ARRAY_TASK_ID}
ROOT=${ROOT:-/home/naa79/project_pi_sz622/naa79/prs-benchmark}
BENCH=${BENCH:-/home/naa79/project_pi_sz622/naa79/prs-feature-bench}
source "$ROOT/scripts/slurm/config.sh"
PTHR=${PTHR:-0.01}
OUT=${OUT:-$ROOT/data/processed/$DISEASE/$COHORT/prsnet2}
mkdir -p "$OUT/chr$CHR/genes" logs

for v in GENO_PREFIX GQC TARGET MIS PRSNET_WINDOWS KEEP_LIST PRSNET_PY PRSNET_SCRIPT; do
    [[ -n "${!v:-}" ]] || { echo "config.sh does not define $v"; exit 1; }
done
for f in "$GENO_PREFIX.bed" "$GENO_PREFIX.bim" "$GENO_PREFIX.fam" "$GQC/gwas.QC.txt" "$GQC/SNP.pvalue" \
         "${TARGET}.PH.bim" "$MIS/target_data.a1" "$PRSNET_WINDOWS" "$KEEP_LIST"; do
    [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
done
echo "== prsnet2 features $DISEASE/$COHORT chr$CHR  P < $PTHR  ($(date))"
echo "raw:   $GENO_PREFIX"; echo "keep:  $KEEP_LIST ($(wc -l < "$KEEP_LIST") people)"; echo "out:   $OUT/chr$CHR"

# 1. per-gene SNP lists, exactly as 12a makes them
"$PRSNET_PY" "$PRSNET_SCRIPT" snps --chr "$CHR" --windows "$PRSNET_WINDOWS" \
    --gwas "$GQC/gwas.QC.txt" --bim "${TARGET}.PH.bim" \
    --out-dir "$OUT/chr$CHR/genes" --manifest "$OUT/chr$CHR/gene_manifest.tsv" \
    --union "$OUT/chr$CHR/window.snplist" ${PRSNET_BIG_SNPS:+--big-snps "$PRSNET_BIG_SNPS"}

# 2. the shard
BENCH_LIB=$BENCH/lib "$PRSNET_PY" "$BENCH/variants/dosage_pack_v0/dosage_pack.py" \
    --raw "$GENO_PREFIX" --bed "${TARGET}.PH" --keep "$KEEP_LIST" --a1 "$MIS/target_data.a1" \
    --pvalue "$GQC/SNP.pvalue" --genes-dir "$OUT/chr$CHR/genes" --manifest "$OUT/chr$CHR/gene_manifest.tsv" \
    --out "$OUT/chr$CHR" --pthr "$PTHR" --block-mb "${BLOCK_MB:-512}" --sub "${SUB:-64}" --max-gap "${MAX_GAP:-2}"
echo "== done chr$CHR  ($(date))"
