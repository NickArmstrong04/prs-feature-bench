#!/bin/bash
# =============================================================================
#  port/prsnet_chain.sbatch -- the PRS-Net feature chain (12a -> 12b -> 12c) for
#  one disease/cohort as ONE Slurm array, one task per chromosome, every stage
#  the bench's verified replacement where one exists (README s11):
#
#      12a  cohort bed      variants/bed_seek_v1/bed_seek.py   byte-identical to plink --make-bed
#      12b  per-gene clump  production's own plink loop, all genes incl. the giants,
#                           $SLURM_CPUS_PER_TASK at a time, largest gene first (no 12bg jobs)
#      12c  scores          variants/ct_sparse_v5/sparse_score.py, 12c's chunk layout,
#                           bit-identical to production feats.f32 on chr19
#      then the chromosome's bed is deleted (52 GB for chr1; the file system is 99 % full)
#      12d  assemble        production's 12_prsnet.py assemble, unchanged, after all tasks
#
#  DRAFT (2026-10-23): assembled from the chr19 full-cohort runs in README s11;
#  NOT yet run under sbatch. Variable names follow $ROOT/scripts/slurm/config.sh.
#
#  Submit (from $BENCH, after `mkdir -p logs`):
#      bash port/submit_chain.sh ad eur [--dry-run]       # 22 jobs, each sized for its chromosome
#      sbatch --export=ALL,DISEASE=ad,COHORT=eur --array=19 port/prsnet_chain.sbatch  # one chromosome
#      sbatch --export=ALL,DISEASE=ad,COHORT=eur,STEP=12d --array=1 --cpus-per-task=4 --mem=32G \
#             --time=4:00:00 port/prsnet_chain.sbatch                                 # after all 22
#  Inside a salloc (no array): CHR=19 DISEASE=ad COHORT=eur bash port/prsnet_chain.sbatch
#  Verify one chromosome against production (clumps exact, scores vs a golden dir):
#      CHR=19 DISEASE=ad COHORT=eur CHAIN_GOLDEN=runs/ct_full/golden_chr19 bash port/prsnet_chain.sbatch verify
#
#  Knobs (env): CHAIN_OUT (scratch run dir; default below), CHAIN_KEEP_BED=1 (do not
#  delete the bed), CHAIN_MAX_GAP (64), CHAIN_CHUNK_GB (1.0), CHAIN_FORCE=1 (redo stages
#  whose outputs exist; the default resumes).
#
#  Resources (see submit_chain.sh for the model): chr19 on 4 CPUs took 67 s (bed) + 475 s (clump) +
#  361 s (score) = 905 s and ~15 GB of scratch while the bed existed. chr1 is 3.5x
#  the SNPs and holds 8 of the 30 giant genes (300k-469k window SNPs each, 7-30 min
#  of one plink each in production), so its clump stage bounds the whole run.
# =============================================================================
#  Submit through port/submit_chain.sh: it sizes each chromosome's CPUs, memory
#  and time from its gene manifest and overrides the defaults below, which are
#  for a chr19-sized chromosome only.
#SBATCH --job-name=prsnet_chain
#SBATCH --array=19
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:45:00
#SBATCH --output=logs/prsnet_chain_%A_%a.out
#SBATCH --error=logs/prsnet_chain_%A_%a.err
set -euo pipefail

STEP=${1:-${STEP:-chr}}
: "${DISEASE:?}" "${COHORT:?}"
if [[ "$STEP" == chr || "$STEP" == verify ]]; then
    CHR=${CHR:-${SLURM_ARRAY_TASK_ID:?set CHR=<n> or run as an array}}
else
    CHR=${CHR:-1}                       # 12d is genome-wide; config.sh still wants a CHR
fi
export DISEASE COHORT CHR
ROOT=${ROOT:-/home/naa79/project_pi_sz622/naa79/prs-benchmark}
BENCH=${BENCH:-/home/naa79/project_pi_sz622/naa79/prs-feature-bench}
source "$ROOT/scripts/slurm/config.sh"
PRSNET_PY=${PRSNET_PY:-$PRSNET_VENV/bin/python}
PRSNET_SCRIPT=${PRSNET_SCRIPT:-$ROOT/scripts/slurm/12_prsnet.py}
[[ -x "$PRSNET_PY" ]] || { echo "MISSING: $PRSNET_PY (12_prsnet.sh setup makes the venv)"; exit 1; }
for v in GENO_PREFIX GQC MIS KEEP_LIST PNET PNET_CHR PRSNET_THRESHOLDS PRSNET_CLUMP_R2 PRSNET_CLUMP_KB PLINK_MODULE PRSNET_GENE_ORDER PRSNET_N_GENES; do
    [[ -n "${!v:-}" ]] || { echo "config.sh does not define $v"; exit 1; }
done
CPUS=${SLURM_CPUS_PER_TASK:-4}
MEM_MB=${SLURM_MEM_PER_NODE:-16000}
export OMP_NUM_THREADS=$CPUS OPENBLAS_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS
export BENCH_LIB=$BENCH/lib
OUT=${CHAIN_OUT:-/home/naa79/scratch_pi_sz622/naa79/prs-feature-bench/prsnet_chain/${DISEASE}_${COHORT}}
C=$OUT/chr$CHR
BED=$C/target_chr${CHR}.PH
FORCE=${CHAIN_FORCE:-0}
t0=$(date +%s); tlast=$t0
lap() { local now; now=$(date +%s); printf '== %-28s %6d s  (stage)  %6d s (total)  %s\n' "$1" $((now - tlast)) $((now - t0)) "$(date +%T)"; tlast=$now; }
need_module() {
    if ! command -v module >/dev/null 2>&1; then
        for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
            [[ -f "$f" ]] && { source "$f"; break; }
        done
    fi
    module load "$PLINK_MODULE" 2>/dev/null || module load "$PLINK_MODULE"
    command -v plink >/dev/null || { echo "plink (1.9) not found after module load $PLINK_MODULE"; exit 1; }
}

# -----------------------------------------------------------------------------
#  12d: production's assembler over this run dir (after all 22 tasks)
# -----------------------------------------------------------------------------
if [[ "$STEP" == 12d ]]; then
    for c in $(seq 1 22); do
        [[ -f "$OUT/chr$c/scores/scores_manifest.tsv" ]] || { echo "chr$c has no scores yet: $OUT/chr$c/scores/scores_manifest.tsv"; exit 1; }
    done
    [[ -n "${PHENO:-}" && -f "$PHENO" ]] || { echo "no phenotype for $DISEASE/$COHORT (PHENO unset or missing)"; exit 1; }
    [[ -f "$PRSNET_GENE_ORDER" ]] || { echo "MISSING: $PRSNET_GENE_ORDER"; exit 1; }
    ANC=UNK; [[ "$COHORT" == eur || "$COHORT" == caucasian ]] && ANC=EUR
    G=$OUT/genome/${DISEASE}_${COHORT}; mkdir -p "$G"
    echo "== 12d assemble $DISEASE/$COHORT  run-dir $OUT  ->  $G  ($(date))"
    "$PRSNET_PY" "$PRSNET_SCRIPT" assemble --run-dir "$OUT" --gene-order "$PRSNET_GENE_ORDER" \
        --pheno "$PHENO" --ancestry "$ANC" --out-dir "$G" --thresholds "$PRSNET_THRESHOLDS" \
        --export "${PRSNET_EXPORT:-memmap}" --expect-n-genes "$PRSNET_N_GENES" --sample-block 5000
    lap "12d assemble"
    ls -la "$G"
    exit 0
fi

# -----------------------------------------------------------------------------
#  verify: this chromosome's clumps against production's .clumped (exact index
#  sets) and its scores against a golden dir (CHAIN_GOLDEN, scores or chunk layout)
# -----------------------------------------------------------------------------
if [[ "$STEP" == verify ]]; then
    mkdir -p "$C/verify"
    if [[ -d "$PNET_CHR/clumps" && -f "$PNET_CHR/clump_manifest.tsv" ]]; then
        "$PRSNET_PY" "$BENCH/lib/clumped_to_tsv.py" "$PNET_CHR/clumps" "$PNET_CHR/clump_manifest.tsv" "$C/verify/prod_clumps.tsv"
        "$PRSNET_PY" "$BENCH/lib/clumped_to_tsv.py" "$C/clumps" "$C/clump_manifest.tsv" "$C/verify/clumps.tsv"
        "$PRSNET_PY" "$BENCH/compare_clumps.py" "$C/verify/prod_clumps.tsv" "$C/verify/clumps.tsv" | tee "$C/verify/compare_clumps.txt"
    else
        echo "no production clumps for chr$CHR at $PNET_CHR/clumps -- clump check skipped"
    fi
    if [[ -n "${CHAIN_GOLDEN:-}" ]]; then
        "$PRSNET_PY" "$BENCH/compare.py" "$CHAIN_GOLDEN" "$C/scores" --rtol 1e-5 --atol 1e-7 | tee "$C/verify/compare_scores.txt"
    else
        echo "CHAIN_GOLDEN not set -- score check skipped (e.g. CHAIN_GOLDEN=runs/ct_full/golden_chr19 for chr19)"
    fi
    exit 0
fi

# -----------------------------------------------------------------------------
#  one chromosome
# -----------------------------------------------------------------------------
mkdir -p "$C/clumps" "$C/scores"
echo "== prsnet chain $DISEASE/$COHORT chr$CHR  $CPUS CPUs  ${MEM_MB} MB  ($(date))"
echo "raw:    $GENO_PREFIX"; echo "keep:   $KEEP_LIST ($(wc -l < "$KEEP_LIST") lines)"; echo "out:    $C"

# gene files: production's 12a `snps` outputs survive at $PNET_CHR; make them there only if absent
GENES=$PNET_CHR/genes; MANIFEST=$PNET_CHR/gene_manifest.tsv; WINDOW=$PNET_CHR/window.snplist
if [[ ! -f "$MANIFEST" || ! -f "$WINDOW" ]]; then
    GENES=$C/genes; MANIFEST=$C/gene_manifest.tsv; WINDOW=$C/window.snplist
    if [[ ! -f "$MANIFEST" || "$FORCE" == 1 ]]; then
        echo "no production gene files for chr$CHR -- making them with 12_prsnet.py snps"
        for f in "${TARGET}.PH.bim" "$PRSNET_WINDOWS"; do [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }; done
        mkdir -p "$GENES"
        "$PRSNET_PY" "$PRSNET_SCRIPT" snps --chr "$CHR" --windows "$PRSNET_WINDOWS" \
            --gwas "$GQC/gwas.QC.txt" --bim "${TARGET}.PH.bim" \
            --out-dir "$GENES" --manifest "$MANIFEST" --union "$WINDOW" ${PRSNET_BIG_SNPS:+--big-snps "$PRSNET_BIG_SNPS"}
        lap "gene files"
    fi
fi
for f in "$GENO_PREFIX.bed" "$GENO_PREFIX.bim" "$GENO_PREFIX.fam" "$GQC/gwas.QC.txt" "$GQC/SNP.pvalue" "$MIS/target_data.a1" "$KEEP_LIST" "$MANIFEST" "$WINDOW"; do
    [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
done
echo "genes:  $MANIFEST ($(awk 'NR>1 && $4>0' "$MANIFEST" | wc -l) with SNPs; $(wc -l < "$WINDOW") window SNPs)"

# --- 12a: the cohort bed, seeked out of the raw release ------------------------
if [[ -f "$BED.bed" && -f "$BED.bim" && -f "$BED.fam" && "$FORCE" != 1 ]]; then
    echo "12a: $BED.bed exists -- kept"
elif [[ -f "$C/scores/scores_manifest.tsv" && "$FORCE" != 1 ]]; then
    echo "12a: scores already exist for chr$CHR and the bed was deleted -- nothing to do"; exit 0
else
    BENCH_THREADS=$CPUS "$PRSNET_PY" "$BENCH/variants/bed_seek_v1/bed_seek.py" \
        --raw "$GENO_PREFIX" --extract "$WINDOW" --keep "$KEEP_LIST" --a1 "$MIS/target_data.a1" \
        --out "$BED" --threads "$CPUS" --block 256 --max-gap "${CHAIN_MAX_GAP:-64}" --unassignable minor
    lap "12a bed_seek_v1"
fi
echo "bed:    $(wc -l < "$BED.bim") variants x $(wc -l < "$BED.fam") people  ($(du -h "$BED.bed" | cut -f1))"

# --- 12b: production's plink --clump per gene, all genes, largest first --------
need_module
PLINK_MEM_EACH=$(( MEM_MB * 85 / 100 / CPUS )); (( PLINK_MEM_EACH <= ${CHAIN_PLINK_MB:-2500} )) || PLINK_MEM_EACH=${CHAIN_PLINK_MB:-2500}   # plink allocates its whole --memory up front: the 2026-10-23 jobs showed MaxRSS == request
clump_one() {
    local g=$1
    local out=$C/clumps/$g
    [[ -f "$out.clumped" && "$FORCE" != 1 ]] && return 0
    plink --bfile "$BED" --extract "$GENES/$g.snplist" --chr "$CHR" \
          --clump-p1 1 --clump-r2 "$PRSNET_CLUMP_R2" --clump-kb "$PRSNET_CLUMP_KB" \
          --clump "$GENES/$g.assoc" "$GQC/gwas.QC.txt" --clump-index-first \
          --clump-snp-field SNP --clump-field P \
          --out "$out" --memory "$PLINK_MEM_EACH" --threads 1 > /dev/null 2>&1 || true
    return 0
}
export -f clump_one; export C BED GENES CHR GQC PRSNET_CLUMP_R2 PRSNET_CLUMP_KB PLINK_MEM_EACH FORCE
awk 'NR>1 && $4>0 {print $4"\t"$1}' "$MANIFEST" | sort -k1,1nr | cut -f2 > "$C/genes_by_size.txt"
echo "12b: $(wc -l < "$C/genes_by_size.txt") genes, $CPUS plink at a time, $PLINK_MEM_EACH MB each; largest: $(awk 'NR>1 && $4>0 {print $1, $4}' "$MANIFEST" | sort -k2,2nr | awk 'NR<=3' | tr '\n' ' ')"
xargs -a "$C/genes_by_size.txt" -P "$CPUS" -I{} bash -c 'clump_one "$1"' _ {}
# clump_manifest.tsv: identical awk to 12_prsnet.sh write_12b_results
awk -v D="$C/clumps" 'BEGIN{OFS="\t"}
     NR==1 {print $0, "n_index", "status"; next}
     { n=0; st="nosnp"
       if ($4>0) { f=D"/"$1".clumped"; st="failed"
           if ((getline l < f) > 0) { st="ok"; while ((getline l < f) > 0) { split(l,a," "); if (length(a)>=3) n++ } }
           close(f) }
       print $0, n, st }' "$MANIFEST" > "$C/clump_manifest.tsv"
NFAIL=$(awk -F'\t' 'NR>1 && $NF=="failed"' "$C/clump_manifest.tsv" | wc -l)
echo "12b: $(awk -F'\t' 'NR>1 && $NF=="ok"' "$C/clump_manifest.tsv" | wc -l) ok, $NFAIL failed, $(awk -F'\t' 'NR>1 && $NF=="nosnp"' "$C/clump_manifest.tsv" | wc -l) without SNPs"
(( NFAIL == 0 )) || { awk -F'\t' 'NR>1 && $NF=="failed" {print "  failed:", $1}' "$C/clump_manifest.tsv"; echo "12b: a plink --clump failed (no .clumped); rerun resumes"; exit 1; }
lap "12b plink clump"

# --- 12c: ct_sparse_v5 kernel, 12c's chunk layout -------------------------------
if [[ -f "$C/scores/scores_manifest.tsv" && "$FORCE" != 1 ]]; then
    echo "12c: $C/scores/scores_manifest.tsv exists -- kept"
else
    "$PRSNET_PY" "$BENCH/lib/make_ct_weights.py" --clumps "$C/clumps" --manifest "$C/clump_manifest.tsv" \
        --gwas "$GQC/gwas.QC.txt" --pvalue "$GQC/SNP.pvalue" --thresholds "$PRSNET_THRESHOLDS" \
        --out "$C/ct_weights.tsv" --genes-out "$C/genes_all.txt"
    lap "12c weights"
    "$PRSNET_PY" "$BENCH/variants/ct_sparse_v5/sparse_score.py" --bed "$BED" --weights "$C/ct_weights.tsv" \
        --out "$C/scores" --mode ct --thresholds "$PRSNET_THRESHOLDS" --all-genes "$C/genes_all.txt" \
        --block 4000 --threads "$CPUS" --chunk-gb "${CHAIN_CHUNK_GB:-1.0}" --out-layout chunks
    lap "12c ct_sparse_v5"
fi
echo "scores: $(awk 'NR>1' "$C/scores/scores_manifest.tsv" | wc -l) chunks, $(awk 'NR>1' "$C/scores/gene_counts.tsv" | wc -l) genes  ($(du -sh "$C/scores" | cut -f1))"

# --- the bed is scratch; 12d needs only scores/ ----------------------------------
if [[ "${CHAIN_KEEP_BED:-0}" == 1 ]]; then
    echo "bed kept (CHAIN_KEEP_BED=1): $BED.*"
else
    rm -f "$BED.bed"; echo "bed deleted ($BED.bed); .bim/.fam kept"
fi
rm -f "$C/ct_weights.tsv"
lap "chr$CHR done"
