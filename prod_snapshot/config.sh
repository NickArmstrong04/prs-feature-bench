# =============================================================================
#  config.sh -- shared configuration for the PRS benchmark pipeline
#
#  Sourced by run_step.sh, 08/09/10 and the submitters. Together with
#  diseases/$DISEASE.sh this is the only thing you should need to edit. Not
#  executable on its own.
#
#  MULTI-DISEASE LAYOUT (2026-09-12)
#  Every path that depends on the trait or the cohort is derived from two
#  variables and nothing else:
#
#      DISEASE   t2d, cad, ad, ...    picks diseases/$DISEASE.sh
#      COHORT    eur | all | caucasian | nopheno
#
#  which give RUN=$DISEASE/$COHORT and therefore
#
#      data/processed/t2d/eur/...     (outputs)
#      data/pheno/t2d/eur/...         (keep-list + phenotype)
#      ref/1000G_EUR/t2d/eur/...      (per-run LD panel)
#      logs/t2d/eur/...               (only when submitted via run_pipeline.sh)
#
#  Those directories are created here, at source time, before any step runs --
#  so a new disease never writes loose files into data/processed/ the way the
#  first run did. The guards below refuse to run if RUN would resolve to the
#  processed root, and warn if step directories are found sitting there.
#
#  Adding a disease = adding diseases/<name>.sh (copy diseases/_template.sh).
#  Nothing in 00-07 changes.
#
#  SECOND METHOD (2026-09-16): LDpred2 lives in 10_ldpred2.sh / 10_ldpred2.R
#  and reads the "LDpred2" block near the end of this file. It shares
#  everything above it (run, phenotype, panel, target QC).
#
#  THIRD METHOD (2026-09-21): PRSice-2 lives in 11_prsice.sh and reads the
#  "PRSice-2" block after the LDpred2 one. Same sharing.
#
#  MODE: genome-wide, one chromosome per Slurm array task. The smoke test
#  (chr10:113-116 Mb around TCF7L2) has served its purpose and is off.
# =============================================================================

# --- Project root ----------------------------------------------------------
ROOT=/home/naa79/project_pi_sz622/naa79/prs-benchmark

# Directory this file lives in (scripts/slurm). Used to find diseases/*.sh.
# $0 under sbatch is Slurm's spool copy, but BASH_SOURCE always points at the
# file actually being sourced, so this works in every invocation style.
CONF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Trait -----------------------------------------------------------------
# Selects diseases/$DISEASE.sh, which supplies the summary statistics, the
# genome build, the canary variant and the phenotype definition.
DISEASE="${DISEASE:-t2d}"

# --- Cohort ----------------------------------------------------------------
# The cohort name IS the ancestry rule; they are not two independent knobs.
# Keeping them separate is how a run ends up in a directory whose name does
# not describe its contents, so PHENO_ANCESTRY is derived here and an
# explicit, disagreeing PHENO_ANCESTRY is a hard error rather than a silent
# relabel.
#
#   all        everyone passing sample QC (what the authors' generate_label.py
#              actually did; reproduces their ~411k cohort size)
#   eur        PC-inferred European (Yurui's ukb_qc_EUR.txt, 463,665)
#   caucasian  UKB field 22006 "Caucasian" (409,378)
#   nopheno    no --keep and no --pheno at all (the Sep-10 exploratory run)
COHORT="${COHORT:-eur}"

case "$COHORT" in
    all)        _ANCESTRY=none      ;;
    eur)        _ANCESTRY=eur       ;;
    caucasian)  _ANCESTRY=caucasian ;;
    nopheno)    _ANCESTRY=none      ;;
    *)  echo "config.sh: unknown COHORT '$COHORT'"
        echo "  known: all | eur | caucasian | nopheno"
        echo "  A new cohort rule needs a case arm here AND a matching rule in"
        echo "  08_make_pheno.py -- do not just invent a name, or the directory"
        echo "  will not describe what is in it."
        exit 1 ;;
esac
if [[ -n "${PHENO_ANCESTRY:-}" && "$PHENO_ANCESTRY" != "$_ANCESTRY" ]]; then
    echo "config.sh: PHENO_ANCESTRY=$PHENO_ANCESTRY contradicts COHORT=$COHORT"
    echo "  (COHORT=$COHORT implies ancestry rule '$_ANCESTRY')"
    echo "  Name the cohort instead of overriding the rule, so the output"
    echo "  directory keeps describing its contents."
    exit 1
fi
PHENO_ANCESTRY=$_ANCESTRY

# --- Run name = path fragment under data/processed, data/pheno, ref/ --------
# Normally "$DISEASE/$COHORT". An explicit RUN still works and may be flat
# (RUN=t2d_pheno), which is how the pre-migration directories are reachable.
RUN_EXPLICIT=0; [[ -n "${RUN:-}" ]] && RUN_EXPLICIT=1
RUN="${RUN:-$DISEASE/$COHORT}"
RUN="${RUN#/}"; RUN="${RUN%/}"

for _v in DISEASE COHORT; do
    [[ "${!_v}" =~ ^[a-z0-9][a-z0-9_]*$ ]] || {
        echo "config.sh: $_v='${!_v}' -- use lowercase letters, digits and _ only"
        exit 1; }
done
# Each path component must be a plain name: this is what rejects RUN=".",
# "..", "a/./b" and anything that could climb out of data/processed or
# resolve back to it.
_bad_run=0
[[ -n "$RUN" ]] || _bad_run=1
IFS='/' read -r -a _run_parts <<< "$RUN"
for _p in ${_run_parts[@]+"${_run_parts[@]}"}; do
    [[ "$_p" =~ ^[a-z0-9][a-z0-9_.-]*$ && "$_p" != .* ]] || _bad_run=1
done
(( _bad_run == 0 )) || {
    echo "config.sh: refusing RUN='$RUN'"
    echo "  Each part must be a plain lowercase name (letters, digits, _ - .)"
    echo "  so that every output lands in a named subdirectory of"
    echo "  data/processed and nowhere else."
    exit 1; }

# --- Per-disease settings --------------------------------------------------
# Must define: BUILD, SUMSTATS_FULL, CANARY_CHR, CANARY_RSID, CANARY_CHRPOS,
# PHENO_SOURCE, and (for PHENO_SOURCE=icd) ICD_CASE_PREFIX,
# ICD_EXCLUDE_PREFIX, ICD_CONTROL_EXCLUDE_PREFIX. See diseases/_template.sh.
DISEASE_FILE="$CONF_DIR/diseases/$DISEASE.sh"
[[ -f "$DISEASE_FILE" ]] || {
    echo "config.sh: no such disease file: $DISEASE_FILE"
    echo "  Available: $(cd "$CONF_DIR/diseases" 2>/dev/null && ls *.sh 2>/dev/null | grep -v '^_' | sed 's/\.sh$//' | tr '\n' ' ')"
    echo "  Add one by copying diseases/_template.sh."
    exit 1; }
DISEASE_NAME=""; BUILD=""; PHENO_SOURCE=""
source "$DISEASE_FILE"

for _v in BUILD SUMSTATS_FULL CANARY_CHR CANARY_RSID CANARY_CHRPOS PHENO_SOURCE; do
    [[ -n "${!_v:-}" ]] || { echo "config.sh: $DISEASE_FILE does not set $_v"; exit 1; }
done
case "$PHENO_SOURCE" in
    icd)
        # Only the case codes are mandatory. Both exclusion sets may legitimately
        # be empty (plenty of traits have no subtype to exclude), so they must be
        # DEFINED but need not be non-empty -- an empty set and a forgotten
        # variable look identical otherwise.
        [[ -n "${ICD_CASE_PREFIX:-}" ]] || { echo "config.sh: PHENO_SOURCE=icd but $DISEASE_FILE does not set ICD_CASE_PREFIX"; exit 1; }
        for _v in ICD_EXCLUDE_PREFIX ICD_CONTROL_EXCLUDE_PREFIX; do
            [[ -n "${!_v+set}" ]] || { echo "config.sh: PHENO_SOURCE=icd but $DISEASE_FILE does not define $_v (use \"\" for none)"; exit 1; }
        done ;;
    external)
        for _v in PHENO_EXTERNAL_KEEP PHENO_EXTERNAL_PHENO; do
            [[ -n "${!_v:-}" ]] || { echo "config.sh: PHENO_SOURCE=external but $DISEASE_FILE does not set $_v"; exit 1; }
        done ;;
    none) ;;
    *)  echo "config.sh: PHENO_SOURCE='$PHENO_SOURCE' -- expected icd | external | none"; exit 1 ;;
esac

# --- Modules (Bouchet) -----------------------------------------------------
# Bouchet has no default PLINK or R module: a bare `module load PLINK` fails,
# and the old scripts hid that failure behind `|| true`, so a job only worked
# when the modules happened to be inherited from the submitting shell.
#
# The pipeline is written in PLINK 1.9 syntax (--make-just-bim, --a1-allele,
# --write-snplist, --clump, --score). It must be the 1.9 module, NOT
# PLINK/2_avx2_20250707 -- plink2 has different flag names and a different
# binary name.
PLINK_MODULE=PLINK/1.9b_7.11-x86_64
R_MODULE=R/4.4.2-gfbf-2024a          # mismatch.R uses base R only; LDpred2
                                     # adds bigsnpr into $LDPRED_RLIB (below)

# --- Chromosome ------------------------------------------------------------
# CHR is normally supplied by the caller. Under `sbatch --array` each step
# script exports CHR=$SLURM_ARRAY_TASK_ID BEFORE sourcing this file. The
# fallback below only applies to an interactive run with nothing set.
CHR="${CHR:-10}"

# --- Region subsetting (OFF) -----------------------------------------------
# NOTE THE SINGLE DASH. With ${VAR:-default}, an explicitly empty value is
# treated as unset and the default comes back -- so `REGION_START= sbatch ...`
# would silently re-run the smoke test while looking like a genome-wide run.
# ${VAR-default} honors an empty value. This is the single most dangerous
# line in the file; do not "tidy" the dash back.
#
# To deliberately re-run the smoke test:
#   CHR=10 REGION_START=113000000 REGION_END=116000000 bash run_step.sh 00
REGION_START="${REGION_START-}"
REGION_END="${REGION_END-}"

# --- Canary variant --------------------------------------------------------
# Set per disease (CANARY_CHR / CANARY_RSID / CANARY_CHRPOS in
# diseases/$DISEASE.sh): a variant that must survive the pipeline if the
# base/target join worked. It lives on ONE chromosome, so every step guards
# its canary block on CANARY_CHR -- without that guard, 21 of 22 array tasks
# would print a false failure.

# --- Inputs ----------------------------------------------------------------
# Target genotypes: PLINK binary prefix, WITHOUT .bed/.bim/.fam. Shared by
# every disease -- the trait only changes the base GWAS and the labels.
GENO_PREFIX=$ROOT/data/raw/ukbb/snps/ukb22828_c${CHR}_b0_v3

# Base GWAS summary statistics: SUMSTATS_FULL, set per disease. Must be the
# 12-column-ready layout 02 expects (tab separated, header
# "SNP Chr Pos EA NEA EAF Beta SE Pvalue") and must be on the SAME genome
# build as the target -- BUILD documents which, and 02's positional join is
# silently wrong if they disagree.

# The authors' allele-alignment R script.
MISMATCH_R=$ROOT/scripts/slurm/mismatch.R

# --- Sample selection and phenotype ----------------------------------------
# Built by 08_make_pheno.sh from Yurui's phenotype/QC tables (read-only
# inputs, this genotype set's ID space). Definition and counts: see
# $PHENO_DIR/README.md. With PHENO present, PLINK computes --hwe in controls
# only, and 03 runs the differential-missingness filter (paper step 3f).
PHENO_DIR=$ROOT/data/pheno/$RUN
PHENO_SRC_ICD=/nfs/roberts/pi/pi_sz622/yc2352/ukbb/pheno/cohort_ICD.csv
PHENO_SRC_SEX=/nfs/roberts/pi/pi_sz622/yc2352/ukbb/pheno/cohort_sex.csv
PHENO_SRC_QC=/nfs/roberts/pi/pi_sz622/yc2352/ukbb/genotype_QC/sample_qc_results/ukb_qc_all.txt
PHENO_SRC_EUR=/nfs/roberts/pi/pi_sz622/yc2352/ukbb/genotype_QC/sample_qc_results/ukb_qc_EUR.txt
PHENO_SRC_ANC=/nfs/roberts/pi/pi_sz622/yc2352/ukbb/genotype_QC/cohort_ancestry.csv

# File basename inside $PHENO_DIR. Defaults to the disease, which keeps the
# existing names (t2d_pheno.tab, t2d_keep.txt) byte-for-byte.
PHENO_BASENAME="${PHENO_BASENAME:-$DISEASE}"
PHENO_DEFAULT=$PHENO_DIR/${PHENO_BASENAME}_pheno.tab
KEEP_DEFAULT=$PHENO_DIR/${PHENO_BASENAME}_keep.txt

# A disease whose labels come from somewhere other than cohort_ICD.csv
# (self-report fields, operation codes, an algorithmic definition, a
# collaborator's file) sets PHENO_SOURCE=external in its disease file and
# points at ready-made files; 08 then refuses to run instead of writing a
# phenotype it has no definition for.
if [[ "$PHENO_SOURCE" == external ]]; then
    PHENO_DEFAULT="$PHENO_EXTERNAL_PHENO"
    KEEP_DEFAULT="$PHENO_EXTERNAL_KEEP"
fi

# KEEP_LIST / PHENO default to those files when they exist; an explicit value
# (env or sbatch --export) overrides; an explicitly EMPTY value disables them
# (phenotype-free run), as does COHORT=nopheno.
KEEP_LIST="${KEEP_LIST-$KEEP_DEFAULT}"
PHENO="${PHENO-$PHENO_DEFAULT}"
if [[ "$COHORT" == nopheno || "$PHENO_SOURCE" == none ]]; then
    KEEP_LIST=""; PHENO=""
fi
[[ -n "$KEEP_LIST" && ! -f "$KEEP_LIST" ]] && KEEP_LIST=""
[[ -n "$PHENO"     && ! -f "$PHENO"     ]] && PHENO=""

# Expand to (--keep FILE) / (--pheno FILE) or to nothing. Use at call sites as
#     ${KEEP_ARG[@]+"${KEEP_ARG[@]}"}      ${PHENO_ARG[@]+"${PHENO_ARG[@]}"}
# The ${a[@]+...} guard is required: under `set -u` a bare "${a[@]}" on an
# empty array is an error in bash < 4.4.
KEEP_ARG=();  [[ -n "$KEEP_LIST" ]] && KEEP_ARG=(--keep "$KEEP_LIST")
PHENO_ARG=(); [[ -n "$PHENO" ]]     && PHENO_ARG=(--pheno "$PHENO")

# --- LD reference panel: 1000 Genomes phase 3, EUR, GRCh37 -----------------
# The paper (l. 169-171) clumps in 1000G EUR for every method, and the
# authors' reference pipeline intersects the variant set with a 1000G panel.
# 05a_build_1000g.sh downloads the public phase 3 VCFs and builds one PLINK
# fileset per chromosome, restricted to the 503 EUR samples and to variants
# that exist in OUR aligned target with the same alleles -- and renamed to
# OUR variant IDs, so --clump can key gwas.QC.txt against it directly.
#
# REF_RAW (the ~15 GB of VCFs) is shared by every run and downloaded once.
# REF_DIR is per-run because the panel's variant set is the intersection with
# THIS run's aligned target, which changes with the disease: a new disease
# re-runs 05a's plink2 steps but re-downloads nothing.
G1000_URL=https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502
G1000_PANEL=integrated_call_samples_v3.20130502.ALL.panel
G1000_POP=EUR
REF_RAW=$ROOT/ref/1000G_${G1000_POP}/raw      # downloaded VCFs (~15 GB), shared
REF_DIR=$ROOT/ref/1000G_${G1000_POP}/$RUN     # per-run panels
REF_PREFIX=$REF_DIR/g1000_${G1000_POP}_chr${CHR}   # per-chromosome PLINK prefix
PLINK2_MODULE=PLINK/2_avx2_20250707           # 05a (VCF import) and 10c (scoring)

# --- Clumping and thresholding (paper, section 2.1) ------------------------
CLUMP_R2=0.5
CLUMP_KB=250
# --q-score-range file: label  lower  upper. Ten thresholds, l. 111.
P_THRESHOLDS="1e-5 1e-4 1e-3 1e-2 5e-2 0.1 0.2 0.3 0.5 1"

# --- Output directories ----------------------------------------------------
# GQC and MIS are PER-CHROMOSOME. Their contents (gwas.a1.txt .. gwas.QC.txt,
# target_data.a1) carry no chromosome in the filename, so with 22 array tasks
# writing to one directory each chromosome would overwrite the last and you
# would end the run holding chr22 only. Subdirectories fix this without
# touching a single filename reference in 02/03/04.
PROC=$ROOT/data/processed/$RUN
SUB=$ROOT/data/test_subset/$RUN/raw          # 00: archived smoke-test subset
TQC=$PROC/target_qc                          # 01, 04: QC'd + aligned target
GQC=$PROC/gwas_qc/chr${CHR}                  # 02, 03: gwas.a1 .. gwas.QC
MIS=$PROC/mismatch_fix/chr${CHR}             # 03: mismatch.R outputs
LOG=$ROOT/logs                               # #SBATCH paths are literal; the
                                             # per-run logs/$RUN tree is used
                                             # only by run_pipeline.sh
CLP=$PROC/clump/chr${CHR}                    # 05: clumping outputs
SCR=$PROC/scores/chr${CHR}                   # 06: per-threshold profiles

# The one path that must never be the processed root itself. This is the
# failure the layout above exists to prevent: 22 array tasks writing
# target_qc/, gwas_qc/, clump/ and scores/ straight into data/processed/,
# where a second disease would quietly overwrite the first.
[[ "$PROC" != "$ROOT/data/processed" ]] || {
    echo "config.sh: PROC resolved to the processed root -- refusing to run."
    echo "  Set DISEASE and COHORT (or RUN) so outputs land in a subdirectory."
    exit 1; }

# --- Derived file names (do not edit) --------------------------------------
SUBSET_GENO=$SUB/chr${CHR}_sub                       # subset PLINK prefix
SUMSTATS_SUB=$SUB/sumstats_chr${CHR}_region.txt      # subset sumstats
TARGET=$TQC/target_chr${CHR}                         # QC'd target prefix
                                                     # (.bim and .snplist share
                                                     #  this -- mismatch.R needs
                                                     #  both)

# --- QC thresholds ---------------------------------------------------------
# Set per disease only if that disease needs different ones; these are the
# values the T2D runs used (see run_status.md, "Target-QC threshold decision").
TARGET_MAF="${TARGET_MAF:-0.001}"   # target: minor allele frequency floor
TARGET_GENO="${TARGET_GENO:-0.02}"  # target: max per-variant missingness
TARGET_HWE="${TARGET_HWE:-1e-10}"   # target: Hardy-Weinberg exact test floor

# Base MAF floor. The paper also states INFO > 0.3, but the sumstats carry no
# INFO column, and the authors' own reference script hardcodes a placeholder
# INFO (0.9) that passes everything -- so the INFO filter is a no-op in their
# run as well. We hardcode 1 in 02; this matches the reference.
BASE_MAF="${BASE_MAF:-0.001}"

# Reported sample size of the base GWAS. UNUSED since 02 adopted the
# reference's per-SNP effective N. Kept only for documentation.
N_STUDY=NA

# --- LDpred2 (second method, 2026-09-16) -----------------------------------
# Everything below is read only by 10_ldpred2.sh / 10_ldpred2.R and by 09 when
# METHOD=ldpred2_*. Nothing in steps 01-07 changes. Read ldpred2.md before
# touching a value: several of these are compromises forced by the 503-sample
# 1000G panel, and the doc says which.
#
# Directory layout, all per run:
#   data/processed/$RUN/ldpred2/
#       chr<N>/          10a: panel.{bk,rds}, corr_chr<N>.rds, df_beta_chr<N>.rds
#       genome/          10b: corr.sbk (genome-wide SFBM), df_beta.rds, ldsc.tsv,
#                             betas.txt, grid_params.tsv, auto_chains.tsv
#       scores/chr<N>/   10c: plink2 .sscore per chromosome
#       genome/prs.ldpred2_{inf,grid,auto}.txt   10d: the matrices 09 reads
#   data/processed/$RUN/eval_ldpred2_{inf,grid,auto}/   09 with METHOD set
LDP=$PROC/ldpred2
LDP_CHR=$LDP/chr${CHR}
LDP_GEN=$LDP/genome
LDP_SCR=$LDP/scores/chr${CHR}

# Shared, downloaded once by `bash 10_ldpred2.sh setup` on a login node:
#   map_hm3_plus.rds   the LDpred2 authors' HapMap3+ variant map (1,444,196
#                      variants, hg19 positions, UKB LD scores) -- LDpred2 is
#                      designed for this variant set
#   genetic_map/       1000G OMNI interpolated genetic maps, chr1-22, used to
#                      convert bp to cM for the 3 cM LD window
LDPRED_REF=$ROOT/ref/ldpred2
LDPRED_HM3=$LDPRED_REF/map_hm3_plus.rds
LDPRED_GENMAP=$LDPRED_REF/genetic_map
# The figshare API endpoint, not the ndownloader link the LDpred2 tutorial
# gives: on 2026-09-17 the latter answered "202 Accepted" with an empty body
# (file staged from cold storage) to wget and to R alike, and every client
# took that as a successful 0-byte download. The API URL redirects straight
# to S3 and gave the 43,814,795-byte file first time. Login node only:
# compute nodes reach CRAN and GitHub but not figshare.
LDPRED_HM3_URL=https://api.figshare.com/v2/file/download/37802721

# R packages go in a project-local library so the module's site library is
# never touched and the install survives module changes. Populated by
# `bash 10_ldpred2.sh setup` (compiles bigsnpr and friends; ~20-40 min once).
#
# Handed to R as LDPRED_RLIB, NOT via R_LIBS_USER: the R module sets its own
# R_LIBS_USER on load and Lmod prepends it to ours, so R would see a
# colon-joined string and install.packages() would treat the whole thing as
# one (non-existent) directory. That is exactly what happened on 2026-09-17.
# 10_ldpred2.R and the setup step put $LDPRED_RLIB at the front of .libPaths().
LDPRED_RLIB=$ROOT/tools/Rlib
export LDPRED_RLIB

# --- LDpred2 parameters (the LDpred2 tutorial's defaults unless noted) -------
LDPRED_WINDOW_CM=3         # LD window, cM (snp_cor size = 3/1000)
# r2 below which a correlation is stored as 0. The tutorial computes LD in
# thousands of samples and keeps everything; with 503 samples the noise floor
# is ~1/503 = 0.002, so pairs under 0.01 are mostly noise and dropping them
# shrinks the genome-wide matrix several-fold. NOT a tutorial default.
LDPRED_THR_R2=0.01
# auto: number of chains (tutorial: 50), burn-in and sampling iterations
LDPRED_AUTO_CHAINS=30
LDPRED_AUTO_BURN=500
LDPRED_AUTO_ITER=500
# auto: correlation shrinkage. The tutorial default is 0.95 "for some mismatch
# with the LD ref" and says to go as low as 0.4 for a LARGE mismatch. Our
# reference is 503 people vs the 2,000+ the LDpred2 authors advise, which is a
# large mismatch -- at 0.95 the T2D auto chains were too noisy and the quality
# filter kept <3 (2026-09-20). Lowered to 0.5. If chains still disagree badly
# (see auto_chains.tsv), 0.4 is the floor the tutorial suggests.
LDPRED_SHRINK_CORR=0.5
# auto: use_MLE=FALSE is the tutorial's recommendation "when GWAS power is
# low" or on convergence issues. AD (3k cases in the target, ~40k in the base)
# is the low-power case; leave it off for both diseases for comparability.
LDPRED_USE_MLE=FALSE
# grid: p (polygenicity) and h2 axes, tutorial defaults -> 21 x 4 x 2 = 168
LDPRED_GRID_P="1e-5 1 21"            # seq_log(from, to, length.out)
LDPRED_GRID_H2="0.3 0.7 1 1.4"       # multiples of the LDSC h2 estimate
# per-SNP effective N. The sumstats carry no per-cohort N, so n_eff is imputed
# from SE and allele frequency, n_eff_j = 4 / (2 p_j (1-p_j) se_j^2) (Prive et
# al. 2022, HGG Advances). Variants whose imputed n_eff falls below this
# fraction of the study-wide estimate quantile(8/se^2, 0.999) were typed in a
# minority of the meta-analysed cohorts and are dropped, mirroring the
# tutorial's "N < 0.7 * max(N)" rule but against a robust maximum.
LDPRED_NEFF_MIN_FRAC=0.5

# --- PRSice-2 (third method, 2026-09-21) -----------------------------------
# Read only by 11_prsice.sh / run_prsice.sh and by 09 when METHOD=prsice.
# Nothing in steps 01-07 or 10 changes. The reasoning is in prsice.md.
#
# Directory layout, all per run:
#   data/processed/$RUN/prsice/
#       chr<N>/     11a: base.txt (P floored), bar_levels.txt,
#                        chr<N>.{all_score.gz,prsice,snp,log}
#       genome/     11b: prs.prsice.txt (the matrix 09 reads), thresholds.tsv
#   data/processed/$RUN/eval_prsice/   09 with METHOD=prsice
PRS=$PROC/prsice
PRS_CHR=$PRS/chr${CHR}
PRS_GEN=$PRS/genome

# The binary, fetched once by `sbatch 11_prsice.sh setup`: PRSice-2 is one
# Linux executable (plus PRSice.R, which we do not use) -- no R packages, no
# compilation. 2.3.5 (2021) is the current release. Shared by every run.
PRSICE_VERSION=2.3.5
PRSICE_DIR=$ROOT/tools/prsice
PRSICE_BIN=$PRSICE_DIR/PRSice_linux
PRSICE_URL=https://github.com/choishingwan/PRSice/releases/download/$PRSICE_VERSION/PRSice_linux.zip

# --- PRSice-2 parameters ---------------------------------------------------
# Clumping reuses CLUMP_R2 / CLUMP_KB above (paper section 2.1) with
# --clump-p 1, and the per-run 1000G EUR panel (REF_PREFIX) as --ld: the
# paper's stated LD source for every method, and what 05 already used.
#
# Threshold grid. PRSice's own default is linear, 5e-8 to 0.5 in steps of
# 5e-5: 10,000 columns (a ~30 GB score matrix at 386k people) with only two
# of them below 1e-4, where AD's C+T optimum sits. A geometric grid of
# 1,000 points over the same range steps 1.6% at a time everywhere, and the
# paper's ten thresholds (P_THRESHOLDS, incl. 1) are merged in so the C+T
# columns are an exact subset. The threshold is chosen on validation, by 09.
PRSICE_GRID=log              # log | linear
PRSICE_P_LOWER=5e-8
PRSICE_P_UPPER=0.5
PRSICE_N_THRESHOLDS=1000     # log grid: number of points
PRSICE_INTERVAL=5e-4         # linear grid only: step (PRSice's default is 5e-5)
PRSICE_INCLUDE_PAPER=1       # 1: merge P_THRESHOLDS into the grid
# 'sum' for the same reason as 06: per-chromosome scores are added in 11b,
# and PRSice's default 'avg' divides by the number of alleles scored, which
# differs by chromosome. MEAN_IMPUTE is PRSice's default and matches PLINK
# --score: a missing genotype contributes 2p * beta.
PRSICE_SCORE=sum
PRSICE_MISSING=MEAN_IMPUTE

# --- Resources -------------------------------------------------------------
# Derived from what Slurm actually granted, so a smaller request on the
# command line (e.g. `sbatch --mem=16G --cpus-per-task=4 --time=3:00:00`)
# is safe: PLINK gets ~85% of the job's memory as workspace and one thread
# per allocated core. Asking PLINK for more memory than the job has does
# not fail gracefully -- the cgroup kills the job. Outside Slurm (an
# interactive `bash 01_...sh`), fall back to modest values.
#
# SLURM_MEM_PER_NODE is in MB and is set whenever --mem is used.
PLINK_MEM=$(( ${SLURM_MEM_PER_NODE:-16000} * 85 / 100 ))   # MB
PLINK_THREADS=${SLURM_CPUS_PER_TASK:-4}

# =============================================================================
#  Helpers
# =============================================================================

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

require_file() {
    [[ -f "$1" ]] || { echo "MISSING INPUT: $1"; echo "$2"; exit 1; }
}

require_bfile() {
    for ext in bed bim fam; do
        [[ -f "${1}.${ext}" ]] || { echo "MISSING INPUT: ${1}.${ext}"; echo "$2"; exit 1; }
    done
}

require_cmd() {
    command -v "$1" >/dev/null || { echo "NOT ON PATH: $1 -- $2"; exit 1; }
}

# Load one or more Lmod modules, loudly. Under sbatch the `module` shell
# function is normally inherited from the submitting shell; if it is not, try
# the usual Lmod init file. Lmod is briefly run with `set -u` off because its
# shell function is not written for nounset mode.
load_modules() {
    if ! type module >/dev/null 2>&1; then
        for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
            [[ -r "$f" ]] && { source "$f"; break; }
        done
    fi
    set +u
    local m
    for m in "$@"; do
        if ! module load "$m"; then
            echo "FAILED: module load $m -- check the module name in config.sh"
            set -u; exit 1
        fi
    done
    set -u
    module list 2>&1 | grep -E 'PLINK|R/' || true
}

# rows excluding a header line
count() { [[ -f "$1" ]] && echo "$(( $(wc -l < "$1") - 1 ))" || echo "NA"; }

# samples surviving a PLINK run, read back from its own log
samples_from_log() {
    grep -oE "[0-9]+ (people|samples) (pass filters and QC|remaining)" "$1" \
        | tail -1 | grep -oE "^[0-9]+" || echo "NA"
}

# One line naming the run, for the top of every step's log.
run_banner() {
    printf '\n\033[1m== run: %s  (disease %s, cohort %s, build %s)\033[0m\n' \
        "$RUN" "$DISEASE" "$COHORT" "$BUILD"
    printf '   sumstats: %s\n' "$SUMSTATS_FULL"
    printf '   outputs:  %s\n' "$PROC"
    printf '   pheno:    %s\n' "${PHENO:-NONE}"
    # An explicit RUN detaches the directory name from DISEASE/COHORT, which is
    # exactly what makes the pre-migration paths reachable -- and exactly how a
    # run gets read under the wrong cohort's name.
    if (( RUN_EXPLICIT )) && [[ "$RUN" != "$DISEASE/$COHORT" ]]; then
        printf '   NOTE: RUN was set explicitly, so "%s" need not describe\n' "$RUN"
        printf '         disease=%s cohort=%s. Trust run_info.txt in that\n' "$DISEASE" "$COHORT"
        printf '         directory, not these two variables.\n'
    fi
}

# Written once per run directory, so a directory found months later says what
# produced it. mv -n and the pid-suffixed temp keep 22 concurrent array tasks
# from colliding.
_write_run_info() {
    local f="$PROC/run_info.txt" tmp
    [[ -e "$f" ]] && return 0
    tmp="$f.$$"
    {
        echo "run:        $RUN"
        echo "disease:    $DISEASE ${DISEASE_NAME:+($DISEASE_NAME)}"
        echo "cohort:     $COHORT  (ancestry rule: $PHENO_ANCESTRY)"
        echo "build:      $BUILD"
        echo "sumstats:   $SUMSTATS_FULL"
        echo "canary:     $CANARY_RSID  chr$CANARY_CHR  $CANARY_CHRPOS"
        echo "pheno dir:  $PHENO_DIR"
        echo "ld panel:   $REF_DIR"
        echo "target QC:  maf $TARGET_MAF  geno $TARGET_GENO  hwe $TARGET_HWE"
        echo "base MAF:   $BASE_MAF"
        echo "clumping:   r2 $CLUMP_R2  kb $CLUMP_KB"
        echo "thresholds: $P_THRESHOLDS"
        echo "created:    $(date '+%Y-%m-%d %H:%M:%S') by ${USER:-unknown}"
    } > "$tmp" 2>/dev/null && mv -n "$tmp" "$f" 2>/dev/null
    rm -f "$tmp" 2>/dev/null
    return 0
}

# The mistake this layout exists to prevent, checked at source time: step
# directories sitting in the processed root are leftovers from a run that had
# no run name, and a later run of a DIFFERENT disease would merge into them.
_check_processed_root() {
    local d stray=()
    for d in target_qc gwas_qc mismatch_fix clump scores eval; do
        [[ -e "$ROOT/data/processed/$d" ]] && stray+=("$d")
    done
    (( ${#stray[@]} )) || return 0
    echo
    echo "  WARNING: unscoped step directories in $ROOT/data/processed:"
    echo "             ${stray[*]}"
    echo "  These predate the per-disease layout. Move them under a run name"
    echo "  (see migrate_layout.sh) -- while they sit there it is easy to read"
    echo "  one disease's results believing they are another's."
    echo
}

mkdir -p "$TQC" "$GQC" "$MIS" "$LOG" "$CLP" "$SCR" "$PHENO_DIR"
# Only the smoke test uses $SUB; with region subsetting off, creating it would
# leave an empty test_subset tree for every disease.
[[ -n "$REGION_START" ]] && mkdir -p "$SUB"
# Printed at the top of every step's log, because with several diseases in
# flight the first question about any log is which run it belongs to.
run_banner
_write_run_info
_check_processed_root

# --- PRS-Net (fourth method, 2026-10-02) -----------------------------------
# Read only by 12_prsnet.sh / 12_prsnet.py / run_prsnet.sh. Nothing in steps
# 01-11 changes. The reasoning is in prsnet.md.
#
# Directory layout, all per run:
#   data/processed/$RUN/prsnet/            (link it to scratch: ~0.5-1 TB)
#       chr<N>/target_chr<N>.PH.{bed,bim,fam}  12a: cohort bed, window SNPs only
#       chr<N>/genes/<gene>.{snplist,assoc}    12a: SNPs per gene window
#       chr<N>/gene_manifest.tsv               12a
#       chr<N>/clumps/<gene>.clumped           12b: the authors' per-gene clump
#       chr<N>/clump_manifest.tsv              12b
#       chr<N>/scores/scores_chunk<k>.npy      12c: samples x genes x 11
#       chr<N>/check/                          check: authors' per-gene output
#       genome/<disease>_<cohort>/             12d: feats.f32, info.csv (+feats/)
#                                              = train.py --data_path genome --dataset <disease>_<cohort>
PNET=$PROC/prsnet
PNET_CHR=$PNET/chr${CHR}
PNET_GEN=$PNET/genome

# Shared, created once by `12_prsnet.sh setup`:
PRSNET_REPO_URL=https://github.com/lihan97/PRS-Net.git
PRSNET_REPO=$ROOT/tools/PRS-Net                       # git clone (code + gene windows + graph)
PRSNET_GENE_BED=$PRSNET_REPO/data/gene_bed_files_10kb # 19,831 one-line BED files, hg19, +/- 10 kb
PRSNET_GRAPH=$ROOT/ref/prsnet/string/ggi_graph.npz   # STRING v12 physical >=800, our 19,831 genes, sorted order
PRSNET_VENV=$ROOT/tools/prsnet_venv                   # numpy, scipy, bed-reader
PRSNET_PY_MODULE=""                                   # a Python >= 3.9 module if the system
                                                      # python3 is older (module avail Python)
PRSNET_REF=$ROOT/ref/prsnet
PRSNET_WINDOWS=$PRSNET_REF/gene_windows.tsv           # setup: chr start end gene
PRSNET_RANGES=$PRSNET_REF/range_list_full.txt         # setup: the --q-score-range file
# The gene -> graph-node mapping. NOT in the repo (their code takes os.listdir
# order on their own disk). One gene symbol per line, line i = node i. Get it
# from Yurui / the authors before 12d; 12a-12c do not need it.
PRSNET_GENE_ORDER=$PRSNET_REF/gene_order.txt
PRSNET_N_GENES=19831                                  # nodes in ggi_graph.bin (model n_genes)

# --- PRS-Net parameters (the authors' code, 2_data_generation.sh) -----------
# ELEVEN thresholds: the paper's text lists ten, the code and the model
# (d_input=11) use these eleven, 0.4 being the extra one. The strings are the
# range labels generate_prsnet_data.py looks for, so keep the spelling.
PRSNET_THRESHOLDS="1e-5 1e-4 1e-3 0.01 0.05 0.1 0.2 0.3 0.4 0.5 1"
PRSNET_EXT_KB=10                 # window extension already baked into the BED files
PRSNET_CLUMP_R2=0.5
PRSNET_CLUMP_KB=250
# LD for clumping comes from the TARGET cohort (the authors clump against
# target_data.PH), not from the 1000G panel the other methods use.
PRSNET_BED_SCOPE=window          # window: only SNPs inside some gene window (all a
                                 # score can use); all: every QC'd SNP (bigger bed)
PRSNET_BIG_SNPS=20000            # genes with more window SNPs than this are clumped one
                                 # at a time with the whole job's memory (12b)
PRSNET_GIANT_MB=10               # windows wider than this get their own 12c chunk
                                 # (30 genes; annotation artefacts, kept as the authors had them)
PRSNET_PARALLEL=${SLURM_CPUS_PER_TASK:-4}   # concurrent PLINK --clump processes in 12b
PRSNET_CHUNK_GENES=100           # 12c: genes per output chunk (~3.4 GB RAM per 100 at 381k samples)
PRSNET_BLOCK_SNPS=1024           # 12c: SNP columns read from the bed per block
PRSNET_EXPORT=memmap             # 12d: memmap | per-sample | both  (per-sample = the
                                 # authors' feats/<id>.npy layout, 332 GB of small files)
PRSNET_CHECK_RTOL=1e-5           # check: relative tolerance vs plink2's printed SCORE1_AVG
