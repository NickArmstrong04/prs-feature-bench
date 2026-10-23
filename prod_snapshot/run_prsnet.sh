#!/bin/bash
# =============================================================================
#  run_prsnet.sh -- submit the PRS-Net feature build for one run as a
#                   dependency-chained set of Slurm jobs
#
#  Companion to run_pipeline.sh / run_ldpred2.sh / run_prsice.sh. Assumes steps
#  01-04 and 08 of the C+T pipeline have finished for this run (gwas.QC.txt,
#  snplist, target_data.a1, the aligned .bim, the phenotype) and that
#  `sbatch 12_prsnet.sh setup` has run once.
#
#      DISEASE=ad COHORT=eur bash run_prsnet.sh --dry-run
#      DISEASE=ad COHORT=eur bash run_prsnet.sh
#      DISEASE=ad COHORT=eur bash run_prsnet.sh --to 12c --array 19      (smoke test: one chromosome)
#      DISEASE=ad COHORT=eur bash run_prsnet.sh --from 12b --array 1,3   (after fixing something by hand)
#      DISEASE=ad COHORT=eur bash run_prsnet.sh --from 12d               (once gene_order.txt is in place)
#      DISEASE=ad COHORT=eur bash run_prsnet.sh --after 1234567          (queue behind the setup job)
#
#  Options
#      --dry-run        print the sbatch commands and exit
#      --from STEP      start at STEP (12a 12b 12c 12d)
#      --to STEP        stop after STEP
#      --array SPEC     override the array spec for 12a/12b/12c (e.g. 19, or 3,14)
#      --after JOBID    make the chain's first job wait (afterok) for JOBID
#
#  CHAIN
#      12a (array 1-22, %4)  -aftercorr->  12b (array)  -aftercorr->  12c (array)  -afterok->  12d
#
#  TIME. 12a is 04's command with --make-bed: one pass over the raw
#  chromosome (chr1 is 258 GB) plus the write -- like 06, hours for the big
#  chromosomes; budget 12 h, %4 so four jobs are not all reading the release
#  at once. 12b is ~900 one-second PLINK runs per chromosome, parallel, PLUS
#  the giant-window genes (7 on chr1), each a chromosome-scale clump at 381k
#  samples that can take hours: budget 24 h / 64 GB, done genes are skipped on
#  a rerun. 12c reads each index SNP once: minutes per chromosome; budget 6 h.
#  12d streams ~330 GB through once: budget 12 h.
# =============================================================================

set -euo pipefail

DRY=0; FROM=""; TO=""; ARRAY=""; AFTER=""
while (( $# )); do
    case "$1" in
        --dry-run)  DRY=1 ;;
        --from)     FROM="${2:?--from needs a step}"; shift ;;
        --to)       TO="${2:?--to needs a step}"; shift ;;
        --array)    ARRAY="${2:?--array needs a spec}"; shift ;;
        --after)    AFTER="${2:?--after needs a job id}"; shift
                    [[ "$AFTER" =~ ^[0-9]+(:[0-9]+)*$ ]] || { echo "--after: '$AFTER' is not a job id"; exit 1; } ;;
        -h|--help)  sed -n '2,36p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1  (try --help)"; exit 1 ;;
    esac
    shift
done

SDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SDIR"
export CHR=1
source "$SDIR/config.sh"
command -v sbatch >/dev/null || { echo "sbatch not on PATH -- run this on a login node"; exit 1; }

RUN_LOG=$ROOT/logs/$RUN
mkdir -p "$RUN_LOG"

# step | script + args | array|single | depends on | deptype | resources
STEPS=(
    "12a|12_prsnet.sh 12a|array||afterok|--array=1-22%4 --cpus-per-task=4 --mem=16G --time=1:00:00"
    "12b|12_prsnet.sh 12b|array|12a|aftercorr|--array=1-22%6 --cpus-per-task=8 --mem=64G --time=24:00:00"
    "12c|12_prsnet.sh 12c|array|12b|aftercorr|--array=1-22%6 --cpus-per-task=4 --mem=16G --time=1:00:00"
    "12d|12_prsnet.sh 12d|single|12c|afterok|--cpus-per-task=4 --mem=16G --time=4:00:00"
)

step_index() {
    local i
    for i in "${!STEPS[@]}"; do
        [[ "${STEPS[$i]%%|*}" == "$1" ]] && { echo "$i"; return 0; }
    done
    echo -1
}
I_FROM=0; I_TO=$(( ${#STEPS[@]} - 1 ))
if [[ -n "$FROM" ]]; then I_FROM=$(step_index "$FROM"); (( I_FROM >= 0 )) || { echo "--from $FROM: no such step"; exit 1; }; fi
if [[ -n "$TO" ]];   then I_TO=$(step_index "$TO");     (( I_TO >= 0 ))   || { echo "--to $TO: no such step"; exit 1; }; fi
(( I_FROM <= I_TO )) || { echo "--from $FROM comes after --to $TO"; exit 1; }

say "run_prsnet -- $RUN"
echo "disease:   $DISEASE ${DISEASE_NAME:+($DISEASE_NAME)}  build $BUILD"
echo "cohort:    $COHORT  (ancestry rule $PHENO_ANCESTRY)"
echo "outputs:   $PNET"
echo "logs:      $RUN_LOG"
echo "features:  $(echo $PRSNET_THRESHOLDS | wc -w) thresholds x $PRSNET_N_GENES genes, clumped in the target at r2 $PRSNET_CLUMP_R2 / $PRSNET_CLUMP_KB kb"
echo "mode:      $([[ $DRY -eq 1 ]] && echo 'DRY RUN' || echo submit)"

# Preflight: everything the chain head needs, so a missing input fails here
# and not 22 times in the queue.
if (( I_FROM == 0 )); then
    missing=0
    for c in $(seq 1 22); do
        for f in "$PROC/gwas_qc/chr${c}/gwas.QC.txt" "$PROC/gwas_qc/chr${c}/snplist" \
                 "$PROC/mismatch_fix/chr${c}/target_data.a1" "$PROC/target_qc/target_chr${c}.PH.bim"; do
            [[ -f "$f" ]] || { echo "  missing: $f"; missing=$((missing+1)); }
        done
    done
    (( missing == 0 )) || { echo "run steps 01-04 of run_pipeline.sh for this run first ($missing files missing)"; exit 1; }
    [[ -n "$KEEP_LIST" && -n "$PHENO" ]] || { echo "no keep list / phenotype for $RUN -- run 08_make_pheno.sh first"; exit 1; }
    if [[ -x "$PRSNET_VENV/bin/python" && -f "$PRSNET_WINDOWS" && -f "$PRSNET_RANGES" ]]; then
        echo "preflight: 01-04 outputs, the phenotype, the venv and the gene windows are present"
    elif [[ -n "$AFTER" ]]; then
        echo "preflight: 01-04 outputs and the phenotype are present; venv/windows are expected from job $AFTER (setup)"
    else
        echo "missing setup outputs -- run: sbatch --cpus-per-task=2 --mem=8G --time=1:00:00 12_prsnet.sh setup"
        echo "  (or pass --after <that job's id> to queue this chain behind it)"; exit 1
    fi
fi
if (( I_TO == 3 )) && [[ ! -f "$PRSNET_GENE_ORDER" ]]; then
    echo
    echo "  NOTE: $PRSNET_GENE_ORDER is not there yet. 12a-12c will run; 12d will stop at its"
    echo "        input check until the gene list (graph node order) is in place -- see prsnet.md."
    echo "        Re-submit it later with:  DISEASE=$DISEASE COHORT=$COHORT bash run_prsnet.sh --from 12d"
fi
if [[ ! -e "$PNET" && -d "$PROC" ]]; then
    echo
    echo "  NOTE: $PNET does not exist yet. It will hold ~0.5-1 TB (the cohort beds and the"
    echo "        feature store); link it to scratch first, as for ldpred2/prsice:"
    echo "          mkdir -p /nfs/roberts/scratch/pi_sz622/naa79/prs-benchmark/processed/$RUN/prsnet"
    echo "          ln -s /nfs/roberts/scratch/pi_sz622/naa79/prs-benchmark/processed/$RUN/prsnet $PNET"
fi
[[ -n "$AFTER" ]] && echo "after:     job $AFTER (the chain head waits for it with afterok)"

EXPORTS="ALL,DISEASE=$DISEASE,COHORT=$COHORT"
(( RUN_EXPLICIT )) && EXPORTS="$EXPORTS,RUN=$RUN"
JOB_TAG="${DISEASE}_${COHORT}_pnet"
(( RUN_EXPLICIT )) && JOB_TAG="${RUN//\//_}_pnet"

declare -A JID=()

submit() {   # submit <step> <script+args> <array|single> <deps> <deptype> <resources>
    local step=$1 script=$2 kind=$3 deps=$4 dep=$5 res="${6:-}"
    local -a args=()
    local name="${JOB_TAG}_${step}"
    args+=(--job-name="$name" --export="$EXPORTS")
    if [[ "$kind" == array ]]; then
        args+=(--output="$RUN_LOG/${step}_c%a_%A.out" --error="$RUN_LOG/${step}_c%a_%A.err")
    else
        args+=(--output="$RUN_LOG/${step}_%j.out" --error="$RUN_LOG/${step}_%j.err")
    fi
    [[ -n "$res" ]] && args+=($res)
    [[ "$kind" == array && -n "$ARRAY" ]] && args+=(--array="$ARRAY")
    local d ids=""
    IFS=',' read -r -a dlist <<< "$deps"
    for d in ${dlist[@]+"${dlist[@]}"}; do
        [[ -n "${JID[$d]:-}" ]] && ids="${ids:+$ids:}${JID[$d]}"
    done
    [[ -z "$ids" && -n "$AFTER" ]] && { ids="$AFTER"; dep=afterok; }
    [[ -n "$ids" ]] && args+=(--dependency="${dep}:${ids}")

    if (( DRY )); then
        printf '  sbatch %s %s\n' "${args[*]}" "$script"
        JID[$step]="<$step>"
        return 0
    fi
    local jid
    jid=$(sbatch --parsable "${args[@]}" $script)
    printf '  %-6s %-22s job %s%s\n' "$step" "$script" "$jid" "${ids:+  (after $ids)}"
    JID[$step]="$jid"
}

say "Submitting"
for i in "${!STEPS[@]}"; do
    (( i >= I_FROM && i <= I_TO )) || continue
    IFS='|' read -r step script kind deps dep res <<< "${STEPS[$i]}"
    submit "$step" "$script" "$kind" "$deps" "$dep" "$res"
done

if (( DRY )); then
    echo; echo "Nothing submitted. Drop --dry-run to submit this chain."; exit 0
fi
echo
echo "Watch:   squeue -u \$USER --format='%.18i %.30j %.8T %.10M %.20E'"
echo "Cancel:  scancel -u \$USER --name=${JOB_TAG}_12b   (per step)"
echo "Check:   CHR=$CANARY_CHR sbatch --cpus-per-task=4 --mem=16G --time=1:00:00 12_prsnet.sh check   (after 12c on that chromosome)"
echo "Result:  $PNET_GEN/${DISEASE}_${COHORT}/{feats.f32,info.csv}"
