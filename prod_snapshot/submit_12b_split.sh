#!/bin/bash
# =============================================================================
#  submit_12b_split.sh -- 12b (per-gene PLINK --clump) as many small Slurm
#                         jobs instead of one 24 h / 64 GB job per chromosome
#
#  For a queue that will not start a day-long 8-CPU/64 GB job, and a submit
#  limit of ~60 jobs per hour. Per chromosome this submits:
#
#      12b-pack   (run here, on the login node: one awk over gene_manifest.tsv)
#      12b-task   one job ARRAY of a few balanced tasks for the small genes
#                 (4 CPU / 16 GB / 4 h each, default <= 8 tasks)
#      12b-big    one job PER giant window (1 CPU, memory sized from the window,
#                 24 h) -- these are what blew the single-job budget
#      12b-finish afterok on all of the above: writes clump_manifest.tsv
#      12c        afterok on finish (unless --no-12c)
#
#  Output is identical to 12b (same clump_one, same files). Done genes are
#  skipped, so re-running this for a chromosome packs only what is still
#  pending (failed / timed-out genes) -- nothing is recomputed.
#
#      DISEASE=ad COHORT=eur bash submit_12b_split.sh --chr 19 --dry-run
#      DISEASE=ad COHORT=eur bash submit_12b_split.sh --chr 19
#      DISEASE=ad COHORT=eur bash submit_12b_split.sh --chr 1-5
#      DISEASE=ad COHORT=eur bash submit_12b_split.sh --chr 6,7,8-11 --no-12c
#
#  Options
#      --chr LIST       chromosomes: 19 | 1-5 | 6,7,8-11   (default 1-22)
#      --dry-run        pack, print the sbatch commands, submit nothing
#      --no-12c         do not queue 12c behind 12b-finish
#      --small-time T   time limit for a small-gene task   (default 4:00:00)
#      --small-mem M    memory for a small-gene task        (default 16G)
#      --small-cpus N   CPUs for a small-gene task          (default 4)
#      --max-jobs N     stop submitting once N sbatch calls have been made in
#                       this invocation (default 55 -- stay under 60/hour);
#                       chromosomes left over are listed at the end
#
#  SUBMIT COUNT. sbatch calls per chromosome = 1 (small array) + #giants +
#  1 (finish) + 1 (12c). Genome-wide that is ~22*3 + 30 giants ~ 100 calls,
#  so two or three invocations an hour apart (--chr 1-7, 8-15, 16-22). If the
#  hourly limit counts ARRAY ELEMENTS as jobs, lower PRSNET_SMALL_TASKS (config
#  or env) -- 4 tasks per chromosome still finishes in well under an hour.
#  A chromosome whose 12a has not finished is skipped with a message.
# =============================================================================

set -euo pipefail

CHRSPEC="1-22"; DRY=0; DO_12C=1; STIME=3:00:00; SMEM=16G; SCPU=4; MAXJOBS=55
while (( $# )); do
    case "$1" in
        --chr)        CHRSPEC="${2:?--chr needs a list}"; shift ;;
        --dry-run)    DRY=1 ;;
        --no-12c)     DO_12C=0 ;;
        --small-time) STIME="${2:?}"; shift ;;
        --small-mem)  SMEM="${2:?}"; shift ;;
        --small-cpus) SCPU="${2:?}"; shift ;;
        --max-jobs)   MAXJOBS="${2:?}"; shift ;;
        -h|--help)    sed -n '2,42p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1  (try --help)"; exit 1 ;;
    esac
    shift
done

# expand "1-5,7,9-11" -> list
CHRS=()
IFS=',' read -r -a parts <<< "$CHRSPEC"
for p in "${parts[@]}"; do
    if [[ "$p" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        for ((c=${BASH_REMATCH[1]}; c<=${BASH_REMATCH[2]}; c++)); do CHRS+=("$c"); done
    elif [[ "$p" =~ ^[0-9]+$ ]]; then CHRS+=("$p")
    else echo "--chr: cannot parse '$p'"; exit 1; fi
done

SDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SDIR"
export CHR=1
source "$SDIR/config.sh"
command -v sbatch >/dev/null || { echo "sbatch not on PATH -- run this on a login node"; exit 1; }
require_file "$SDIR/12_prsnet.sh" "12_prsnet.sh must sit next to this script"
grep -q 'step_12b_pack' "$SDIR/12_prsnet.sh" || { echo "this 12_prsnet.sh has no 12b-pack step -- update it first"; exit 1; }

RUN_LOG=$ROOT/logs/$RUN; mkdir -p "$RUN_LOG"
EXPORTS="ALL,DISEASE=$DISEASE,COHORT=$COHORT"
(( RUN_EXPLICIT )) && EXPORTS="$EXPORTS,RUN=$RUN"
JOB_TAG="${DISEASE}_${COHORT}_pnet"
(( RUN_EXPLICIT )) && JOB_TAG="${RUN//\//_}_pnet"
BIG_TIME=${PRSNET_BIG_TIME:-1:00:00}

say "submit_12b_split -- $RUN"
echo "chromosomes: ${CHRS[*]}"
echo "small tasks: $SCPU CPU / $SMEM / $STIME, <= ${PRSNET_SMALL_TASKS:-8} per chromosome, ~${PRSNET_TASK_SNPS:-150000} window SNPs each"
echo "giants:      1 CPU / sized per window / $BIG_TIME, one job each"
echo "then:        12b-finish$([[ $DO_12C -eq 1 ]] && echo ' -> 12c')"
echo "mode:        $([[ $DRY -eq 1 ]] && echo 'DRY RUN' || echo "submit (max $MAXJOBS sbatch calls)")"

NSUB=0; JID=""
sb() {   # sb <jobname> <log-stem | - > <extra sbatch args...> -- <script args...>   -> sets JID
    local name=$1 stem=$2; shift 2
    local -a args=(--job-name="$name" --export="$EXPORTS")
    [[ "$stem" != - ]] && args+=(--output="$RUN_LOG/${stem}_%j.out" --error="$RUN_LOG/${stem}_%j.err")
    while [[ "$1" != -- ]]; do args+=("$1"); shift; done; shift
    NSUB=$((NSUB + 1))
    if (( DRY )); then printf '  sbatch %s 12_prsnet.sh %s\n' "${args[*]}" "$*"; JID="dry$NSUB"; return 0; fi
    JID=$(sbatch --parsable "${args[@]}" 12_prsnet.sh "$@"); JID=${JID%%;*}
    printf '  %-34s job %s\n' "$name" "$JID"
}

LEFT=()
for c in "${CHRS[@]}"; do
    export CHR=$c
    PNET_CHR_C=$PNET/chr$c
    if [[ ! -f "$PNET_CHR_C/gene_manifest.tsv" || ! -f "$PNET_CHR_C/target_chr$c.PH.bed" ]]; then
        echo; echo "chr$c: 12a has not finished (no gene_manifest.tsv / cohort .bed) -- skipped"; LEFT+=("$c"); continue
    fi
    # cost of this chromosome in sbatch calls, before packing: known only after pack; pack is cheap, do it
    say "chr$c -- pack"
    bash 12_prsnet.sh 12b-pack | sed 's/^/  /'
    T=$PNET_CHR_C/tasks
    nsmall=0; [[ -f "$T/small.tsv" ]] && nsmall=$(wc -l < "$T/small.tsv")
    nbig=0;   [[ -s "$T/big.tsv"   ]] && nbig=$(wc -l < "$T/big.tsv")
    need=$(( (nsmall > 0) + nbig + 1 + DO_12C ))
    if (( nsmall == 0 && nbig == 0 )); then
        echo "  chr$c: nothing pending -- only finish$([[ $DO_12C -eq 1 ]] && echo '/12c') will be submitted"
        need=$(( 1 + DO_12C ))
    fi
    if (( NSUB + need > MAXJOBS )); then
        echo "  chr$c needs $need sbatch calls; $NSUB of $MAXJOBS used -- leaving it for the next invocation"
        LEFT+=("$c"); continue
    fi

    say "chr$c -- submit"
    deps=""
    if (( nsmall > 0 )); then
        sb "${JOB_TAG}_12bt_c$c" - --array=0-$((nsmall-1)) --cpus-per-task="$SCPU" --mem="$SMEM" --time="$STIME" \
           --output="$RUN_LOG/12bt_c${c}_t%a_%A.out" --error="$RUN_LOG/12bt_c${c}_t%a_%A.err" -- 12b-task
        deps="$JID"
    fi
    if (( nbig > 0 )); then
        while IFS=$'\t' read -r -u 3 g nsnp mem; do
            sb "${JOB_TAG}_12bg_c${c}_$g" "12bg_c${c}_$g" --cpus-per-task=1 --mem="${mem}M" --time="$BIG_TIME" -- 12b-big "$g"
            deps="${deps:+$deps:}$JID"
        done 3< "$T/big.tsv"
    fi
    sb "${JOB_TAG}_12bf_c$c" "12bf_c$c" --cpus-per-task=1 --mem=4G --time=0:30:00 ${deps:+--dependency=afterok:$deps} -- 12b-finish
    fin=$JID
    if (( DO_12C )); then
        sb "${JOB_TAG}_12c" - --array="$c" --cpus-per-task=4 --mem=16G --time=1:00:00 \
           --output="$RUN_LOG/12c_c%a_%A.out" --error="$RUN_LOG/12c_c%a_%A.err" --dependency=afterok:"$fin" -- 12c
    fi
done

echo
echo "sbatch calls this invocation: $NSUB$([[ $DRY -eq 1 ]] && echo ' (dry run)')"
if (( ${#LEFT[@]} )); then
    echo "not submitted (12a pending, or the --max-jobs budget): ${LEFT[*]}"
    echo "  next hour: DISEASE=$DISEASE COHORT=$COHORT bash submit_12b_split.sh --chr $(IFS=,; echo "${LEFT[*]}")"
fi
if (( ! DRY )); then
    echo
    echo "Watch:   squeue -u \$USER --format='%.18i %.40j %.8T %.10M %.20E' | grep 12b"
    echo "Giants:  a 12bg job that dies on memory -> raise PRSNET_BIG_MEM_MAX; on time -> PRSNET_BIG_TIME; then re-run this for that chromosome"
    echo "Resume:  re-running this script for a chromosome packs only the genes without a .clumped"
fi
