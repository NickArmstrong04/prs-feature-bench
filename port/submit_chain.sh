#!/bin/bash
# =============================================================================
#  port/submit_chain.sh -- submit port/prsnet_chain.sbatch as 22 separately
#  sized jobs (one per chromosome) instead of one uniform array, so each job
#  asks for what its chromosome needs and backfill can start most of them at
#  once. Cheap enough for the login node: 22 `wc -l` and one awk per manifest.
#
#      bash port/submit_chain.sh <disease> <cohort> [--dry-run] [--chr 1,2,19]
#             [--no-12d | --12d-only] [--parts 8] [--cmp /prod/feats.f32] [--genome-dir DIR]
#             [--snps-only] [--scale 1.5] [--after JOBID[:JOBID]]
#
#  --after adds afterok dependencies on already-running jobs (e.g. --12d-only --after <chr1 rerun>).
#  --scale multiplies every chromosome job's time request (default 1.0; the model's
#  1.3x + 5 min margin was ~15 % high on the big AD chromosomes and ran out on uc chr1).
#
#  A disease production never ran PRS-Net on (af, ms, uc ...) has no gene files under
#  $PNET/chr<k>; the sizing needs them, so first `--snps-only` submits STEP=snps as a
#  22-task array (1 CPU, well under a minute each, 5 min requested) writing them under $CHAIN_OUT/chr<k>, and once it
#  is done a plain submit finds them there. $CHAIN_OUT in the environment is passed on
#  (--export=ALL), so set it explicitly to keep runs apart; BENCH is this checkout.
#
#  After the chromosome jobs it chains 12d as three dependent steps of
#  port/assemble_rows.py (prep -> rows array of --parts tasks -> finish), so one
#  command takes a cohort from nothing to feats.f32; --12d-only runs just those
#  three on an existing run dir, --cmp makes finish cmp the result against a
#  file, --genome-dir puts the output elsewhere than $CHAIN_OUT/genome/<d>_<c>.
#  Rows jobs: production's single 12d took 1737 s / 32 GB (memmap pages); a part
#  holds one 5000-sample block (4.4 GB) plus the chunk slices it reads, so 12 GB,
#  2 CPUs, time 2 x 1.3 x 1737/parts + 5 min until the first run calibrates it.
#
#  Sizing model (calibrated on chr19 at 380,962 people, 4 CPUs, README s11:
#  12a 67 s, 12b 475 s, 12c 361 s for 156,485 window SNPs and 1,525 genes,
#  ERICH4 113,666 SNPs = 584 s of one plink in production):
#      bed    = 67 s   x W/156485 x 4/C
#      clump  = max( sum_g t_g / C , max_g t_g ),  t_g = 1 s + 5.1 ms x SNPs_g
#      score  = 361 s  x W/156485            (write-bound; CPUs barely matter)
#      time   = 1.3 x (bed + clump + score) + 5 min, rounded up to 5 min; 1.5x + 10 min when a gene
#               has > 100k SNPs (uc/eur chr1 used 80 of its 85 min on 2026-10-24: the clump and score
#               stages ran 1.5x slower than ad's chr1 the day before on the same sizes)
#      CPUs   = 8 if W > 300k or a gene > 200k SNPs, else 4
#      mem    = 16 GB at 4 CPUs, 24 GB at 8 (each plink capped at 2.5 GB; the first run
#               requested 32 GB for giant chromosomes and MaxRSS simply equalled the request)
#  W = window SNPs (window.snplist), from production's $PNET/chr<k>/ files.
#  Afterwards, calibrate with:  sacct -j <ids> --format=JobID%14,JobName,Elapsed,TotalCPU,MaxRSS,State
# =============================================================================
set -euo pipefail
DISEASE=${1:?disease}; COHORT=${2:?cohort}; shift 2
DRY=0; CHRS=$(seq 1 22 | tr '\n' ' '); DO_CHR=1; DO_12D=1; PARTS=8; CMP=""; GDIR=""; SNPS=0; SCALE=1.0; AFTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --chr) CHRS=${2//,/ }; shift 2 ;;
        --no-12d) DO_12D=0; shift ;;
        --12d-only) DO_CHR=0; shift ;;
        --parts) PARTS=$2; shift 2 ;;
        --cmp) CMP=$2; shift 2 ;;
        --genome-dir) GDIR=$2; shift 2 ;;
        --snps-only) SNPS=1; shift ;;
        --scale) SCALE=$2; shift 2 ;;
        --after) AFTER=$2; shift 2 ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
done
(( DO_CHR == 1 || DO_12D == 1 )) || { echo "--no-12d and --12d-only together leave nothing to submit"; exit 1; }
ROOT=${ROOT:-/home/naa79/project_pi_sz622/naa79/prs-benchmark}
BENCH=$(cd "$(dirname "$0")/.." && pwd)     # this checkout, whatever the shell exports (a worktree submits its own scripts)
PNET=$ROOT/data/processed/$DISEASE/$COHORT/prsnet
OUT=${CHAIN_OUT:-/home/naa79/scratch_pi_sz622/naa79/prs-feature-bench/prsnet_chain/${DISEASE}_${COHORT}}
mkdir -p "$BENCH/logs"
cd "$BENCH"
echo "run dir: $OUT   bench: $BENCH"
if (( SNPS == 1 )); then
    echo "gene files -> $OUT/chr<k>/{genes,gene_manifest.tsv,window.snplist}  (STEP=snps, array 1-22, 1 CPU, 8G, 5 min)"
    (( DRY == 1 )) && exit 0
    J=$(sbatch --parsable --job-name="pnet_snps" --array=1-22 --cpus-per-task=1 --mem=8G --time=0:05:00 \
               --export=ALL,DISEASE="$DISEASE",COHORT="$COHORT",STEP=snps,BENCH="$BENCH" port/prsnet_chain.sbatch)
    echo "submitted ${J%%;*}; when COMPLETED (sacct -j ${J%%;*} -X --format=JobID,Elapsed,State), run the plain submit"
    exit 0
fi
printf '%-5s %9s %6s %9s %5s %5s %6s %6s %6s %8s %7s %s\n' chr windowSNPs genes maxgene CPUs mem bed_s clump_s scor_s time jobid note
TOT_EST=0; IDS=()
(( DO_CHR == 1 )) || CHRS=""
for k in $CHRS; do
    M=$PNET/chr$k/gene_manifest.tsv; WF=$PNET/chr$k/window.snplist                       # production's (ad)
    [[ -f "$M" && -f "$WF" ]] || { M=$OUT/chr$k/gene_manifest.tsv; WF=$OUT/chr$k/window.snplist; }   # or --snps-only's
    [[ -f "$M" && -f "$WF" ]] || { printf '%-5s %s\n' "$k" "no gene_manifest.tsv / window.snplist under $PNET/chr$k or $OUT/chr$k -- run --snps-only first; skipped"; continue; }
    W=$(wc -l < "$WF")
    read -r G MAXG TSUM <<< "$(awk 'NR>1 && $4>0 {g++; t+=1+0.0051*$4; if ($4>m) m=$4} END{printf "%d %d %.0f", g, m, t}' "$M")"
    read -r C MEM BED CLUMP SCORE TSEC TIME NOTE <<< "$(awk -v W="$W" -v MAXG="$MAXG" -v TSUM="$TSUM" -v S="$SCALE" 'BEGIN{
        C = (W > 300000 || MAXG > 200000) ? 8 : 4
        mem = (C == 8) ? "24G" : "16G"      # plink capped at 2.5 GB each (CHAIN_PLINK_MB); python stages < 10 GB
        bed = 67 * W / 156485 * 4 / C
        tmax = 1 + 0.0051 * MAXG
        clump = TSUM / C; if (tmax > clump) clump = tmax
        score = 361 * W / 156485
        # giant chromosomes: 1.5x + 10 min (uc chr1 2026-10-24: 12b 3282 s vs 2217 est, 12c 1362 vs 1167,
        # 80 min of an 85 min request); the rest keep 1.3x + 5 min (all 21 finished at 25-75 % of theirs)
        if (MAXG > 100000) tsec = (1.5 * (bed + clump + score) + 600) * S
        else               tsec = (1.3 * (bed + clump + score) + 300) * S
        tmin = int((tsec + 299) / 300) * 5
        note = (MAXG > 20000) ? "giant genes" : "-"
        printf "%d %s %.0f %.0f %.0f %.0f %02d:%02d:00 %s", C, mem, bed, clump, score, tsec, int(tmin / 60), tmin % 60, note }')"
    TOT_EST=$((TOT_EST + TSEC))
    JOB=dry
    if (( DRY == 0 )); then
        JOB=$(sbatch --parsable --job-name="pnet_c$k" --array="$k" --cpus-per-task="$C" --mem="$MEM" --time="$TIME" \
                     --export=ALL,DISEASE="$DISEASE",COHORT="$COHORT",BENCH="$BENCH" port/prsnet_chain.sbatch)
        JOB=${JOB%%;*}; IDS+=("${JOB%%_*}")
    fi
    printf '%-5s %9d %6d %9d %5d %5s %6d %6d %6d %8s %7s %s\n' "$k" "$W" "$G" "$MAXG" "$C" "$MEM" "$BED" "$CLUMP" "$SCORE" "$TIME" "${JOB%%_*}" "$NOTE"
done
(( DO_CHR == 1 )) && printf 'estimated compute if run one after another: %d min; the longest single job bounds the wall when they run together\n' $((TOT_EST / 60))

# --- 12d as prep -> rows[0..PARTS-1] -> finish, each waiting on the one before --------
if (( DO_12D == 1 )); then
    EXP="ALL,DISEASE=$DISEASE,COHORT=$COHORT,CHAIN_PARTS=$PARTS,BENCH=$BENCH"
    [[ -n "$GDIR" ]] && EXP+=",CHAIN_GENOME_DIR=$GDIR"
    [[ -n "$CMP" ]] && EXP+=",CHAIN_CMP=$CMP"
    ALLDEP=("${IDS[@]}"); [[ -n "$AFTER" ]] && ALLDEP+=(${AFTER//:/ })
    DEP=""; (( ${#ALLDEP[@]} > 0 )) && DEP="--dependency=afterok:$(IFS=:; echo "${ALLDEP[*]}")"
    RSEC=$(awk -v P="$PARTS" 'BEGIN{printf "%d", 2 * 1.3 * 1737 / P + 300}')
    RMIN=$(( (RSEC + 299) / 300 * 5 ))
    RTIME=$(printf '%02d:%02d:00' $((RMIN / 60)) $((RMIN % 60)))
    FTIME=0:10:00; [[ -n "$CMP" ]] && FTIME=0:30:00
    sub() {   # sub <name> <array|-> <cpus> <mem> <time> <dependency|-> <STEP>
        local arr=(--array="$2"); [[ "$2" == - ]] && arr=(--array=0)   # the header says --array=19
        local dep=(); [[ "$6" != - ]] && dep=("$6")
        if (( DRY == 1 )); then echo "dry"; return; fi
        local j; j=$(sbatch --parsable --job-name="$1" "${arr[@]}" --cpus-per-task="$3" --mem="$4" --time="$5" "${dep[@]}" \
                            --export="$EXP,STEP=$7" port/prsnet_chain.sbatch)
        echo "${j%%;*}"
    }
    JP=$(sub pnet_prep   -                1 4G  0:10:00 "${DEP:--}"                 prep);   JP=${JP%%_*}
    JR=$(sub pnet_rows   "0-$((PARTS-1))" 2 12G "$RTIME" "--dependency=afterok:$JP" rows);  JR=${JR%%_*}
    JF=$(sub pnet_finish -                1 2G  "$FTIME" "--dependency=afterok:$JR" finish); JF=${JF%%_*}
    printf '12d: prep %s (1 CPU, 4G, 10 min) -> rows %s (%d x 2 CPU, 12G, %s) -> finish %s (1 CPU, 2G, %s%s)\n' \
        "$JP" "$JR" "$PARTS" "$RTIME" "$JF" "$FTIME" "${CMP:+, cmp vs $CMP}"
    IDS+=("$JP" "$JR" "$JF")
fi
if (( DRY == 0 && ${#IDS[@]} > 0 )); then
    echo "watch:  squeue -u \$USER -o '%.12i %.10j %.4C %.8m %.10M %.10l %.3t %R'"
    echo "after:  sacct -j $(IFS=,; echo "${IDS[*]}") --format=JobID%14,JobName,Elapsed,TotalCPU,MaxRSS,ReqMem,State | grep -v '^[0-9]*_[0-9]*\.'"
fi
