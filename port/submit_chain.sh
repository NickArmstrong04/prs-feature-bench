#!/bin/bash
# =============================================================================
#  port/submit_chain.sh -- submit port/prsnet_chain.sbatch as 22 separately
#  sized jobs (one per chromosome) instead of one uniform array, so each job
#  asks for what its chromosome needs and backfill can start most of them at
#  once. Cheap enough for the login node: 22 `wc -l` and one awk per manifest.
#
#      bash port/submit_chain.sh <disease> <cohort> [--dry-run] [--chr 1,2,19]
#
#  Sizing model (calibrated on chr19 at 380,962 people, 4 CPUs, README s11:
#  12a 67 s, 12b 475 s, 12c 361 s for 156,485 window SNPs and 1,525 genes,
#  ERICH4 113,666 SNPs = 584 s of one plink in production):
#      bed    = 67 s   x W/156485 x 4/C
#      clump  = max( sum_g t_g / C , max_g t_g ),  t_g = 1 s + 5.1 ms x SNPs_g
#      score  = 361 s  x W/156485            (write-bound; CPUs barely matter)
#      time   = 1.3 x (bed + clump + score) + 5 min, rounded up to 5 min (the 1.5x + 10 min
#               first used asked for ~2x the actual on 2026-10-23: chr1 est 3767 s, actual 3229 s)
#      CPUs   = 8 if W > 300k or a gene > 200k SNPs, else 4
#      mem    = 16 GB at 4 CPUs, 24 GB at 8 (each plink capped at 2.5 GB; the first run
#               requested 32 GB for giant chromosomes and MaxRSS simply equalled the request)
#  W = window SNPs (window.snplist), from production's $PNET/chr<k>/ files.
#  Afterwards, calibrate with:  sacct -j <ids> --format=JobID%14,JobName,Elapsed,TotalCPU,MaxRSS,State
# =============================================================================
set -euo pipefail
DISEASE=${1:?disease}; COHORT=${2:?cohort}; shift 2
DRY=0; CHRS=$(seq 1 22 | tr '\n' ' ')
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --chr) CHRS=${2//,/ }; shift 2 ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
done
ROOT=${ROOT:-/home/naa79/project_pi_sz622/naa79/prs-benchmark}
BENCH=${BENCH:-$(cd "$(dirname "$0")/.." && pwd)}
PNET=$ROOT/data/processed/$DISEASE/$COHORT/prsnet
[[ -d "$PNET" ]] || { echo "no $PNET -- run 12a's snps step first (or point ROOT elsewhere)"; exit 1; }
mkdir -p "$BENCH/logs"
cd "$BENCH"
printf '%-5s %9s %6s %9s %5s %5s %6s %6s %6s %8s %7s %s\n' chr windowSNPs genes maxgene CPUs mem bed_s clump_s scor_s time jobid note
TOT_EST=0; IDS=()
for k in $CHRS; do
    M=$PNET/chr$k/gene_manifest.tsv; WF=$PNET/chr$k/window.snplist
    [[ -f "$M" && -f "$WF" ]] || { printf '%-5s %s\n' "$k" "no gene_manifest.tsv / window.snplist under $PNET/chr$k -- skipped"; continue; }
    W=$(wc -l < "$WF")
    read -r G MAXG TSUM <<< "$(awk 'NR>1 && $4>0 {g++; t+=1+0.0051*$4; if ($4>m) m=$4} END{printf "%d %d %.0f", g, m, t}' "$M")"
    read -r C MEM BED CLUMP SCORE TSEC TIME NOTE <<< "$(awk -v W="$W" -v MAXG="$MAXG" -v TSUM="$TSUM" 'BEGIN{
        C = (W > 300000 || MAXG > 200000) ? 8 : 4
        mem = (C == 8) ? "24G" : "16G"      # plink capped at 2.5 GB each (CHAIN_PLINK_MB); python stages < 10 GB
        bed = 67 * W / 156485 * 4 / C
        tmax = 1 + 0.0051 * MAXG
        clump = TSUM / C; if (tmax > clump) clump = tmax
        score = 361 * W / 156485
        tsec = 1.3 * (bed + clump + score) + 300
        tmin = int((tsec + 299) / 300) * 5
        note = (MAXG > 20000) ? "giant genes" : "-"
        printf "%d %s %.0f %.0f %.0f %.0f %02d:%02d:00 %s", C, mem, bed, clump, score, tsec, int(tmin / 60), tmin % 60, note }')"
    TOT_EST=$((TOT_EST + TSEC))
    JOB=dry
    if (( DRY == 0 )); then
        JOB=$(sbatch --parsable --job-name="pnet_c$k" --array="$k" --cpus-per-task="$C" --mem="$MEM" --time="$TIME" \
                     --export=ALL,DISEASE="$DISEASE",COHORT="$COHORT" port/prsnet_chain.sbatch)
        JOB=${JOB%%;*}; IDS+=("${JOB%%_*}")
    fi
    printf '%-5s %9d %6d %9d %5d %5s %6d %6d %6d %8s %7s %s\n' "$k" "$W" "$G" "$MAXG" "$C" "$MEM" "$BED" "$CLUMP" "$SCORE" "$TIME" "${JOB%%_*}" "$NOTE"
done
printf 'estimated compute if run one after another: %d min; the longest single job bounds the wall when they run together\n' $((TOT_EST / 60))
if (( DRY == 0 && ${#IDS[@]} > 0 )); then
    echo "watch:  squeue -u \$USER -o '%.12i %.10j %.4C %.8m %.10M %.10l %.3t %R'"
    echo "after:  sacct -j $(IFS=,; echo "${IDS[*]}") --format=JobID%14,JobName,Elapsed,TotalCPU,MaxRSS,ReqMem,State | grep -v '^[0-9]*_[0-9]*\.'"
fi
