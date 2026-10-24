#!/bin/bash
# =============================================================================
#  port/full_run_prep.sh -- before a clean genome-wide run of the chain that is
#  then trained and compared with production's AD result:
#
#      bash port/full_run_prep.sh readiness  <cohort> af ms uc    # which diseases have the C+T outputs
#                                                                  # the chain reads (no production
#                                                                  # PRS-Net run is needed)
#      bash port/full_run_prep.sh inventory  <disease> <cohort>   # read-only: every input the chain
#                                                                  # and the trainer need, production's
#                                                                  # training results, disk; login-node cheap
#      bash port/full_run_prep.sh clean      <disease> <cohort>   # du of what a clean run would remove
#      bash port/full_run_prep.sh clean      <disease> <cohort> --yes   # remove it
#
#  clean removes only the chain's SCRATCH outputs for this disease/cohort
#  (CHAIN_OUT: chr*/ scores + clumps + bim/fam, and genome/ = the chain's own
#  feats.f32 copy). It never touches production's data/processed tree, runs/,
#  results.tsv, or the goldens under runs/ct_full (CLAUDE.md).
# =============================================================================
set -uo pipefail
CMD=${1:?readiness|inventory|clean}
if [[ "$CMD" == readiness ]]; then
    COH=${2:?cohort}; shift 2; DIS="${*:?diseases}"
    DISEASE=ad; COHORT=$COH          # any disease that exists: the top-level source only needs CONF_DIR
else
    DISEASE=${2:?disease}; COHORT=${3:?cohort}; YES=${4:-}
fi
ROOT=${ROOT:-/home/naa79/project_pi_sz622/naa79/prs-benchmark}
BENCH=${BENCH:-$(cd "$(dirname "$0")/.." && pwd)}
export DISEASE COHORT CHR=19
source "$ROOT/scripts/slurm/config.sh" || { echo "cannot source $ROOT/scripts/slurm/config.sh"; exit 1; }
OUT=${CHAIN_OUT:-/home/naa79/scratch_pi_sz622/naa79/prs-feature-bench/prsnet_chain/${DISEASE}_${COHORT}}
SCRATCH=$(dirname "$(dirname "$OUT")")

chk() {  # chk <label> <path>  -> ok/MISSING with size
    local lab=$1 p=$2
    if [[ -f "$p" ]]; then printf '  %-26s ok       %10s  %s\n' "$lab" "$(du -h "$p" | cut -f1)" "$p"
    elif [[ -d "$p" ]]; then printf '  %-26s ok (dir) %10s  %s\n' "$lab" "$(ls "$p" | wc -l) files" "$p"
    else printf '  %-26s MISSING  %10s  %s\n' "$lab" "" "$p"; fi
}

case "$CMD" in
readiness)
    # bash port/full_run_prep.sh readiness <cohort> af ms uc ...  -- which diseases the chain can run
    # on now: the C+T outputs it reads per chromosome (gwas.QC.txt, SNP.pvalue, target_data.a1,
    # target_chr<k>.PH.bim for `snps`), the keep list / phenotype, and any production gene files.
    echo "disease files: $(cd "$CONF_DIR/diseases" && ls *.sh | grep -v '^_' | sed 's/\.sh$//' | tr '\n' ' ')"
    for d in $DIS; do
        echo; echo "== $d/$COH"
        [[ -f "$CONF_DIR/diseases/$d.sh" ]] || { echo "  no diseases/$d.sh"; continue; }
        out=$(DISEASE=$d COHORT=$COH CHR=19 bash -c 'source "'"$ROOT"'/scripts/slurm/config.sh" >/dev/null 2>&1 || { echo CONFIG_FAIL; exit 0; }
            [[ -f "${KEEP_LIST:-/x}" ]] && echo "  keep list: $(wc -l < "$KEEP_LIST") people  $KEEP_LIST"
            [[ -f "${PHENO:-/x}" ]] && echo "  pheno:     $(wc -l < "$PHENO") lines, cases $(awk "NR>1 && \$NF==2" "$PHENO" | wc -l), header: $(head -1 "$PHENO")  $PHENO"
            miss=""
            for k in $(seq 1 22); do
                export CHR=$k; source "'"$ROOT"'/scripts/slurm/config.sh" >/dev/null 2>&1
                for f in "$GQC/gwas.QC.txt" "$GQC/SNP.pvalue" "$MIS/target_data.a1" "$TARGET.PH.bim"; do
                    [[ -f "$f" ]] || miss+=" chr$k:$(basename "$f")"
                done
            done
            if [[ -z "$miss" ]]; then echo "  C+T outputs: all 22 chromosomes present (gwas.QC.txt, SNP.pvalue, target_data.a1, target_chr<k>.PH.bim)"
            else echo "  C+T outputs MISSING:$miss"; fi
            n=0; for k in $(seq 1 22); do [[ -f "$PNET/chr$k/gene_manifest.tsv" ]] && n=$((n+1)); done
            echo "  production gene files: $n/22 chromosomes under $PNET (0 = the chain makes them with 12_prsnet.py snps)"
            echo "  gene windows: $([[ -f "$PRSNET_WINDOWS" ]] && echo ok || echo MISSING) $PRSNET_WINDOWS"
            ls -d "'"$ROOT"'"/outputs/prsnet/$DISEASE/$COHORT/train* 2>/dev/null | sed "s/^/  prior training: /"
            ls "'"$ROOT"'"/outputs/*/$DISEASE/$COHORT/ 2>/dev/null | head -3 | sed "s/^/  other method outputs: /"')
        echo "$out"
    done
    echo
    echo "== scratch and project disk"
    du -sh "$SCRATCH"/* 2>/dev/null; df -h "$SCRATCH" "$ROOT" | awk 'NR==1 || /\//'
    ;;
inventory)
    echo "== chain inputs ($DISEASE/$COHORT), from config.sh"
    chk "raw bed (chr19 example)" "$GENO_PREFIX.bed"
    chk "gwas.QC.txt (chr19)" "$GQC/gwas.QC.txt"
    chk "SNP.pvalue (chr19)" "$GQC/SNP.pvalue"
    chk "target_data.a1 (chr19)" "$MIS/target_data.a1"
    chk "keep list" "${KEEP_LIST:-<unset>}"
    chk "phenotype" "${PHENO:-<unset>}"
    chk "gene_order.txt" "$PRSNET_GENE_ORDER"
    chk "graph (PRSNET_GRAPH)" "$PRSNET_GRAPH"
    chk "graph .bin (13_ default)" "$ROOT/tools/PRS-Net/data/ggi_graph.bin"
    chk "venv python" "$PRSNET_VENV/bin/python"
    chk "12_prsnet.py" "$ROOT/scripts/slurm/12_prsnet.py"
    chk "train_prsnet.py" "$ROOT/scripts/slurm/train_prsnet.py"
    echo "  torch in venv:            $("$PRSNET_VENV/bin/python" -c 'import torch;print(torch.__version__)' 2>/dev/null || echo MISSING)"
    echo
    echo "== production gene files per chromosome (gene_manifest.tsv / window.snplist / clumps)"
    for k in $(seq 1 22); do
        d=$PNET/chr$k
        printf '  chr%-2s manifest %s  window %s  clumps %s\n' "$k" \
            "$([[ -f $d/gene_manifest.tsv ]] && echo ok || echo MISSING)" \
            "$([[ -f $d/window.snplist ]] && echo ok || echo MISSING)" \
            "$([[ -d $d/clumps ]] && ls "$d/clumps" | wc -l || echo none)"
    done
    echo
    echo "== production feature store and training results (the 'old approach')"
    chk "prod feats.f32" "$PNET/genome/${DISEASE}_${COHORT}/feats.f32"
    chk "prod feats.meta.json" "$PNET/genome/${DISEASE}_${COHORT}/feats.meta.json"
    chk "prod info.csv" "$PNET/genome/${DISEASE}_${COHORT}/info.csv"
    chk "ld_leakage pairs" "$ROOT/outputs/prsnet/$DISEASE/$COHORT/ld_leakage/pairs.tsv.gz"
    for d in "$ROOT/outputs/prsnet/$DISEASE/$COHORT"/train*; do
        [[ -d "$d" ]] || continue
        echo "  -- $d"
        [[ -f "$d/results.tsv" ]] && sed 's/^/     /' "$d/results.tsv" || echo "     (no results.tsv)"
        [[ -f "$d/graph.json" ]] && echo "     graph.json: $(tr -d '\n' < "$d/graph.json")"
        [[ -f "$d/log.txt" ]] && { echo "     log.txt head:"; head -5 "$d/log.txt" | sed 's/^/       /'; }
    done
    echo
    echo "== training jobs in sacct (how production's train was submitted)"
    sacct -u "$USER" -S 2026-09-01 --format=JobID%14,JobName%22,Partition,AllocTRES%40,Elapsed,State -X 2>/dev/null \
        | grep -i 'gpu' | awk 'NR<=20' || echo "  (sacct unavailable)"
    echo
    echo "== scratch state (what clean would remove) and disk"
    [[ -d "$OUT" ]] && du -sh "$OUT"/chr* "$OUT"/genome 2>/dev/null | sort -k2 || echo "  $OUT: absent"
    du -sh "$SCRATCH"/* 2>/dev/null
    df -h "$SCRATCH" "$ROOT" | awk 'NR==1 || /\//'
    echo
    echo "== bench runs/ bulk in scratch (kept: runs/ entries are the experiment log)"
    ls -la "$BENCH/runs" 2>/dev/null | grep -- '->' || echo "  (no symlinked runs)"
    ;;
clean)
    [[ -d "$OUT" ]] || { echo "nothing to clean: $OUT absent"; exit 0; }
    echo "== would remove under $OUT:"
    du -sh "$OUT"/chr* "$OUT"/genome 2>/dev/null | sort -k2
    du -sh "$OUT"
    if [[ "$YES" != --yes ]]; then echo; echo "dry run; add --yes to remove"; exit 0; fi
    rm -rf "$OUT"/chr[0-9]* "$OUT"/genome
    echo "removed; left: $(ls -A "$OUT" | tr '\n' ' ')"
    df -h "$SCRATCH" | awk 'NR==1 || /\//'
    ;;
*) echo "unknown command $CMD"; exit 1 ;;
esac
