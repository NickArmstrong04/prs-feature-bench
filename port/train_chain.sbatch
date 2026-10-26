#!/bin/bash
# =============================================================================
#  port/train_chain.sbatch -- train production's PRS-Net (scripts/slurm/train_prsnet.py,
#  unchanged) on the feats.f32 the chain assembled in scratch, one split per GPU job,
#  with the settings of production's ad/eur baseline run (outputs/prsnet/ad/eur/train:
#  --load ram, 20,000 steps, batch 512, eval batch 128, lr 1e-4, eval every 100, early
#  stop 50, STRING ppi graph), so the AUROC lines up with that results.tsv.
#
#      sbatch --export=ALL,DISEASE=uc,COHORT=eur,CHAIN_OUT=/home/naa79/scratch_pi_sz622/naa79/prs-feature-bench/prsnet_chain/uc_eur \
#             port/train_chain.sbatch                       # 6 jobs, one per split
#      sbatch --array=0 ...                                 # one split (smoke test: add TRAIN_ARGS="--n-steps 500")
#      ... GPU_KEEPALIVE=1 ...                               # see below: keeps the idle-GPU monitor off the load
#
#  Reads  $CHAIN_OUT/genome/<disease>_<cohort>/feats.f32 (+ feats.meta.json, info.csv)
#  Writes $TRAIN_OUT (default $CHAIN_OUT/train): results.tsv (one row per split, appended),
#         split<k>_best.pt, split<k>_test_preds.tsv, split<k>_gates.tsv, log.txt, graph.json
#  Memory: --load ram holds the whole store (n x 19,831 x 11 x 4 bytes: 341 GB at 390,637
#  people); production asked 400G on gpu_b200. Time: ad/uc splits early-stopped after
#  36-90 min of training, but af (32,689 cases) ran to the step cap, 95-100 min, and with
#  six jobs loading the store at once (18-25 min each) two of six hit a 2 h 30 limit; 4 h.
#  Concurrency: the array is throttled to 2 (%2). The load is a serial read of the 341 GB
#  store from scratch and slows with every job reading at once; when four diseases' arrays
#  (24 jobs) became eligible together on 2026-10-25 some loads passed an hour and the
#  cluster cancelled them for not touching the GPU for 60 min. Two at a time loads in
#  2-5 min; a disease then takes 3 waves of ~2 h instead of one.
# =============================================================================
#SBATCH --job-name=pnet_train
#SBATCH --partition=gpu_b200
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --array=0-5%2
#SBATCH --output=logs/train_chain_%A_%a.out
#SBATCH --error=logs/train_chain_%A_%a.err
set -euo pipefail
: "${DISEASE:?}" "${COHORT:?}"
export DISEASE COHORT CHR=1
ROOT=${ROOT:-/home/naa79/project_pi_sz622/naa79/prs-benchmark}
source "$ROOT/scripts/slurm/config.sh"
PY=${PRSNET_PY:-$PRSNET_VENV/bin/python}
OUT=${CHAIN_OUT:-/home/naa79/scratch_pi_sz622/naa79/prs-feature-bench/prsnet_chain/${DISEASE}_${COHORT}}
DS=${DISEASE}_${COHORT}
DATA=${CHAIN_GENOME_PARENT:-$OUT/genome}          # train_prsnet.py wants the parent of <dataset>/
TRAIN_OUT=${TRAIN_OUT:-$OUT/train}
SPLIT=${SLURM_ARRAY_TASK_ID:?run as an array over splits 0-5}
[[ -f "$DATA/$DS/feats.meta.json" ]] || { echo "MISSING: $DATA/$DS/feats.meta.json -- 12d has not finished"; exit 1; }
[[ -f "$PRSNET_GRAPH" ]] || { echo "MISSING: $PRSNET_GRAPH"; exit 1; }
mkdir -p "$TRAIN_OUT"
echo "== PRS-Net train $DS split $SPLIT  features $DATA/$DS  graph $PRSNET_GRAPH  -> $TRAIN_OUT  ($(date))"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "(no GPU visible)"
# GPU_KEEPALIVE=1: the cluster cancels a GPU job that shows no GPU use for 60 min, and
# --load ram reads the whole store from scratch before the first forward pass; that read
# took 2-5 min alone but 60-70 min when a rows array or other users were hammering scratch
# (three splits lost on 2026-10-25/26). This runs a 2048x2048 matmul every 20 s in the
# background for the life of the job (~0.1 % of a B200, ~600 MB) so the monitor sees use
# during the load. It is a workaround of that rule, not a fix of the load -- opt in only.
if [[ "${GPU_KEEPALIVE:-0}" == 1 ]]; then
    "$PY" -I -c 'import time, torch
x = torch.randn(2048, 2048, device="cuda")
while True:
    (x @ x).sum().item(); time.sleep(20)' &
    KEEPALIVE=$!; trap 'kill $KEEPALIVE 2>/dev/null' EXIT
    echo "gpu keepalive pid $KEEPALIVE"
fi
"$PY" -I -W ignore "$ROOT/scripts/slurm/train_prsnet.py" \
    --data-path "$DATA" --dataset "$DS" --graph "$PRSNET_GRAPH" --gene-order "$PRSNET_GENE_ORDER" \
    --out-dir "$TRAIN_OUT" --load ram --splits "$SPLIT" \
    --n-steps 20000 --batch-size 512 --eval-batch 128 --lr 1e-4 --eval-interval 100 --early-stop 50 \
    ${TRAIN_ARGS:-}
echo; echo "== results so far ($TRAIN_OUT/results.tsv):"; cat "$TRAIN_OUT/results.tsv"
echo "== production ad/eur baseline for comparison:"; cat "$ROOT/outputs/prsnet/ad/eur/train/results.tsv" 2>/dev/null || true
