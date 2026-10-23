#!/usr/bin/env bash
# 13_prsnet_train.sh -- train PRS-Net (pure-PyTorch port) on a 12d feature store.
#
#   bash 13_prsnet_train.sh setup                        # one-time: torch into the prsnet venv, run the tests
#   DISEASE=ad COHORT=eur bash 13_prsnet_train.sh test   # port self-test (CPU, ~1 min)
#   DISEASE=ad COHORT=eur bash 13_prsnet_train.sh train [--splits 0] [--load ram] [extra train_prsnet.py args]
#   DISEASE=ad COHORT=eur TAG=identity bash 13_prsnet_train.sh train --graph-mode identity ...
#   bash 13_prsnet_train.sh summary                      # table of every train*/ run: AUROC + embedding leakage
#
# TAG (optional) names the experiment: outputs go to outputs/prsnet/<disease>/<cohort>/train_<TAG>/
# (no TAG = train/, the baseline). LEAK=1 (default) adds the embedding-leakage
# metric using the 14_ld_leakage.py outputs for this disease/cohort when present.
#
# Run `train` inside a GPU allocation, e.g.
#   sbatch --partition=gpu_b200 --qos=normal --gpus=1 --cpus-per-task=8 --mem=400G --time=3:00:00 \
#          --export=ALL,DISEASE=ad,COHORT=eur,TAG=identity --wrap "bash 13_prsnet_train.sh train --splits 0 --load ram --graph-mode identity"
# Memory: --load memmap needs little RAM but reads ~9 TB of random rows per
# split from NFS (the authors' protocol); --load ram needs the whole store
# (~330 GB for AD) and is the fast path; --load ram-f16 halves that.
#
# Reads config.sh for ROOT / RUN / PNET (the PRS-Net block) and the venv.
set -euo pipefail
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SDIR/config.sh" ]]; then source "$SDIR/config.sh"; else source "${SLURM_SUBMIT_DIR:?}/config.sh"; fi
CMD="${1:-}"; shift || true

PY="${PRSNET_PY:-$ROOT/tools/prsnet_venv/bin/python}"
GRAPH="${PRSNET_GRAPH:-$ROOT/tools/PRS-Net/data/ggi_graph.bin}"
ORDER="${PRSNET_GENE_ORDER:-$ROOT/ref/prsnet/gene_order.txt}"
TAG="${TAG:-}"
LEAK="${LEAK:-1}"

case "$CMD" in
  setup)
    # Blackwell (sm_100/sm_120) needs a CUDA >= 12.8 wheel. The PyPI default
    # wheel is usually recent enough; if torch.cuda reports "no kernel image",
    # use: pip install torch --index-url https://download.pytorch.org/whl/cu128
    "$PY" -m pip install --quiet --upgrade torch scipy
    "$PY" - <<'EOF'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
EOF
    "$PY" -I -W ignore "$SDIR/test_prsnet_torch.py" "$GRAPH"
    ;;
  test)
    "$PY" -I -W ignore "$SDIR/test_prsnet_torch.py" "$GRAPH"
    ;;
  train)
    : "${DISEASE:?set DISEASE}" "${COHORT:?set COHORT}"
    DATA="$PNET/genome"; DS="${DISEASE}_${COHORT}"
    [[ -f "$DATA/$DS/feats.meta.json" ]] || { echo "MISSING: $DATA/$DS/feats.meta.json -- run 12d first"; exit 1; }
    OUT="$ROOT/outputs/prsnet/$DISEASE/$COHORT/train${TAG:+_$TAG}"
    mkdir -p "$OUT"
    echo "features: $DATA/$DS"; echo "graph:    $GRAPH"; echo "out:      $OUT"
    nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "(no GPU visible -- will run on CPU)"
    ORD=(); [[ -f "$ORDER" ]] && ORD=(--gene-order "$ORDER")
    LK=()
    LDD="$ROOT/outputs/prsnet/$DISEASE/$COHORT/ld_leakage"
    if [[ "$LEAK" == "1" && -f "$LDD/pairs.tsv.gz" && -f "$LDD/sampled_rows.txt" ]]; then
      LK=(--leak-pairs "$LDD/pairs.tsv.gz" --leak-rows "$LDD/sampled_rows.txt")
      echo "leakage:  $LDD (pairs + sampled rows)"
    fi
    # LD-edge modes default to the same pairs file unless --ld-pairs is given explicitly
    LDP=()
    if [[ " $* " == *" --graph-mode ppi+ld "* || " $* " == *" --graph-mode ld "* ]] && [[ " $* " != *" --ld-pairs "* ]]; then
      [[ -f "$LDD/pairs.tsv.gz" ]] || { echo "MISSING: $LDD/pairs.tsv.gz -- run 14_ld_leakage.py first"; exit 1; }
      LDP=(--ld-pairs "$LDD/pairs.tsv.gz")
    fi
    "$PY" -I -W ignore "$SDIR/train_prsnet.py" --data-path "$DATA" --dataset "$DS" --graph "$GRAPH" \
        --out-dir "$OUT" "${ORD[@]}" "${LK[@]}" "${LDP[@]}" "$@"
    echo; cat "$OUT/results.tsv"
    ;;
  summary)
    # One line per run directory: mean +/- sd test AUROC over the splits present, learned LD weight,
    # and the embedding-level leakage (gnn layer, mean |r|) for adjacent-no-shared vs ppi_far, averaged over splits.
    "$PY" - "$ROOT/outputs/prsnet" <<'EOF'
import sys, os, glob, json, numpy as np
root = sys.argv[1]
print(f"{'run':40s}{'mode':10s}{'n':>3}{'test AUROC':>20}{'ld_w':>8}  gnn-layer mean|r|: adj_noshared  adj_shared  ppi_far  random   | enc adj_noshared")
for res in sorted(glob.glob(os.path.join(root, "*", "*", "train*", "results.tsv"))):
    d = os.path.dirname(res)
    rows = [l.split("\t") for l in open(res).read().strip().split("\n")[1:]]
    if not rows: continue
    t = np.array([float(r[3]) for r in rows])
    ldw = [float(r[7]) for r in rows if len(r) > 7 and r[7] not in ("nan", "")]
    mode = json.load(open(os.path.join(d, "graph.json")))["mode"] if os.path.exists(os.path.join(d, "graph.json")) else "ppi"
    leak = {}
    for f in glob.glob(os.path.join(d, "split*_embed_leakage.tsv")):
        for l in open(f).read().strip().split("\n")[1:]:
            p = l.split("\t"); leak.setdefault((p[0], p[1]), []).append(float(p[4]))
    def g(layer, cls):
        v = [np.mean(x) for k, x in leak.items() if k[0] == layer and k[1] == cls]
        return f"{v[0]:.3f}" if v else "   -  "
    name = os.path.relpath(d, root)
    sd = f"{t.std(ddof=1):.4f}" if len(t) > 1 else "  -   "
    print(f"{name:40s}{mode:10s}{len(t):>3}{t.mean():>12.4f} +/- {sd}{(f'{np.mean(ldw):+.3f}' if ldw else '   -  '):>8}"
          f"  {g('gnn','adjacent_no_shared_snp'):>31}  {g('gnn','adjacent_shared_snp'):>10}  {g('gnn','ppi_far'):>7}  {g('gnn','random_diff_chr'):>6}   | {g('enc','adjacent_no_shared_snp')}")
EOF
    ;;
  *)
    sed -n '2,22p' "$0"; exit 1 ;;
esac
