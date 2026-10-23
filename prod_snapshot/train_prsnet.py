#!/usr/bin/env python3
"""
train_prsnet.py -- the authors' scripts/train.py + src/trainer.py, DGL-free,
reading the 12d feature store (feats.f32 memmap) instead of 381k .npy files.

  python3 train_prsnet.py --data-path $PNET/genome --dataset ad_eur \
          --graph tools/PRS-Net/data/ggi_graph.bin --out-dir outputs/prsnet/ad/eur/train

Protocol kept exactly as in train.py / trainer.py (defaults):
  - set_random_seed(22); six 8:1:1 splits from generate_splits (seeds 42-47,
    cases and controls split separately, copied verbatim)
  - train: balanced sampling with replacement (half cases, half controls per
    draw), batch 512, "epoch" of 10,000,000 draws -> 19,531 steps (their loop
    runs len(train_loader) steps and the n_steps=20000 break is never reached)
  - AdamW lr 1e-4, weight_decay 0, BCEWithLogitsLoss
  - validate every 100 steps; test AUROC is recorded at the best-validation
    step; early stop after 50 evaluations without improvement
  - model: PRSNet() defaults (d_hidden 64, 1 GIN layer, 11 thresholds)
What differs: the feature reader (one memmap / RAM array, rows gathered by
index -- same numbers as their per-sample .npy files, 12d wrote both layouts
from the same array), AUROC via the Mann-Whitney rank statistic instead of
torchmetrics (identical for ties-free scores; ties handled by mid-ranks like
torchmetrics/sklearn), and no DataLoader workers (the gather is a single
fancy-index on an array, workers would only copy it).

LD experiments (Oct 19 2026, ld_experiments.md) -- everything below is off by
default and the default run is the authors' model unchanged:
  --graph-mode ppi        the graph file as given (default)
               identity   self-loops only (E1): the GIN becomes a per-gene MLP
               shuffled   degree-preserving rewiring of the PPI set (E2), --shuffle-seed
               ppi+ld     PPI plus a second, LD edge type with a learnable weight (E3)
               ld         LD edges only, identity in place of PPI (E4)
  --ld-pairs pairs.tsv.gz from 14_ld_leakage.py; adjacent pairs become the LD
               edges, weighted by --ld-weight {signed|abs|binary} of the r in
               --ld-col (default r_P<=1), kept if |r| >= --ld-min-abs-r; --ld-init
  --leak-pairs pairs.tsv.gz + --leak-rows sampled_rows.txt: after each split,
               the embedding-level leakage metric (prsnet_ld.embedding_leakage)
               -> split<k>_embed_leakage.tsv and a block in log.txt
  (--gene-order is required for --ld-pairs / --leak-pairs: the pairs name genes.)

Outputs in --out-dir:
  results.tsv            split  seed  best_val_auroc  test_auroc  best_step  steps_run  minutes  ld_weight
  split<k>_best.pt       state_dict at the best-validation step (loads into the DGL model too)
  split<k>_test_preds.tsv sample_id  label  logit        (test set, best checkpoint)
  split<k>_gates.tsv     per-gene mean attention gate over the test set (interpretability)
  split<k>_embed_leakage.tsv   (only with --leak-pairs)
  graph.json             what graph was actually used (mode, edge counts, LD settings)
  log.txt
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch
from torch import nn
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prsnet_torch import PRSNet, load_ggi_graph
import prsnet_ld


# ----------------------------------------------------------------- authors' utils
def set_random_seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_splits(labels, seeds=(42, 43, 44, 45, 46, 47), train_ratio=0.8, val_ratio=0.1):
    """Verbatim from src/utils.py (numpy semantics preserved, including the
    in-place shuffle of pos_indexs that carries over between seeds)."""
    splits = []
    pos_indexs = np.where(labels == 1)[0]
    neg_indexs = np.where(labels == 0)[0]
    for seed in seeds:
        np.random.seed(seed)
        np.random.shuffle(pos_indexs)
        shuffled_indices = np.random.permutation(len(pos_indexs))
        total_length = len(pos_indexs)
        train_size = int(total_length * train_ratio)
        val_size = int(total_length * val_ratio)
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:train_size + val_size]
        test_indices = shuffled_indices[train_size + val_size:]
        train_pos_data = pos_indexs[train_indices]
        val_pos_data = pos_indexs[val_indices]
        test_pos_data = pos_indexs[test_indices]
        train_neg_indices = np.random.choice(neg_indexs, size=int(len(neg_indexs) * 0.8), replace=False)
        val_neg_indices = np.random.choice(np.setdiff1d(neg_indexs, train_neg_indices), size=int(len(neg_indexs) * 0.1), replace=False)
        test_neg_indices = np.setdiff1d(np.setdiff1d(neg_indexs, train_neg_indices), val_neg_indices)
        train_indices = np.concatenate((train_pos_data, train_neg_indices))
        val_indices = np.concatenate((val_pos_data, val_neg_indices))
        test_indices = np.concatenate((test_pos_data, test_neg_indices))
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        splits.append((train_indices, val_indices, test_indices))
    return splits


def auroc(logits, labels):
    """Mann-Whitney AUROC with mid-ranks for ties (== torchmetrics AUROC(task='binary'))."""
    logits = np.asarray(logits, dtype=np.float64); labels = np.asarray(labels)
    n1 = int((labels == 1).sum()); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(logits)
    return float((r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------------------- GPU keep-alive
class GpuKeepAlive:
    """Bouchet cancels GPU jobs whose GPU shows ~0% utilisation for the first
    hour, and loading the 330 GB feature store from NFS takes 10-20 min of CPU
    time before the first kernel runs. This runs a small matmul loop on the
    device in a background thread (a few % utilisation) until stop() is called.
    No effect on results: nothing it computes is used."""
    def __init__(self, device, enabled=True):
        import threading
        self.enabled = enabled and device.type == "cuda"
        self._stop = threading.Event()
        if self.enabled:
            self._t = threading.Thread(target=self._run, args=(device,), daemon=True); self._t.start()

    def _run(self, device):
        a = torch.randn(2048, 2048, device=device)
        while not self._stop.is_set():
            a = (a @ a).tanh()
            torch.cuda.synchronize(device)
            self._stop.wait(0.5)

    def stop(self):
        if self.enabled:
            self._stop.set(); self._t.join()


# ----------------------------------------------------------------- feature store
class FeatureStore:
    """feats.f32 from 12d: rows = info.csv order, columns = gene*11 + threshold."""
    def __init__(self, data_path, dataset, load="memmap", log=print):
        d = os.path.join(data_path, dataset)
        self.meta = json.load(open(os.path.join(d, "feats.meta.json")))
        self.n, self.n_genes, self.T = self.meta["n_samples"], self.meta["n_genes"], self.meta["n_thresholds"]
        path = self.meta["feats"] or os.path.join(d, "feats.f32")
        shape = (self.n, self.n_genes * self.T)
        nbytes = shape[0] * shape[1] * 4
        assert os.path.getsize(path) == nbytes, f"{path}: {os.path.getsize(path)} bytes, meta says {nbytes}"
        mm = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
        if load == "memmap":
            self.F = mm
        else:
            dtype = np.float16 if load == "ram-f16" else np.float32
            log(f"loading {nbytes / 2**30:.0f} GiB into RAM as {np.dtype(dtype).name} ...")
            t0 = time.time()
            self.F = np.empty(shape, dtype=dtype)
            step = max(1, 2**28 // (shape[1] * 4))         # ~256 MB per copy
            for i in range(0, self.n, step):
                self.F[i:i + step] = mm[i:i + step]
            log(f"  loaded in {(time.time() - t0) / 60:.1f} min")
            del mm
        info = np.genfromtxt(os.path.join(d, "info.csv"), delimiter=",", names=True, dtype=None, encoding=None)
        assert len(info) == self.n, f"info.csv has {len(info)} rows, feats.meta.json says {self.n}"
        self.sample_ids = info["sample_id"].astype(str)
        self.labels = info["label"].astype(np.int64)

    def rows(self, idx):
        """Gather rows (sorted for sequential-ish I/O), return in the requested order as float32 torch."""
        idx = np.asarray(idx)
        order = np.argsort(idx, kind="stable")
        got = np.asarray(self.F[idx[order]], dtype=np.float32)
        out = np.empty_like(got); out[order] = got
        return torch.from_numpy(out)


class BalancedSampler:
    """src/dataset.py balanced_sampling: pick a class uniformly, then a member uniformly (with replacement)."""
    def __init__(self, labels, rng):
        self.classes = np.unique(labels)
        self.members = {c: np.where(labels == c)[0] for c in self.classes}
        self.rng = rng

    def draw(self, k):
        cls = self.rng.choice(self.classes, size=k)
        out = np.empty(k, dtype=np.int64)
        for c in self.classes:
            m = cls == c
            out[m] = self.rng.choice(self.members[c], size=int(m.sum()), replace=True)
        return out


# ----------------------------------------------------------------- graph variants
def build_graph(a, gene_names, device, log):
    """Returns (A for the model, dict for graph.json). A is a tuple (A_ppi, A_ld) in the LD modes."""
    info = {"mode": a.graph_mode, "graph_file": a.graph}
    src, dst, n = prsnet_ld.load_edges(a.graph)
    und = prsnet_ld.undirected_simple(src, dst)
    info["ppi_undirected_edges"] = int(len(und))
    if a.graph_mode == "ppi":
        A, n2 = load_ggi_graph(a.graph, expect_nodes=None, expect_edges=None, device=device)
        info["stored_edges"] = int(len(src))
    elif a.graph_mode == "identity":
        A = prsnet_ld.identity_graph(n, device); info["stored_edges"] = n
    elif a.graph_mode == "shuffled":
        A, done, E = prsnet_ld.shuffled_graph(src, dst, n, seed=a.shuffle_seed, device=device)
        info.update(shuffle_seed=a.shuffle_seed, swaps_applied=int(done), stored_edges=int(4 * E + n))
        log(f"graph: degree-preserving shuffle, {done:,} swaps applied over {E:,} undirected PPI edges (seed {a.shuffle_seed})")
    elif a.graph_mode in ("ppi+ld", "ld"):
        if not a.ld_pairs or gene_names is None:
            raise SystemExit("--graph-mode ppi+ld / ld need --ld-pairs and --gene-order")
        A_ld, n_ld, mean_abs = prsnet_ld.ld_graph(a.ld_pairs, gene_names, weight=a.ld_weight, min_abs_r=a.ld_min_abs_r,
                                                  r_col=a.ld_col, device=device)
        if a.graph_mode == "ppi+ld":
            A_base, _ = load_ggi_graph(a.graph, expect_nodes=None, expect_edges=None, device=device)
        else:
            A_base = prsnet_ld.identity_graph(n, device)
        A = (A_base, A_ld)
        info.update(ld_pairs=a.ld_pairs, ld_col=a.ld_col, ld_weight_mode=a.ld_weight, ld_min_abs_r=a.ld_min_abs_r,
                    ld_undirected_edges=int(n_ld), ld_mean_abs_r=mean_abs, ld_init=a.ld_init,
                    base="ppi" if a.graph_mode == "ppi+ld" else "identity")
        log(f"graph: {a.graph_mode} -- {n_ld:,} undirected LD edges (|r| >= {a.ld_min_abs_r}, mean |r| {mean_abs:.3f}, "
            f"weight {a.ld_weight}) on top of {'PPI' if a.graph_mode == 'ppi+ld' else 'identity'}")
    else:
        raise SystemExit(f"unknown --graph-mode {a.graph_mode}")
    return A, n, info


# ----------------------------------------------------------------- training
def evaluate(model, A, store, idx, device, batch_size, want_preds=False, want_gates=False):
    model.eval()
    logits, gate_sum = [], None
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            x = store.rows(idx[i:i + batch_size]).to(device, non_blocking=True)
            p, w = model(A, x)
            logits.append(p.reshape(-1).float().cpu().numpy())
            if want_gates:
                g = w.reshape(x.shape[0], -1).sum(0).cpu().numpy()
                gate_sum = g if gate_sum is None else gate_sum + g
    logits = np.concatenate(logits)
    score = auroc(logits, store.labels[idx])
    out = {"auroc": score}
    if want_preds:
        out["logits"] = logits
    if want_gates:
        out["gates"] = gate_sum / len(idx)
    return out


def train_split(k, seed, split, store, A, device, a, log):
    tr, va, te = split
    set_random_seed(a.seed)                       # the authors seed once per run; per-split re-seeding
    torch.manual_seed(a.seed + k)                 # keeps split k reproducible when run alone (--splits k)
    rng = random.Random(a.seed + k)
    use_ld = a.graph_mode in ("ppi+ld", "ld")
    model = PRSNet(n_genes=store.n_genes, d_input=store.T, use_ld=use_ld, ld_init=a.ld_init).to(device)
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0)
    sampler = BalancedSampler(store.labels[tr], np.random.RandomState(a.seed + k))
    n_steps = min(a.n_steps + 1, a.epoch_draws // a.batch_size)    # authors: range(len(train_loader)), break at n_steps
    log(f"split {k} seed {seed}: train {len(tr):,} (cases {int(store.labels[tr].sum()):,})  val {len(va):,}  "
        f"test {len(te):,} (cases {int(store.labels[te].sum()):,})  steps {n_steps:,}")
    best_val, best_test, best_step, stale = 0.0, 0.0, -1, 0
    best_state = None
    running = []
    t0 = time.time()
    if a.eval_only:                                # load the saved checkpoint, skip training
        ck = os.path.join(a.out_dir, f"split{k}_best.pt")
        best_state = torch.load(ck, map_location="cpu")
        model.load_state_dict(best_state)
        best_val = evaluate(model, A, store, va, device, a.eval_batch)["auroc"]
        best_test = evaluate(model, A, store, te, device, a.eval_batch)["auroc"]
        log(f"eval-only: loaded {ck}  val {best_val:.4f}  test {best_test:.4f}")
        n_steps, step = 0, -1
    for step in range(n_steps):
        model.train()
        opt.zero_grad()
        b = tr[sampler.draw(a.batch_size)]
        x = store.rows(b).to(device, non_blocking=True)
        y = torch.from_numpy(store.labels[b].astype(np.float32)).reshape(-1, 1).to(device)
        preds, _ = model(A, x)
        loss = loss_fn(preds, y)
        loss.backward()
        opt.step()
        running.append(loss.item())
        if (step + 1) % a.log_interval == 0:
            extra = f"  ld_w {model.ld_weights()[0]:+.4f}" if use_ld else ""
            log(f"[{step + 1}] loss {np.mean(running):.3f}{extra}  ({(time.time() - t0) / 60:.1f} min)")
            running = []
        if (step + 1) % a.eval_interval == 0:
            val = evaluate(model, A, store, va, device, a.eval_batch)["auroc"]
            if val > best_val:
                best_val, best_step, stale = val, step + 1, 0
                best_test = evaluate(model, A, store, te, device, a.eval_batch)["auroc"]
                best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
            else:
                stale += 1
                if stale == a.early_stop:
                    log(f"early stop at step {step + 1}")
                    break
            log(f"[{step + 1}] val {val:.4f}  best_val {best_val:.4f} @ {best_step}  test@best {best_test:.4f}")
    minutes = (time.time() - t0) / 60
    # best checkpoint: save, and dump test predictions + gene gates for the record
    model.load_state_dict(best_state)
    if not a.eval_only:
        torch.save(best_state, os.path.join(a.out_dir, f"split{k}_best.pt"))
    ev = evaluate(model, A, store, te, device, a.eval_batch, want_preds=True, want_gates=True)
    if abs(ev["auroc"] - best_test) > 1e-4:          # float32 reload differs in the last bits; a real mismatch is far larger
        log(f"WARNING: reloaded checkpoint test AUROC {ev['auroc']:.6f} vs recorded {best_test:.6f}")
    if not a.eval_only:
        with open(os.path.join(a.out_dir, f"split{k}_test_preds.tsv"), "w") as w:
            w.write("sample_id\tlabel\tlogit\n")
            for s, y, p in zip(store.sample_ids[te], store.labels[te], ev["logits"]):
                w.write(f"{s}\t{y}\t{p:.6f}\n")
        with open(os.path.join(a.out_dir, f"split{k}_gates.tsv"), "w") as w:
            w.write("node\tgene\tmean_gate_test\n")
            for i, g in enumerate(ev["gates"]):
                w.write(f"{i}\t{a.gene_names[i] if a.gene_names else ''}\t{g:.6f}\n")
    ld_w = model.ld_weights()[0] if use_ld else float("nan")
    # embedding-level LD leakage at the best checkpoint
    if a.leak_pairs_loaded is not None:
        t1 = time.time()
        res = prsnet_ld.embedding_leakage(model, A, store, a.leak_rows_loaded, a.leak_pairs_loaded, device,
                                          batch=a.eval_batch)
        prsnet_ld.write_leakage(os.path.join(a.out_dir, f"split{k}_embed_leakage.tsv"), res)
        log(f"embedding leakage, split {k} ({(time.time() - t1) / 60:.1f} min):\n" + prsnet_ld.format_leakage(res))
    return best_val, best_test, best_step, min(step + 1, n_steps), minutes, ld_w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", required=True, help="12d genome dir (holds <dataset>/feats.f32)")
    ap.add_argument("--dataset", required=True, help="e.g. ad_eur")
    ap.add_argument("--graph", required=True, help="ggi_graph.bin or .npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gene-order", default=None, help="gene_order.txt; labels the gates file, required for --ld-pairs/--leak-pairs")
    ap.add_argument("--load", choices=["memmap", "ram", "ram-f16"], default="memmap",
                    help="memmap: page cache does the work (needs ~feature-store size of free RAM to be fast); "
                         "ram: np.fromfile the whole store (332 GB for 19,836 genes x 381k people); "
                         "ram-f16: half that, values rounded to float16 (~3 significant digits)")
    ap.add_argument("--splits", default="0,1,2,3,4,5", help="which of the six splits to run, e.g. 0 for a smoke test")
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--n-steps", type=int, default=20000)
    ap.add_argument("--epoch-draws", type=int, default=10_000_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-batch", type=int, default=1024,
                    help="rows per inference batch; identical numbers at any size, larger keeps the GPU busier (was 128)")
    ap.add_argument("--no-keepalive", action="store_true", help="disable the background GPU keep-alive loop")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--early-stop", type=int, default=50)
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--cpu", action="store_true")
    # --- LD experiments
    ap.add_argument("--graph-mode", choices=["ppi", "identity", "shuffled", "ppi+ld", "ld"], default="ppi")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--ld-pairs", default=None, help="pairs.tsv.gz from 14_ld_leakage.py (LD edges)")
    ap.add_argument("--ld-col", default="r_P<=1", help="which r column of pairs.tsv.gz weights the LD edges")
    ap.add_argument("--ld-weight", choices=["signed", "abs", "binary"], default="signed")
    ap.add_argument("--ld-min-abs-r", type=float, default=0.0)
    ap.add_argument("--ld-init", type=float, default=0.0, help="initial value of the learnable LD message weight")
    ap.add_argument("--leak-pairs", default=None, help="pairs.tsv.gz: compute embedding-level leakage after each split")
    ap.add_argument("--leak-rows", default=None, help="sampled_rows.txt from 14_ld_leakage.py (row indices of the people)")
    ap.add_argument("--leak-max-random", type=int, default=50000)
    ap.add_argument("--eval-only", action="store_true",
                    help="no training: load split<k>_best.pt from --out-dir, report val/test AUROC and the leakage metric "
                         "(results.tsv, preds and gates are left untouched)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    logf = open(os.path.join(a.out_dir, "log.txt"), "a")
    def log(msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    log(f"device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    log(f"torch {torch.__version__}  args {vars(a)}")

    set_random_seed(a.seed)
    keep = GpuKeepAlive(device, enabled=not a.no_keepalive)     # GPU busy while the store loads (see class)
    store = FeatureStore(a.data_path, a.dataset, a.load, log)
    log(f"features: {store.n:,} samples x {store.n_genes:,} genes x {store.T} thresholds  (cases {int(store.labels.sum()):,})")
    a.gene_names = [l.strip() for l in open(a.gene_order) if l.strip()] if a.gene_order else None
    if a.gene_names is not None:
        assert len(a.gene_names) == store.n_genes, f"gene order has {len(a.gene_names)} genes, store has {store.n_genes}"
    A, n_nodes, ginfo = build_graph(a, a.gene_names, device, log)
    assert n_nodes == store.n_genes, f"graph has {n_nodes} nodes, feature store has {store.n_genes} genes"
    log(f"graph: {ginfo}")
    json.dump(ginfo, open(os.path.join(a.out_dir, "graph.json"), "w"), indent=2)

    a.leak_pairs_loaded = a.leak_rows_loaded = None
    if a.leak_pairs:
        if not a.leak_rows or a.gene_names is None:
            raise SystemExit("--leak-pairs needs --leak-rows and --gene-order")
        a.leak_pairs_loaded = prsnet_ld.load_leak_pairs(a.leak_pairs, a.gene_names, max_random=a.leak_max_random)
        a.leak_rows_loaded = np.loadtxt(a.leak_rows, dtype=np.int64)
        log("leakage pairs: " + ", ".join(f"{c} {len(v[0]):,}" for c, v in a.leak_pairs_loaded.items())
            + f"; {len(a.leak_rows_loaded):,} people")

    splits = generate_splits(store.labels)        # all six, so split k is the same whichever subset runs
    seeds = [42, 43, 44, 45, 46, 47]
    res_path = os.path.join(a.out_dir, "results.tsv")
    if not os.path.exists(res_path) and not a.eval_only:
        open(res_path, "w").write("split\tseed\tbest_val_auroc\ttest_auroc\tbest_step\tsteps_run\tminutes\tld_weight\n")
    for k in [int(s) for s in a.splits.split(",")]:
        bv, bt, bs, nr, mins, ld_w = train_split(k, seeds[k], splits[k], store, A, device, a, log)
        if a.eval_only:
            continue
        open(res_path, "a").write(f"{k}\t{seeds[k]}\t{bv:.6f}\t{bt:.6f}\t{bs}\t{nr}\t{mins:.1f}\t{ld_w:.4f}\n")
        log(f"---------------- split {k} final: best_val {bv:.4f}  test {bt:.4f}"
            + (f"  ld_weight {ld_w:+.4f}" if not np.isnan(ld_w) else "") + f"  ({mins:.0f} min) ----------------")
    rows = [l.split("\t") for l in open(res_path).read().strip().split("\n")[1:]] if os.path.exists(res_path) else []
    tests = np.array([float(r[3]) for r in rows])
    if len(tests):
        log(f"test AUROC over {len(tests)} split(s): mean {tests.mean():.4f}  sd {tests.std(ddof=1) if len(tests) > 1 else 0:.4f}")
    keep.stop()


if __name__ == "__main__":
    main()
