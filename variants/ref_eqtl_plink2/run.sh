#!/bin/bash
# =============================================================================
#  variants/ref_eqtl_plink2/run.sh -- the ORACLE for eQTL-weighted features
#
#  Feature (i, gene g, channel c) = sum_j  w_{j,g,c} * dosage_i(a1_j)
#  i.e. predicted expression of g in cell type c for person i, with missing
#  calls mean-imputed (plink2's default; the same rule 12c uses).
#
#  This variant asks plink2 to compute every column at once from one wide
#  score file (multi-column --score, scoresums). It is slow-ish and exact;
#  it is what golden/eqtl_<set> is made from. Fast variants are compared
#  against it. Do not optimise this file.
#
#  Needs fixture/weights/$WEIGHTS.tsv (contract in eqtl/weights_to_plink2.py).
#  Called by run_variant.sh as:  WEIGHTS=<set> run.sh <fixture_dir> <out_dir>
# =============================================================================
set -euo pipefail
FIX=$1; OUT=$2
W=$OUT/work; mkdir -p "$W"
PY=${BENCH_PY:-python3}; NT=${BENCH_THREADS:-4}
WEIGHTS=${WEIGHTS:-onek1k}
WFILE=$FIX/weights/$WEIGHTS.tsv
[[ -f "$WFILE" ]] || { echo "no weight set: $WFILE  (see eqtl/README section of README.md)"; exit 1; }
BENCH=$(cd "$(dirname "$0")/../.." && pwd)

if ! command -v plink2 >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load PLINK/2_avx2_20250707 2>/dev/null || true
fi
command -v plink2 >/dev/null || { echo "plink2 not found"; exit 1; }

"$PY" "$BENCH/eqtl/weights_to_plink2.py" "$WFILE" "$FIX/geno.bim" "$W/wide.tsv" --min-snps 1
NCOL=$(awk 'NR>1' "$W/wide.tsv.columns.tsv" | wc -l)
(( NCOL > 0 )) || { echo "no usable columns"; exit 1; }

# plink2 --score: col 1 = SNP, col 2 = A1, score columns 3..(NCOL+2);
# header-read names the scores; scoresums instead of averages.
# plink2's cols= syntax and the exact SUM column names changed between
# releases -- the first run on Bouchet should confirm the header printed
# below (the production `check` had the same kind of surprise, C.6 (3)).
plink2 --bfile "$FIX/geno" \
       --score "$W/wide.tsv" 1 2 header-read cols=+scoresums,-scoreavgs \
       --score-col-nums 3-$((NCOL + 2)) \
       --out "$W/oracle" --threads "$NT" --memory 8000 > "$W/plink2.log" 2>&1 \
    || { tail -n 20 "$W/plink2.log"; exit 1; }
head -c 400 "$W/oracle.sscore" | head -n 1 | cut -c1-200; echo

# .sscore -> harness layout (columns in the wide file's order)
"$PY" - "$W/oracle.sscore" "$W/wide.tsv.columns.tsv" "$OUT" <<'EOF'
import sys, numpy as np
ss, colf, out = sys.argv[1:4]
with open(ss) as fh:
    hdr = fh.readline().lstrip("#").rstrip("\n").split("\t")
    rows = [l.rstrip("\n").split("\t") for l in fh if l.strip()]
# score columns = everything ending in _SUM, in file order
sc = [i for i, h in enumerate(hdr) if h.endswith("_SUM") and h != "NAMED_ALLELE_DOSAGE_SUM"]
names = [hdr[i][:-4] for i in sc]
cols = [l.rstrip("\n").split("\t") for l in open(colf)][1:]
want = [f"{g}|{c}" for _, g, c, _ in cols]
assert names == want, f"plink2 column order differs from the wide file: {names[:3]} vs {want[:3]}"
fid_i, iid_i = hdr.index("FID") if "FID" in hdr else 0, hdr.index("IID")
n, m = len(rows), len(sc)
S = np.memmap(f"{out}/scores.f32", dtype=np.float32, mode="w+", shape=(n, m))
for r, f in enumerate(rows):
    S[r, :] = np.array([f[i] for i in sc], dtype=np.float64)
S.flush()
with open(f"{out}/samples.txt", "w") as w:
    for f in rows:
        w.write(f"{f[fid_i]}\t{f[iid_i]}\n")
with open(f"{out}/columns.tsv", "w") as w:
    w.write("col\tgene\tchannel\tn_snps\n")
    for j, (_, g, c, k) in enumerate(cols):
        w.write(f"{j}\t{g}\t{c}\t{k}\n")
print(f"oracle: {n} samples x {m} columns -> {out}")
EOF
