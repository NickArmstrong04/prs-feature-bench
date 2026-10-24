# source env.sh  -- in every new shell / salloc before using the harness
# BENCH is the directory this file lives in, so a git worktree of the repo
# (e.g. ../prs-feature-bench-ld on the ld-ablation branch) gets its own BENCH,
# runs/ and results.tsv when its own env.sh is sourced.
export BENCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BENCH_PY=/home/naa79/project_pi_sz622/naa79/prs-benchmark/tools/prsnet_venv/bin/python
module load PLINK/1.9b_7.11-x86_64 2>/dev/null     # plink2 variants load their own module (same Lmod family)
cd "$BENCH"
