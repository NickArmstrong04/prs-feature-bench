# source env.sh  -- in every new shell / salloc before using the harness
export BENCH=/home/naa79/project_pi_sz622/naa79/prs-feature-bench
export BENCH_PY=/home/naa79/project_pi_sz622/naa79/prs-benchmark/tools/prsnet_venv/bin/python
module load PLINK/1.9b_7.11-x86_64 2>/dev/null     # plink2 variants load their own module (same Lmod family)
cd "$BENCH"
