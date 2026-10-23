# prs-feature-bench — rules for this session

This repo is a test harness for versions of the PRS-Net gene-feature scripts.
Read README.md first (sections 2 and 6 are the contract and the method).

## Never
- Edit `variants/ref_ct/`, `variants/ref_eqtl_plink2/`, `lib/ref_12_prsnet.py`,
  anything under `fixture/`, `golden/` or `prod_snapshot/` (read-only copies
  of the production PRS-Net scripts; `prod_snapshot/MANIFEST.txt` has the
  provenance and md5s — production is the source of truth, re-snapshot from
  there). If a reference looks wrong, say so
  and stop; the fix belongs in the production scripts on the cluster.
- Loosen `--rtol`/`--atol` to make a variant pass. A numeric deviation gets a
  new, named golden and a note here, not a wider tolerance.
- Run anything on the login node that takes more than a few seconds of CPU.
  Use the `salloc` the user has open; if none is open, say which request to
  make (`salloc -c 4 --mem=16G -t 2:00:00` is the pattern that starts fast
  on this account; 8 CPUs is refused by the interactive QOS).
- Delete `runs/` entries or `results.tsv` rows. They are the experiment log.

## Always
- Make a change as a new `variants/<name>/` (copy the closest variant), one
  idea per variant, and run it through `bash run_variant.sh <name> --tag "…"`
  — never by calling a variant's scripts directly, because only the harness
  records timing, commit and verdict.
- Commit after each PASSing variant with a message that says what changed and
  the wall time vs the previous best at the same `--threads`.
- Before claiming a speed-up, show two rows of `results.tsv` at equal threads.
- When a run FAILs, read `runs/<v>_<stamp>/compare.txt` and `run.log` before
  theorising; the worst columns are listed first.
- Keep the Python plain numpy/scipy/bed-reader unless a variant's whole point
  is a new dependency; then install it into the bench venv and say so.

## Environment on Bouchet
- `BENCH_PY` = the production PRS-Net venv python (`$PRSNET_VENV/bin/python` in
  `$ROOT/scripts/slurm/config.sh`); it has numpy, scipy, bed-reader.
- plink 1.9 / plink2 are Lmod modules in the same family — loading one unloads
  the other (`PLINK/1.9b_7.11-x86_64`, `PLINK/2_avx2_20250707`). The reference
  scripts load what they need; a variant needing both must resolve absolute
  paths right after each load.
- `set -o pipefail` + `sort | head` kills jobs silently (SIGPIPE); use
  `awk 'NR<=n'`.

## Where things are recorded
- `results.tsv` — every run.
- `README.md` §7 — the optimization list; strike items as they are tried,
  with the results row that settled them.
- Project doc `claude/feature_bench.md` (claude.ai project "Benchmarking")
  mirrors README.md; update it when the README changes materially.
