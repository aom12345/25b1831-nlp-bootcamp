#!/usr/bin/env bash
# Runs every GPU step after Task 2, in order, logging to out/. Run from the kit root with the venv:
#   bash code/run_gpu_chain.sh
set -e
export HF_HOME=${HF_HOME:-D:/inter_iit/hf_cache} HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=${PY:-python}
step() { echo "=== $(date +%H:%M:%S) $*"; }

step "Task 3: val_idea"
$PY code/run_system.py --mode idea --items data/val_items.csv --out predictions/val_idea > out/log_val_idea.txt 2>&1
$PY metrics.py predictions/val_idea --json out/val_idea.json --csv out/val_idea.csv > out/score_val_idea.txt 2>&1

step "Task 4: baseline timing (same session as the fast runs)"
$PY baseline/run_baseline.py --items data/val_items.csv --out out/baseline_val > out/log_baseline_timing.txt 2>&1

for N in 16 8; do
  step "Task 4: fast candidate nfe_step=$N"
  $PY code/run_system.py --mode idea --items data/val_items.csv --out out/val_fast_nfe$N --set nfe_step=$N > out/log_val_fast_nfe$N.txt 2>&1
  $PY metrics.py out/val_fast_nfe$N --json out/val_fast_nfe$N.json --csv out/val_fast_nfe$N.csv > out/score_val_fast_nfe$N.txt 2>&1
done

step "Task 3: t3_idea (40 eval)"
$PY code/run_system.py --mode idea --items data/eval_items.csv --out predictions/t3_idea > out/log_t3_idea.txt 2>&1
step "chain done"
