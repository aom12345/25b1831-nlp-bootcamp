#!/usr/bin/env bash
# Everything left, locally, one after another (GPU never shared):
#   wait for t3_idea -> round 3 experiments -> Task 4 timing (baseline, then 16 steps) -> score -> t4_fast
cd /d/inter_iit/TTS
export HF_HOME='D:\inter_iit\hf_cache' HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=/d/inter_iit/tts_venv/Scripts/python.exe
LOG=out/chain_rest.txt
step() { echo "=== $(date +%H:%M:%S) $*" | tee -a $LOG; }

step "waiting for t3_idea"
until grep -q "^Done:" out/log_t3_idea.txt 2>/dev/null; do sleep 30; done
step "t3_idea done: $(ls predictions/t3_idea | wc -l) files"

step "round 3 synth (PCI, EPSS-7, 8 steps)"
$PY -X utf8 code/experiments/exp3.py synth > out/exp3_synth.log 2>&1 || step "round 3 synth FAILED (continuing)"
step "round 3 score"
$PY -X utf8 code/experiments/exp3.py score > out/exp3_score.txt 2> out/exp3_score.err || step "round 3 score FAILED (continuing)"

step "Task 4: baseline timing (24 val)"
$PY baseline/run_baseline.py --items data/val_items.csv --out out/baseline_val > out/log_baseline_timing.txt 2>&1
step "Task 4: fast timing nfe_step=16 (24 val), same session"
$PY code/run_system.py --mode idea --items data/val_items.csv --out out/val_fast_nfe16 --set nfe_step=16 > out/log_val_fast_nfe16.txt 2>&1
step "Task 4: score fast val"
$PY metrics.py out/val_fast_nfe16 --json out/val_fast_nfe16.json --csv out/val_fast_nfe16.csv > out/score_val_fast_nfe16.txt 2>&1
step "Task 4: t4_fast (40 eval)"
$PY code/run_system.py --mode idea --items data/eval_items.csv --out predictions/t4_fast --set nfe_step=16 > out/log_t4_fast.txt 2>&1
step "chain done: t4_fast $(ls predictions/t4_fast | wc -l) files"
