cd /d/inter_iit/TTS
export HF_HOME='D:\inter_iit\hf_cache' HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=/d/inter_iit/tts_venv/Scripts/python.exe
L=out/t4_bf16_chain.txt
step() { echo "=== $(date +%H:%M:%S) $*" | tee -a $L; }
step "Task 4: baseline timing (24 val)"
$PY baseline/run_baseline.py --items data/val_items.csv --out out/baseline_val > out/log_baseline_timing.txt 2>&1
step "Task 4: bf16 32 steps timing (24 val), same session"
$PY code/run_system.py --mode idea --items data/val_items.csv --out out/val_fast_bf16 --bf16 > out/log_val_fast_bf16.txt 2>&1
step "Task 4: score bf16 val"
$PY metrics.py out/val_fast_bf16 --json out/val_fast_bf16.json --csv out/val_fast_bf16.csv > out/score_val_fast_bf16.txt 2>&1
$PY -c "import json; b=json.load(open('out/baseline_val_timing.json'))['total_synth_s']; f=json.load(open('out/val_fast_bf16_timing.json'))['total_synth_s']; c=json.load(open('out/val_fast_bf16.json'))['cer']; print(f'baseline {b}s  bf16 {f}s  speedup {b/f:.3f}  CER {c}  limit 0.0736  {\"PASS\" if c<=0.0736 else \"FAIL\"}')" | tee -a $L
step "Task 4: t4_fast (40 eval, bf16)"
rm -rf predictions/t4_fast
$PY code/run_system.py --mode idea --items data/eval_items.csv --out predictions/t4_fast --bf16 > out/log_t4_fast.txt 2>&1
step "t4_fast: $(ls predictions/t4_fast | wc -l) files"
step "done"
