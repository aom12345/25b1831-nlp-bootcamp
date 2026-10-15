cd /d/inter_iit/TTS
export HF_HOME='D:\inter_iit\hf_cache' HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=/d/inter_iit/tts_venv/Scripts/python.exe
L=out/t4_chain.txt
step() { echo "=== $(date +%H:%M:%S) $*" | tee -a $L; }
IDS=val_hd_01,val_hd_03,val_hd_06,val_hd_07,val_hg_04,val_hi_02
step "fp16 nfe16, 6 items"
$PY -X utf8 code/experiments/run_fast.py --out out/t4c_fp16_nfe16_6 --nfe 16 --half fp16 --ids $IDS > out/t4c_fp16_nfe16_6.log 2>&1 || step "fp16 FAILED"
step "score fp16 6 items vs fp32 nfe16"
$PY -X utf8 - > out/t4c_fp16_score.txt 2>/dev/null <<'PYEOF'
import sys, json, csv; sys.path.insert(0, '.')
import metrics, numpy as np
ids = "val_hd_01,val_hd_03,val_hd_06,val_hd_07,val_hg_04,val_hi_02".split(",")
items = {r['id']: r for r in metrics.read_csv('data/val_items.csv')}
prompts = metrics.load_prompts('data/prompts.csv')
m = metrics.Models()
rows = {r['id']: r for r in metrics.evaluate_folder('out/t4c_fp16_nfe16_6', [items[i] for i in ids], prompts, m)}
ref = {r['id']: r for r in csv.DictReader(open('out/val_fast_nfe16.csv', encoding='utf-8'))}
t16 = json.load(open('out/t4c_fp16_nfe16_6_timing.json'))['per_item_s']
t32 = json.load(open('out/val_fast_nfe16_timing.json'))['per_item_s']
for i in ids:
    print(f"{i}  fp32 {float(ref[i]['cer']):.3f}/{float(ref[i]['sim']):.2f} {t32[i]:.1f}s   fp16 {rows[i]['cer']:.3f}/{rows[i]['sim']:.2f} {t16[i]:.1f}s  {rows[i]['reasons'] or rows[i]['hyp']}")
print("MEAN fp32 %.3f %.1fs   fp16 %.3f %.1fs" % (np.mean([float(ref[i]['cer']) for i in ids]), sum(t32[i] for i in ids),
      np.mean([rows[i]['cer'] for i in ids]), sum(t16[i] for i in ids)))
PYEOF
for N in 12 10; do
  step "nfe$N full 24"
  $PY -X utf8 code/experiments/run_fast.py --out out/t4c_nfe$N --nfe $N > out/t4c_nfe$N.log 2>&1
  $PY metrics.py out/t4c_nfe$N --json out/t4c_nfe$N.json --csv out/t4c_nfe$N.csv > out/score_t4c_nfe$N.txt 2>&1
  step "nfe$N CER $(grep -m1 '"cer"' out/t4c_nfe$N.json)"
done
step "done"
# Run afterwards, by hand (fp16 gave NaN audio, so bf16 instead):
#   $PY -X utf8 code/experiments/run_fast.py --out out/t4c_bf16_nfe16_6 --nfe 16 --half bf16 --ids $IDS
#   for N in 16 32; do $PY -X utf8 code/experiments/run_fast.py --out out/t4c_bf16_nfe$N --nfe $N --half bf16
#     $PY metrics.py out/t4c_bf16_nfe$N --json out/t4c_bf16_nfe$N.json --csv out/t4c_bf16_nfe$N.csv; done
