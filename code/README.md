# Code — Hindi–Hinglish voice cloning bootcamp

All commands run from the kit root (the folder with `metrics.py`), with the kit's `baseline/` and
`data/` folders in place and the scripts of this folder copied to `code/`.

## Environment

Tested on Windows 11, NVIDIA RTX 2050 (4 GB), Python 3.13.5, seed 0 everywhere.

```bash
python -m venv tts_venv
tts_venv/Scripts/python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# The kit pins IndicF5 to fb652a96…, which is the Hugging Face *model* revision and does not
# exist on GitHub; the GitHub code is installed from main without its deps (its setup.py wants
# numpy<=1.26, which has no Python 3.13 wheel). The model weights stay pinned by config.yaml.
tts_venv/Scripts/python -m pip install --no-deps "git+https://github.com/ai4bharat/IndicF5.git@13f7c4d627cc10111aea8fe9c0039462cacacdc7"
tts_venv/Scripts/python -m pip install "numpy>=2.1,<2.3" "scipy>=1.15,<2" transformers==4.49.0 \
    accelerate==1.15.0 soundfile==0.14.0 librosa==0.11.0 pyyaml==6.0.3 pandas==2.2.3 "speechbrain>=1.0" \
    "huggingface-hub>=0.26,<1.0" "datasets<4" wandb cached_path click ema_pytorch hydra-core safetensors \
    tomli torchdiffeq tqdm vocos "x_transformers>=1.31.14" pydub jieba pypinyin matplotlib \
    transformers_stream_generator
huggingface-cli login        # after accepting the IndicF5 terms
```

Windows only: SpeechBrain symlinks its files, which needs admin rights, so copy the
`speechbrain/spkrec-ecapa-voxceleb` snapshot into `~/.cache/bootcamp_ecapa/` and add a copy of
`label_encoder.txt` named `label_encoder.ckpt`. torch 2.8 is used because torchaudio ≥ 2.9 loads
audio through torchcodec, which needs FFmpeg.

## Files

| File | What it does |
|---|---|
| `normalize.py` | Task 2 rule-based frontend `normalize(text)`; with no arguments writes `predictions/t2_normalized.csv` |
| `test_normalize.py` | Checks `normalize()` against every validation `spoken` reference (metrics.py canonicalization) and the t2 sanity rules |
| `translit.py` | Task 3 rule-based English → Devanagari spelling (no lookup table); with no arguments scores its rules on the validation-only dev words |
| `run_system.py` | Synthesizes an item file with `--mode baseline|norm|idea`, times it like `baseline/run_baseline.py`; `--bf16` for Task 4 |
| `pilot.py`, `duration_analysis.py` | Quick pilots on a few validation items; duration (characters vs syllables) analysis |
| `experiments/` | Side experiments on validation items only (not needed to reproduce the submission); see its README |
| `baseline_val_timing.json`, `val_fast_bf16_timing.json` | Task 4 timing files (same session) |

## Reproduce every submitted file

```bash
# Task 1
python metrics.py data/baseline_val --json out/baseline_val_scores.json --csv out/baseline_val_scores.csv
# predictions/t1_val_errors.csv: tags written by hand after listening (see report)

# Task 2
python code/test_normalize.py
python code/normalize.py                                    # -> predictions/t2_normalized.csv
python code/run_system.py --mode norm --items data/eval_items.csv --out predictions/t2_norm

# Task 3 (val_control = Task 2 text, baseline otherwise)
python code/run_system.py --mode norm --items data/val_items.csv  --out predictions/val_control
python code/run_system.py --mode idea --items data/val_items.csv  --out predictions/val_idea
python code/run_system.py --mode idea --items data/eval_items.csv --out predictions/t3_idea
python metrics.py predictions/val_control --json out/val_control.json --csv out/val_control.csv
python metrics.py predictions/val_idea    --json out/val_idea.json    --csv out/val_idea.csv

# Task 4: Task 3 system with the sampler under bfloat16 autocast, 32 steps.
# Same machine, same session, back to back, nothing else on the GPU.
python baseline/run_baseline.py --items data/val_items.csv --out out/baseline_val         # -> out/baseline_val_timing.json
python code/run_system.py --mode idea --items data/val_items.csv --out out/val_fast_bf16 --bf16   # -> out/val_fast_bf16_timing.json
python metrics.py out/val_fast_bf16 --json out/val_fast_bf16.json --csv out/val_fast_bf16.csv
python code/run_system.py --mode idea --items data/eval_items.csv --out predictions/t4_fast --bf16
```

Task 4 result (RTX 2050): baseline 1437.7 s, fast 515.3 s → speed-up 2.79×; validation CER 0.0625
(Task 3: 0.0613, limit 1.2×). The timing files are in this folder: `baseline_val_timing.json`,
`val_fast_bf16_timing.json`.

`--bf16` runs the flow-matching sampler under `torch.autocast(dtype=bfloat16)`. It does not change the
model weights, the text, the reference clip or any decoding setting; it only computes in 16-bit
bfloat16 instead of 32-bit floats. fp16 was tried and produced invalid (NaN) audio; fewer steps
(12, 10, 8) failed the CER limit. Details in `experiments/` and `EXPERIMENTS.md` (in this folder).

## AI coding assistant

Claude Code (Anthropic) was used to write and debug the scripts in this folder, set up the
environment, run the experiments, and help draft `idea.md` and `report.md`, which I reviewed and
edited.
