# Experiments (not needed to reproduce the submission)

Side experiments on validation items only. Results and conclusions: `EXPERIMENTS.md` in the kit root.
Run from the kit root with the venv; audio goes to `out/`.

| Script | What it tests |
|---|---|
| `exp.py` | Round 1: seeds, commas inside numbers, guidance 3.0 (6 items) |
| `exp2.py` | Round 2: V4 transcript fix, guidance 3.0 over 3 seeds; scores with Whisper + IndicWav2Vec |
| `exp3.py` | Round 3: PCI and EPSS sampler variants, 8 steps |
| `rerank.py` | Picking the best of 3 seeds with a second recognizer |
| `run_fast.py` | Task 4 candidates: fewer steps, fp16/bf16 autocast |
| `run_rest.sh`, `t4_chain.sh`, `t4_bf16_chain.sh` | The run orders used (logs in `out/*chain*.txt`) |
| `kaggle/` | Kaggle notebook + metadata (could not run: no GPU session / token secret) |

`exp*.py` default to `D:\inter_iit\TTS`; set `TTS_KIT` to run elsewhere.
