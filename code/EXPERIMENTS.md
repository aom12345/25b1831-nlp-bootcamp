# Experiments log: side experiments around the Task 3 idea

All runs: validation items only, local RTX 2050, kit `metrics.py` (Whisper large-v3-turbo + ECAPA) unless stated.
Format in tables: `CER/SIM`. Lower CER is better.

**None of these experiments is part of the submitted idea.** The idea is `--mode idea` in
`code/run_system.py`: `translit_text(normalize(text))` with baseline decoding settings
(`"overrides": {}` in `out/val_idea_timing.json`). Only the 6 hinglish items get different text from
the control; hindi, hard and long items get identical text.

---

## 0. Reference point: the idea itself (24 val items, seed 0)

| | control (Task 2) | idea | idea at 16 steps (Task 4 candidate) |
|---|---|---|---|
| all | 0.129 / 0.839 | **0.061 / 0.841** | 0.063 / 0.843 |
| hinglish | 0.309 | **0.039** | 0.029 |
| hindi | 0.037 | 0.037 | 0.044 |
| hard | 0.107 | 0.107 | 0.111 |
| long | 0.039 | 0.039 | 0.048 |

Source: `out/val_idea.json`, `out/val_fast_nfe16.json`.

---

## 1. Syllable-based duration (pilot, 6 hinglish items, seed 0)

**What:** set `speed` so the total duration follows syllable count instead of character count
(`latin_syldur`, `translit_syldur` in `code/pilot.py`).
**Why:** the length formula counts characters, which over- or under-allocates time for some words.

| id | control | translit (= idea) | latin_syldur | translit_syldur |
|---|---|---|---|---|
| val_hg_01 | 0.387 | 0.032 | 0.355 | 0.032 |
| val_hg_02 | 0.324 | 0.027 | 0.324 | 0.027 |
| val_hg_03 | 0.280 | 0.020 | 0.300 | 0.020 |
| val_hg_04 | 0.263 | 0.132 | 0.263 | **0.053** |
| val_hg_05 | 0.283 | 0.000 | 0.283 | 0.000 |
| val_hg_06 | 0.318 | 0.023 | 0.295 | 0.023 |
| **mean** | 0.309 / 0.80 | 0.039 / 0.81 | 0.303 / 0.81 | 0.026 / 0.81 |

**Result:** the whole gain (0.039 → 0.026) comes from one item, `val_hg_04`. The 16-step run, which
doesn't touch duration, also gives `val_hg_04` exactly 0.053, and seeds 1 and 2 of the plain idea give
0.053 and 0.000 (section 3). **Conclusion: noise, not a duration effect.** Duration alone does nothing
for Latin text (`latin_syldur` 0.303).

Source: `out/pilot3.txt`, audio in `out/pilot/`.

---

## 2. Where the time goes (6 items, 26 calls)

**What:** timed each part of a synthesis call.

| part | time per call |
|---|---|
| reference preprocessing | ≈ 0.0–0.1 s |
| flow-matching sampling (`ema.sample`) | ≈ 21–35 s (all of it) |
| vocoder | ≈ 0.0–0.1 s |

**Result:** essentially all the time is in the sampling steps. Caching the reference clip would not
speed anything up. An earlier estimate in chat of "about 370 s of fixed overhead" was wrong.

**Task 4 note:** the 8-step full run (976 s) was slower than the 16-step run (865 s) because the
syllable pilot was using the GPU at the same time. That timing is invalid and needs re-running alone.
The 16-step run passes the CER limit (0.0632 ≤ 1.2 × 0.0613 = 0.0736) with a speed-up of
1360.8 s / 864.8 s = 1.57× (same chain run).

Source: `out/exp_synth.log`, `out/*_timing.json`.

---

## 3. Seed noise, commas in numbers, guidance strength (6 items)

**Items:** `val_hd_01`, `val_hd_03`, `val_hd_06`, `val_hd_07` (hard), `val_hg_04`, `val_hi_02` — each a
known failure.

**Conditions:**
- `idea_s0/s1/s2`: the idea, seeds 0, 1, 2 (seed 0 = `predictions/val_idea`)
- `numcomma`: comma after लाख/करोड़/हज़ार when another number word follows
  (e.g. "एक लाख, सैंतालीस हज़ार, तीन सौ बयासी"), hard items only, seeds 0 and 1
- `cfg3_s0`: guidance strength 3.0 instead of 2.0, seed 0

| id | idea_s0 | idea_s1 | idea_s2 | numcomma_s0 | numcomma_s1 | cfg3_s0 |
|---|---|---|---|---|---|---|
| val_hd_01 | 0.095/0.89 | 0.127/0.87 | 0.143/0.89 | 0.111/0.87 | 0.111/0.90 | 0.095/0.88 |
| val_hd_03 | 0.224/0.85 | 0.163/0.86 | 0.163/0.86 | 0.224/0.85 | 0.184/0.86 | 0.163/0.86 |
| val_hd_06 | 0.130/0.83 | 0.065/0.83 | 0.130/0.86 | 0.065/0.84 | 0.078/0.85 | 0.052/0.84 |
| val_hd_07 | 0.156/0.84 | 0.133/0.84 | 0.200/0.85 | 0.200/0.84 | 0.178/0.86 | 0.156/0.86 |
| val_hg_04 | 0.132/0.80 | 0.053/0.83 | 0.000/0.81 | – | – | 0.079/0.80 |
| val_hi_02 | 0.078/0.82 | 0.000/0.87 | 0.020/0.83 | – | – | 0.078/0.83 |
| **hard mean** | 0.151 | 0.122 | 0.159 | 0.150 | 0.138 | 0.117 |

**Results:**
- **Seed noise is large.** Same text, different seed: hard mean ranges 0.122–0.159; `val_hg_04`
  0.132 → 0.053 → 0.000; `val_hi_02` 0.078 → 0.000 → 0.020. The `val_hi_02` "regression vs baseline"
  noted in `IDEA_NOTES.md` is seed noise.
- **Commas in numbers:** 0.150 / 0.138 vs 0.144 averaged over the plain seeds. No effect.
- **Guidance 3.0:** 0.117, within the seed range from one seed. Inconclusive.

Source: `out/exp_score.txt`, audio in `out/exp/`, script `code/experiments/exp.py`.

---

## 4. Picking among candidates with a second recognizer (no new audio)

**What:** for each of the 6 items, pick one of the 3 seed candidates using an independent recognizer,
`ai4bharat/indicwav2vec-hindi` (greedy decoding), by its CER against the text fed to the model (not
the `spoken` reference). Then score the picked file with the kit's Whisper.

| id | s0 | s1 | s2 | picked | picked CER | best possible |
|---|---|---|---|---|---|---|
| val_hd_01 | 0.095 | 0.127 | 0.143 | s1 | 0.127 | 0.095 |
| val_hd_03 | 0.224 | 0.163 | 0.163 | s1 | 0.163 | 0.163 |
| val_hd_06 | 0.130 | 0.065 | 0.130 | s0 | 0.130 | 0.065 |
| val_hd_07 | 0.156 | 0.133 | 0.200 | s0 | 0.156 | 0.133 |
| val_hg_04 | 0.132 | 0.053 | 0.000 | s0 | 0.132 | 0.000 |
| val_hi_02 | 0.078 | 0.000 | 0.020 | s0 | 0.078 | 0.000 |
| **mean** | | | | | **0.131** | 0.076 (avg of seeds 0.112) |

**Results:**
- **No gain:** picking with IndicWav2Vec gives 0.131, worse than the seed average (0.112). The
  "best possible" 0.076 is only reachable by picking with the same recognizer that scores, which the
  final scoring (unseen recognizers) won't reward.
- **IndicWav2Vec cannot hear numbers:** its vocabulary has no digits, so every spoken number comes out
  as `<unk>`. It cannot judge hard items.
- **The recognizers disagree on the same audio:** IndicWav2Vec scores all three seeds of `val_hg_04`
  at 0.026 and of `val_hi_02` at 0.020, while Whisper swings 0.000–0.132. Most of the seed-to-seed
  CER differences on ordinary words are Whisper's own uncertainty, not worse audio.

Source: `out/exp_rerank.txt`, script `code/experiments/rerank.py`.

---

## 5. Reference transcripts vs what Whisper hears (verifier check only)

| prompt | CER (transcript vs Whisper) | likely transcript issues (to confirm by ear) |
|---|---|---|
| V1 | 0.155 | ऑडर (ऑर्डर?), "वो अभी तक" heard as "वो अब भी तक" |
| V2 | 0.053 | small |
| V3 | 0.123 | "जाना पड़ता" heard as "जाना पड़ता है" |
| V4 | 0.106 | "संस्कृति के **सात**" heard as "**साथ**" |

Correcting a reference transcript is allowed by the rules. V4 also has the lowest SIM on every item
(0.74–0.80). **Not tested yet.**

Source: end of `out/exp_score.txt`.

---

## 5b. Round 2: V4 transcript fix and guidance 3.0 over 3 seeds

Script `code/experiments/exp2.py`, scores in `out/exp2_score.txt`. Each cell is scored by two recognizers:
Whisper (CER vs `spoken`) and IndicWav2Vec (CER vs model text, with number words removed because it
can't transcribe numbers).

**V4 transcript fix** ("संस्कृति के सात" → "के साथ"), V4's 5 short items:

| | orig s0 | orig s1 | fix s0 | fix s1 |
|---|---|---|---|---|
| Whisper CER | 0.048 | 0.041 | 0.052 | 0.041 |
| IndicWav2Vec CER | 0.075 | 0.070 | 0.080 | 0.070 |
| SIM | 0.78 | 0.77 | 0.78 | 0.77 |

**Result: no effect.** V4's low SIM comes from the speaker, not the transcript. V4's short items are also
very stable across seeds (4 of 5 identical CER).

**Guidance strength 3.0 vs 2.0**, 6-item set, means over seeds 0–2:

| | Whisper CER | Whisper CER (hard) | IndicWav2Vec CER | SIM |
|---|---|---|---|---|
| guidance 2.0 (the idea) | 0.112 | 0.144 | 0.068 | 0.847 |
| guidance 3.0 | **0.099** | **0.121** | **0.056** | 0.833 |

**Result: about 12% lower CER on both recognizers, SIM −0.014.** Small set (6 items), and a decoding
setting (a "baseline" under the rules, not an idea). Not part of the finalized idea; candidate for
Q3, needs a 24-item check.

---

## 5c. Round 3: sampler methods from recent papers (6 items × seeds 0–2)

Script `code/experiments/exp3.py` (wraps `f5_tts.model.cfm.odeint`; weights untouched), scores in
`out/exp3_score.txt`. Seeds 0–2 of the plain idea are reused from rounds 1–2 (same GPU).

- **PCI**, Prompt-Consistency Inference ([Zadorozhnyy et al. 2026, arXiv:2610.04757](https://arxiv.org/abs/2610.04757)):
  before every sampling step, reset the reference-clip frames to their exact noisy-interpolated value.
- **EPSS-7**, Empirically Pruned Step Sampling ([Zheng et al., Interspeech 2025, arXiv:2505.19931](https://arxiv.org/abs/2505.19931)):
  7 steps on the non-uniform grid SS([0,2,4,6,8,16,24,32]/32), sway −1.
- **nfe8:** plain 8 uniform steps, for comparison with EPSS-7.

| method (mean of 3 seeds) | Whisper CER | Whisper hard | IndicWav2Vec CER | SIM | synth time/item |
|---|---|---|---|---|---|
| idea (32 steps) | 0.112 | 0.144 | 0.068 | 0.846 | ≈ 28 s |
| PCI (32 steps) | 0.095 | 0.127 | 0.069 | 0.847 | 28.3 s |
| **EPSS-7** | **0.104** | **0.129** | **0.064** | 0.844 | **6.2 s** |
| nfe8 | 0.118 | 0.151 | 0.070 | 0.837 | 7.1 s |

**Results:**
- **PCI:** −15% on Whisper, no change on IndicWav2Vec or SIM. Only one recognizer agrees, so weak
  evidence. The paper's SIM gain is not reproduced here.
- **EPSS-7:** about 4.5× faster than 32 steps with no loss on either recognizer, and better than plain
  8 steps on both. Candidate for Task 4 (could reach about 4× speed-up instead of 1.92×). It changes the
  sampler's time grid through a code wrapper, so confirm in the public channel that this counts as a
  decoding setting. Needs a full 24-item check plus same-session timing before use.

---

## 5d. Final runs (local RTX 2050, `out/chain_rest.txt`)

| output | result |
|---|---|
| `predictions/t3_idea/` | 40 files (finalized idea, seed 0), format OK, all kit sanity checks pass; 2493 s |
| Task 4 timing (same session, GPU not shared) | baseline 1060.4 s, 16 steps 551.8 s → **speed-up 1.92×** |
| Task 4 validation (`out/val_fast_nfe16.json`) | CER **0.0632** (limit 1.2 × 0.0613 = 0.0736 ✓), SIM 0.843 |
| `predictions/t4_fast/` | 40 files, 16 steps, all sanity checks pass |

Timing files for `code/` (required by the rules): `out/baseline_val_timing.json`,
`out/val_fast_nfe16_timing.json`.

---

## 5e. Task 4 candidates (24 val items, seed 0)

Runner `code/experiments/run_fast.py` (fewer steps and/or autocast around the sampler; weights untouched).
Limit: CER ≤ 1.2 × 0.0613 = 0.0736. Speed-ups below divide by the 1060.4 s baseline timed earlier the
same day; they are **not** same-session numbers (this laptop's timings drift with heat, e.g. the
12-step run took 735 s, slower than 16 steps did earlier).

| candidate | val CER | hard | SIM | synth time | approx. speed-up | verdict |
|---|---|---|---|---|---|---|
| 16 steps fp32 (current `t4_fast`) | 0.0632 | 0.111 | 0.843 | 551.8 s (same session) | 1.92× | passes |
| 12 steps fp32 | 0.0738 | 0.137 | 0.845 | 734.6 s | – | **fails** (by 0.0002) |
| 10 steps fp32 | 0.0856 | 0.152 | 0.843 | 345.1 s | – | **fails** |
| 8 steps fp32 | 0.0862 | 0.121 | 0.843 | – | – | **fails** |
| fp16 autocast | – | – | – | – | – | **broken**: NaN audio (overflow) |
| **bf16, 32 steps** | **0.0625** | 0.112 | 0.842 | 353.3 s | ≈ 3.0× | passes, quality ≈ the idea (0.0613) |
| **bf16, 16 steps** | **0.0692** | 0.129 | 0.844 | 177.9 s | ≈ 6.0× | passes, small margin (ratio 1.13) |

**Results:**
- Cutting steps below 16 fails the CER limit; the losses concentrate in hard, hindi and long items.
- bf16 autocast (the GPU's 16-bit format with a wide range) gives about 3× at the same step count with
  CER essentially unchanged. fp16 overflows and produces invalid audio.
- bf16 + 16 steps is about 6× but sits 0.0044 under the limit, within single-seed noise.

**Rules:** bf16 changes numerical precision, not weights or settings the model sees. It is not a listed
decoding setting, so mention it in the README (and ask in the channel if unsure). Using it in the
submission needs a `--bf16` option in `code/run_system.py` (the README must reproduce `t4_fast`),
then a same-session baseline + candidate timing and a regenerated `t4_fast`.

**Final choice (run 15 Oct, `out/t4_bf16_chain.txt`):** bf16, 32 steps. Same session, GPU not shared:
baseline 1437.7 s, bf16 515.3 s → **speed-up 2.79×**; val CER **0.0625** (limit 0.0736 ✓), SIM 0.842.
`predictions/t4_fast/` regenerated with `code/run_system.py --bf16` (40 files, all checks pass). The
previous 16-step files are kept in `out/t4_fast_nfe16_backup/`.

---

## 6. Other checks

- **Vocabulary coverage:** every character in the model input texts and prompt transcripts is in
  IndicF5's vocabulary (`checkpoints/vocab.txt`, 104 Devanagari characters). No silent unknown tokens.
- **Long-text splitting:** F5's `chunk_text` (`utils_infer.py:75`) splits only after `; : , . ! ?`
  plus a space, never at the danda (।). Long items are cut at commas. Not tested.

---

## Summary

| Experiment | Outcome |
|---|---|
| Syllable-based duration | Noise (one flipping item) |
| Commas inside numbers | No effect |
| Guidance strength 3.0 | Inconclusive (one seed) |
| Picking among candidates with IndicWav2Vec | No gain; can't judge numbers |
| 16 steps (Task 4) | Passes, 1.57× |
| 8 steps (Task 4) | Timing invalid, re-run needed |
| Reference transcript fix (V4 साथ) | No effect |
| Guidance 3.0, 3 seeds | About −12% CER on both recognizers, SIM −0.014 (6 items) |
| PCI sampler | −15% on Whisper only; weak |
| EPSS-7 sampler | Same quality at about 4.5× speed; Task 4 candidate |
| 16 steps, same-session timing | 1.92×, CER 0.0632 ✓ |
| **bf16, 32 steps (chosen for Task 4)** | **2.79×, CER 0.0625 ✓** |

**Takeaways for report Q2/Q3:** single-seed differences below about 0.03–0.05 are not interpretable on
6–8 items; the remaining hinglish/hindi error is mostly recognizer disagreement.

## Still to do

1. Copy `out/baseline_val_timing.json` and `out/val_fast_bf16_timing.json` into `code/`; fill the Task 4
   fields in `results.json` (bf16, 32 steps; 1437.7 s / 515.3 s = 2.79×; val CER 0.0625).
2. Mention bf16 in `code/README.md`; ask in the public channel if unsure it is allowed.
3. idea.md part (4) and report.md: your own words (rule). Pointers: `DRAFT_idea_report.md`.
