"""Small, fast experiments on a few validation items. Audio cached in out/exp/<cond>/<id>.wav.
    python exp.py synth   # synthesize missing conditions (+ timing profile)
    python exp.py score   # score everything with the kit metrics + transcribe val prompts
"""
import csv, re, sys, time, json
from pathlib import Path
import numpy as np

import os
KIT = Path(os.environ.get("TTS_KIT", r"D:\inter_iit\TTS"))
sys.path[:0] = [str(KIT), str(KIT / "code")]
from normalize import normalize
from translit import translit_text
from run_system import peak_guard

OUT = KIT / "out" / "exp"
IDS = ["val_hd_01", "val_hd_03", "val_hd_06", "val_hd_07", "val_hg_04", "val_hi_02"]
HARD = IDS[:4]

SCALE = r"(लाख|करोड़|हज़ार)"
NOT_NUM = r"(?!रुपये|रुपए|का|की|के|को|से|में|तक|पर|है|हैं|छात्रों)"


def idea_text(t):
    return translit_text(normalize(t))


def numcomma(t):
    """Comma after a scale word when another number word follows (phrase break inside numbers)."""
    return re.sub(SCALE + r" " + NOT_NUM, r"\1, ", idea_text(t))


# name -> (text_fn, overrides, seed, ids)
CONDS = {
    "idea_s1": (idea_text, {}, 1, IDS),
    "idea_s2": (idea_text, {}, 2, IDS),
    "numcomma_s0": (numcomma, {}, 0, HARD),
    "numcomma_s1": (numcomma, {}, 1, HARD),
    "cfg3_s0": (idea_text, {"cfg_strength": 3.0}, 0, IDS),
}
EXISTING = {"idea_s0": KIT / "predictions" / "val_idea"}   # already made, seed 0, same machine


def rd(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


ITEMS = {r["id"]: r for r in rd(KIT / "data" / "val_items.csv")}
PROMPTS = {r["prompt_id"]: dict(r, path=str(KIT / "data" / r["file"])) for r in rd(KIT / "data" / "prompts.csv")}


def synth():
    from baseline.tts import get_engine, save_wav
    eng = get_engine()
    timers = {"prep": 0.0, "sample": 0.0, "vocoder": 0.0}

    def wrap(obj, name, key):
        f = getattr(obj, name)
        def g(*a, **k):
            import torch
            torch.cuda.synchronize(); t = time.perf_counter()
            r = f(*a, **k)
            torch.cuda.synchronize(); timers[key] += time.perf_counter() - t
            return r
        setattr(obj, name, g)
    wrap(eng.U, "preprocess_ref_audio_text", "prep")
    wrap(eng.ema, "sample", "sample")
    wrap(eng.vocoder, "decode", "vocoder")

    for c, (fn, ov, seed, ids) in CONDS.items():
        for i in ids:
            p = OUT / c / f"{i}.wav"
            if p.exists():
                continue
            it = ITEMS[i]; pr = PROMPTS[it["prompt_id"]]
            text = fn(it["text"])
            for k in timers: timers[k] = 0.0
            t0 = time.perf_counter()
            y = eng(text, pr["path"], pr["transcript"], seed=seed, **ov)
            tot = time.perf_counter() - t0
            save_wav(p, peak_guard(y))
            print(f"{c:<13}{i}  {tot:5.1f}s  prep {timers['prep']:.1f} sample {timers['sample']:.1f} "
                  f"voc {timers['vocoder']:.1f}  | {text}", flush=True)


def score():
    import metrics
    m = metrics.Models()
    items = [ITEMS[i] for i in IDS]
    dirs = {**EXISTING, **{c: OUT / c for c in CONDS}}
    res = {}
    for c, d in dirs.items():
        ids = CONDS[c][3] if c in CONDS else IDS
        rows = metrics.evaluate_folder(d, [ITEMS[i] for i in ids], PROMPTS, m)
        res[c] = {r["id"]: r for r in rows}
    print(f"\n{'id':<11}" + "".join(f"{c:>16}" for c in dirs))
    for i in IDS:
        print(f"{i:<11}" + "".join(
            f"{res[c][i]['cer']:>10.3f}/{res[c][i]['sim']:.2f}" if i in res[c] else f"{'-':>16}" for c in dirs))
    print(f"{'MEAN(own)':<11}" + "".join(
        f"{np.mean([r['cer'] for r in res[c].values()]):>10.3f}/{np.mean([r['sim'] for r in res[c].values()]):.2f}" for c in dirs))
    print(f"{'MEAN(hard)':<11}" + "".join(
        f"{np.mean([res[c][i]['cer'] for i in HARD]):>16.3f}" for c in dirs))
    for c in dirs:
        print(f"\n--- {c}")
        for i in res[c]:
            print(f"{i:<11}{res[c][i]['cer']:.3f}  {res[c][i]['reasons'] or res[c][i]['hyp']}")
    print("\n--- val prompt transcripts vs Whisper (verifier check only)")
    for pid in ["V1", "V2", "V3", "V4"]:
        pr = PROMPTS[pid]
        y, _ = metrics.load_audio(pr["path"])
        hyp = m.transcribe([y])[0]
        print(f"{pid} CER {metrics.cer(pr['transcript'], hyp):.3f}\n  kit: {pr['transcript']}\n  asr: {hyp}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"synth": synth, "score": score}[sys.argv[1]]()
