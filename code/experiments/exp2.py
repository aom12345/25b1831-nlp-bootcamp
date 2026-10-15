"""Round 2: V4 transcript fix and guidance 3.0 over more seeds. Audio cached in out/exp/<cond>/<id>.wav.
    python exp2.py synth
    python exp2.py score
Scores with the kit's Whisper (CER vs spoken) and IndicWav2Vec (CER vs model text; numbers ignored).
"""
import re, sys, gc
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp
from exp import ITEMS, PROMPTS, OUT, KIT, idea_text
import metrics

V4 = ["val_hi_04", "val_hg_02", "val_hg_06", "val_hd_04", "val_hd_08"]
V4_FIX = PROMPTS["V4"]["transcript"].replace("के सात ", "के साथ ")
assert V4_FIX != PROMPTS["V4"]["transcript"]

# name -> (seed, overrides, ids, transcript override or None)
CONDS = {
    "v4orig_s0": (0, {}, V4, None),          # = predictions/val_idea
    "v4orig_s1": (1, {}, V4, None),
    "v4fix_s0": (0, {}, V4, V4_FIX),
    "v4fix_s1": (1, {}, V4, V4_FIX),
    "idea_s0": (0, {}, exp.IDS, None),       # = predictions/val_idea
    "idea_s1": (1, {}, exp.IDS, None),       # round 1
    "idea_s2": (2, {}, exp.IDS, None),       # round 1
    "cfg3_s0": (0, {"cfg_strength": 3.0}, exp.IDS, None),   # round 1
    "cfg3_s1": (1, {"cfg_strength": 3.0}, exp.IDS, None),
    "cfg3_s2": (2, {"cfg_strength": 3.0}, exp.IDS, None),
}
EXISTING = {"v4orig_s0": KIT / "predictions" / "val_idea", "idea_s0": KIT / "predictions" / "val_idea"}


def folder(c):
    return EXISTING.get(c, OUT / c)


def synth():
    from baseline.tts import get_engine, save_wav
    from run_system import peak_guard
    eng = get_engine()
    for c, (seed, ov, ids, tr) in CONDS.items():
        if c in EXISTING:
            continue
        for i in ids:
            p = folder(c) / f"{i}.wav"
            if p.exists():
                continue
            it = ITEMS[i]; pr = PROMPTS[it["prompt_id"]]
            y = eng(idea_text(it["text"]), pr["path"], tr or pr["transcript"], seed=seed, **ov)
            save_wav(p, peak_guard(y))
            print(f"{c:<11}{i} done", flush=True)


def w2v():
    import torch
    from transformers import Wav2Vec2Processor, AutoModelForCTC
    name = "ai4bharat/indicwav2vec-hindi"
    proc = Wav2Vec2Processor.from_pretrained(name)
    model = AutoModelForCTC.from_pretrained(name).to("cuda").eval()

    def tr(y):
        with torch.inference_mode():
            x = proc(y, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
            return proc.batch_decode(model(x).logits.argmax(-1))[0]
    return tr, model


NUM = re.compile(r"(एक|दो|तीन|चार|पाँच|छह|सात|आठ|नौ|दस|बारह|बीस|पच्चीस|छब्बीस|अड़तीस|सैंतालीस|"
                 r"बयासी|इक्यावन|चालीस|पचास|बानवे|सौ|हज़ार|लाख|करोड़)")


def w2v_cer(text, hyp):
    """IndicWav2Vec writes numbers as <unk>: drop number words from the reference and <unk> from hyp."""
    return metrics.cer(NUM.sub(" ", text), hyp.replace("<unk>", " "))


def score():
    sys.stdout.reconfigure(encoding="utf-8")
    tr, model = w2v()
    wv = {}
    for c, (_, _, ids, _) in CONDS.items():
        for i in ids:
            y, _ = metrics.load_audio(folder(c) / f"{i}.wav")
            wv[c, i] = w2v_cer(idea_text(ITEMS[i]["text"]), tr(y))
    del tr, model; gc.collect()
    import torch; torch.cuda.empty_cache()

    m = metrics.Models()
    wh, sim, hyp = {}, {}, {}
    for c, (_, _, ids, _) in CONDS.items():
        for r in metrics.evaluate_folder(folder(c), [ITEMS[i] for i in ids], PROMPTS, m):
            wh[c, r["id"]], sim[c, r["id"]], hyp[c, r["id"]] = r["cer"], r["sim"], r["reasons"] or r["hyp"]

    def table(conds, ids, title):
        print(f"\n== {title}   cells: WhisperCER / IndicW2V CER / SIM")
        print(f"{'id':<11}" + "".join(f"{c:>22}" for c in conds))
        for i in ids:
            print(f"{i:<11}" + "".join(f"{wh[c, i]:>10.3f}/{wv[c, i]:.3f}/{sim[c, i]:.2f}" for c in conds))
        print(f"{'MEAN':<11}" + "".join(
            f"{np.mean([wh[c, i] for i in ids]):>10.3f}/{np.mean([wv[c, i] for i in ids]):.3f}/"
            f"{np.mean([sim[c, i] for i in ids]):.2f}" for c in conds))

    table(["v4orig_s0", "v4orig_s1", "v4fix_s0", "v4fix_s1"], V4, "V4 transcript fix (साथ)")
    table(["idea_s0", "idea_s1", "idea_s2", "cfg3_s0", "cfg3_s1", "cfg3_s2"], exp.IDS, "guidance 3.0 vs 2.0")
    print("\n== transcripts (Whisper)")
    for c in ["v4orig_s0", "v4orig_s1", "v4fix_s0", "v4fix_s1", "cfg3_s1", "cfg3_s2"]:
        for i in CONDS[c][2]:
            print(f"{c:<11}{i:<11}{wh[c, i]:.3f}  {hyp[c, i]}")


if __name__ == "__main__":
    {"synth": synth, "score": score}[sys.argv[1]]()
