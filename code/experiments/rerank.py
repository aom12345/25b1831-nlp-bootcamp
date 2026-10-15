"""Rerank existing seed candidates with an independent recognizer, then score picks with the kit's Whisper.
The verifier compares its transcript with the text we fed the model (no `spoken` used for picking).
    python rerank.py [model_id]
"""
import sys, re, unicodedata
from pathlib import Path
import numpy as np

KIT = Path(r"D:\inter_iit\TTS")
sys.path[:0] = [str(KIT), str(KIT / "code"), str(Path(__file__).parent)]
import exp, metrics

CANDS = {"s0": KIT / "predictions" / "val_idea", "s1": exp.OUT / "idea_s1", "s2": exp.OUT / "idea_s2"}
MODEL = sys.argv[1] if len(sys.argv) > 1 else "ai4bharat/indicwav2vec-hindi"


def load_verifier():
    if MODEL == "kit":                      # the kit's own Whisper (metrics.py)
        m = metrics.Models()
        return lambda y: m.transcribe([y])[0]
    import torch
    from transformers import AutoProcessor, AutoModelForCTC
    from transformers import Wav2Vec2Processor
    proc = (AutoProcessor if "mms" in MODEL else Wav2Vec2Processor).from_pretrained(MODEL)
    model = AutoModelForCTC.from_pretrained(MODEL, torch_dtype=torch.float16 if "mms" in MODEL else None)
    if "mms" in MODEL:
        proc.tokenizer.set_target_lang("hin"); model.load_adapter("hin")
    model = model.to("cuda").eval()

    def tr(y):
        with torch.inference_mode():
            x = proc(y, sampling_rate=16000, return_tensors="pt").input_values.to("cuda", dtype=model.dtype)
            ids = model(x).logits.argmax(-1)
        return proc.batch_decode(ids)[0]
    return tr


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tr = load_verifier()
    vcer = {}
    for c, d in CANDS.items():
        for i in exp.IDS:
            y, _ = metrics.load_audio(d / f"{i}.wav")
            hyp = tr(y)
            vcer[c, i] = metrics.cer(exp.idea_text(exp.ITEMS[i]["text"]), hyp)
            print(f"{c} {i} verifierCER {vcer[c, i]:.3f}  {hyp}")
    import gc, torch
    del tr; gc.collect(); torch.cuda.empty_cache()

    m = metrics.Models()
    wcer = {}
    for c, d in CANDS.items():
        rows = metrics.evaluate_folder(d, [exp.ITEMS[i] for i in exp.IDS], exp.PROMPTS, m)
        for r in rows:
            wcer[c, r["id"]] = r["cer"]
    print(f"\n{'id':<11}" + "".join(f"{c:>8}" for c in CANDS) + "   pick  pickCER  oracle")
    picks, orac = [], []
    for i in exp.IDS:
        pick = min(CANDS, key=lambda c: (vcer[c, i], c))
        picks.append(wcer[pick, i]); orac.append(min(wcer[c, i] for c in CANDS))
        print(f"{i:<11}" + "".join(f"{wcer[c, i]:>8.3f}" for c in CANDS) + f"   {pick:>4}  {wcer[pick, i]:.3f}   {orac[-1]:.3f}")
    hard = slice(0, 4)
    print(f"\nWhisper CER  mean-of-seeds {np.mean([wcer[c, i] for c in CANDS for i in exp.IDS]):.3f}   "
          f"verifier-pick {np.mean(picks):.3f}   oracle {np.mean(orac):.3f}")
    print(f"hard only    mean-of-seeds {np.mean([wcer[c, i] for c in CANDS for i in exp.HARD]):.3f}   "
          f"verifier-pick {np.mean(picks[hard]):.3f}   oracle {np.mean(orac[hard]):.3f}")


if __name__ == "__main__":
    main()
