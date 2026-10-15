"""Round 3: sampler-level methods from recent papers, on the 6-item set, seeds 0-2.
  pci    Prompt-Consistency Inference (Zadorozhnyy et al. 2026, arXiv:2610.04757): before every velocity
         evaluation, reset the prompt frames to t*f + (1-t)*eps (their exact conditional path).
  epss7  Empirically Pruned Step Sampling (Zheng et al. 2025, arXiv:2505.19931): 7 steps on the grid
         SS([0,2,4,6,8,16,24,32]/32) with sway coef -1.
  nfe8   plain 8 uniform steps (comparison for epss7)
Implemented by wrapping f5_tts.model.cfm.odeint; model weights untouched.
    python exp3.py synth
    python exp3.py score
"""
import os, sys, time, json, gc
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp
from exp import ITEMS, PROMPTS, OUT, KIT, idea_text

MODE = {"name": None}


def install_wrapper():
    import torch
    import f5_tts.model.cfm as cfm
    orig = cfm.odeint

    def wrapped(fn, y0, t, **kw):
        name = MODE["name"] or ""
        if name.startswith("epss7"):
            base = torch.tensor([0, 2, 4, 6, 8, 16, 24, 32], device=t.device, dtype=t.dtype) / 32
            t = base + (-1.0) * (torch.cos(torch.pi / 2 * base) - 1 + base)
        if name.startswith("pci"):
            free = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))
            cond = free["step_cond"]                         # prompt mel, zeros elsewhere
            pmask = (cond.abs().sum(-1, keepdim=True) > 0)   # prompt frames
            inner = fn

            def fn(tt, x):
                xp = tt * cond + (1 - tt) * y0               # exact conditional path, same noise
                return inner(tt, torch.where(pmask, xp, x))
        return orig(fn, y0, t, **kw)
    cfm.odeint = wrapped


CONDS = {}
for s in (0, 1, 2):
    if os.environ.get("TTS_KIT"):            # Kaggle: regenerate the plain idea on the same GPU
        CONDS[f"idea_s{s}"] = (s, {})
    CONDS[f"pci_s{s}"] = (s, {})
    CONDS[f"epss7_s{s}"] = (s, {})
    CONDS[f"nfe8_s{s}"] = (s, {"nfe_step": 8})
OUT = OUT.parent / "exp3"
# locally the plain-idea seeds already exist (same GPU): reuse them
BASE = {} if os.environ.get("TTS_KIT") else {
    "idea_s0": KIT / "predictions" / "val_idea", "idea_s1": OUT.parent / "exp" / "idea_s1",
    "idea_s2": OUT.parent / "exp" / "idea_s2"}


def synth():
    from baseline.tts import get_engine, save_wav
    from run_system import peak_guard
    import torch
    eng = get_engine()
    install_wrapper()
    times = {}
    for c, (seed, ov) in CONDS.items():
        MODE["name"] = c
        for i in exp.IDS:
            p = OUT / c / f"{i}.wav"
            if p.exists():
                continue
            it = ITEMS[i]; pr = PROMPTS[it["prompt_id"]]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            y = eng(idea_text(it["text"]), pr["path"], pr["transcript"], seed=seed, **ov)
            torch.cuda.synchronize(); times[f"{c}/{i}"] = round(time.perf_counter() - t0, 2)
            save_wav(p, peak_guard(y))
            print(f"{c:<10}{i}  {times[f'{c}/{i}']:.1f}s", flush=True)
    MODE["name"] = None
    (OUT / "exp3_times.json").write_text(json.dumps(times, indent=1))


def score():
    sys.stdout.reconfigure(encoding="utf-8")
    import exp2, metrics, torch
    tr, model = exp2.w2v()
    dirs = {**BASE, **{c: OUT / c for c in CONDS}}
    wv = {}
    for c, d in dirs.items():
        for i in exp.IDS:
            y, _ = metrics.load_audio(d / f"{i}.wav")
            wv[c, i] = exp2.w2v_cer(idea_text(ITEMS[i]["text"]), tr(y))
    del tr, model; gc.collect(); torch.cuda.empty_cache()
    m = metrics.Models()
    wh, sim = {}, {}
    for c, d in dirs.items():
        for r in metrics.evaluate_folder(d, [ITEMS[i] for i in exp.IDS], PROMPTS, m):
            wh[c, r["id"]], sim[c, r["id"]] = r["cer"], r["sim"]
    print(f"{'condition':<10}{'Whisper all':>12}{'Whisper hard':>13}{'IW2V':>8}{'SIM':>7}")
    for c in dirs:
        print(f"{c:<10}{np.mean([wh[c, i] for i in exp.IDS]):>12.3f}{np.mean([wh[c, i] for i in exp.HARD]):>13.3f}"
              f"{np.mean([wv[c, i] for i in exp.IDS]):>8.3f}{np.mean([sim[c, i] for i in exp.IDS]):>7.3f}")
    print("\nper seed-averaged method:")
    for meth in ["idea", "pci", "epss7", "nfe8"]:
        cs = [c for c in dirs if c.startswith(meth + "_s")]
        print(f"{meth:<10}{np.mean([wh[c, i] for c in cs for i in exp.IDS]):>12.3f}"
              f"{np.mean([wh[c, i] for c in cs for i in exp.HARD]):>13.3f}"
              f"{np.mean([wv[c, i] for c in cs for i in exp.IDS]):>8.3f}"
              f"{np.mean([sim[c, i] for c in cs for i in exp.IDS]):>7.3f}")
    print("\nper item (Whisper CER, seed-mean):")
    print(f"{'id':<11}" + "".join(f"{m_:>8}" for m_ in ["idea", "pci", "epss7", "nfe8"]))
    for i in exp.IDS:
        print(f"{i:<11}" + "".join(
            f"{np.mean([wh[c, i] for c in dirs if c.startswith(m_ + '_s')]):>8.3f}" for m_ in ["idea", "pci", "epss7", "nfe8"]))
    t = json.loads((OUT / "exp3_times.json").read_text())
    print("\nmean synth time per item:", {m_: round(np.mean([v for k, v in t.items() if k.startswith(m_ + "_s")]), 1)
                                          for m_ in ["pci", "epss7", "nfe8"]})


if __name__ == "__main__":
    {"synth": synth, "score": score}[sys.argv[1]]()
