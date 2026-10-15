#!/usr/bin/env python3
"""
Cheap pilots: synthesize a few validation items under several text/decoding variants, then
score them with the kit's metrics (same CER/SIM code as metrics.py) next to val_control.

    python code/pilot.py --variants translit oracle --ids val_hg_01,val_hg_02
    python code/pilot.py --variants translit --category hinglish

Audio is cached in out/pilot/<variant>/<id>.wav, so re-running only scores.
Pilots are exploration on validation items only; they are not submission files.
"""
import argparse, csv, gc, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(HERE))

from normalize import normalize            # noqa: E402
from translit import translit_text          # noqa: E402
from run_system import peak_guard, read_csv  # noqa: E402

PILOT = KIT / "out" / "pilot"
RE_LATIN_RUN = re.compile(r"[A-Za-z]+(?:\s+[A-Za-z]+)*")


def _isolate(text):
    """Commas around each run of Latin words (a prosodic boundary), then transliterate."""
    t = RE_LATIN_RUN.sub(lambda m: f", {m.group(0)},", normalize(text))
    t = re.sub(r"\s*,(\s*,)*\s*", ", ", t)
    t = re.sub(r"\s*,\s*([।?!.])", r"\1", t)
    return translit_text(t).strip(" ,")


# name -> fn(item) -> (model_text, overrides)
VARIANTS = {
    "translit": lambda it: (translit_text(normalize(it["text"])), {}),
    "oracle": lambda it: (it["spoken"].split("||")[0].strip(), {}),     # diagnostic ceiling only
    "isolate": lambda it: (_isolate(it["text"]), {}),
    "norm_slow": lambda it: (normalize(it["text"]), {"speed": 0.85}),   # diagnostic: is it rushing?
    # syllable-based length: speed = (time the char formula gives) / (time the syllables need)
    "latin_syldur": lambda it: _syldur(normalize(it["text"]), it),
    "translit_syldur": lambda it: _syldur(translit_text(normalize(it["text"])), it),
}


def _syldur(text, it):
    from duration_analysis import syllables, chars
    p = _PROMPTS[it["prompt_id"]]
    ref_chars, ref_syl = chars(p["transcript"]) + 1, syllables(p["transcript"])
    speed = (chars(text) / ref_chars) / (syllables(text) / ref_syl)
    return text, {"speed": round(min(max(speed, 0.7), 1.4), 3)}


_PROMPTS = {r["prompt_id"]: r for r in read_csv(KIT / "data" / "prompts.csv")}


def synthesize(variants, items, prompts, seed):
    todo = [(v, it) for v in variants for it in items
            if not (PILOT / v / f"{it['id']}.wav").exists()]
    if not todo:
        return
    from baseline.tts import get_engine, save_wav
    engine = get_engine()
    for k, (v, it) in enumerate(todo, 1):
        text, ov = VARIANTS[v](it)
        p = prompts[it["prompt_id"]]
        y = engine(text, p["path"], p["transcript"], seed=seed, **ov)
        save_wav(PILOT / v / f"{it['id']}.wav", peak_guard(y))
        print(f"[{k}/{len(todo)}] {v} {it['id']}: {text}", flush=True)
    import torch, baseline.tts as bt
    bt._ENGINE = None
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--ids")
    ap.add_argument("--category")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", default=str(KIT / "predictions" / "val_control"))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    items = read_csv(KIT / "data" / "val_items.csv")
    if a.ids:
        keep = set(a.ids.split(","))
        items = [it for it in items if it["id"] in keep]
    if a.category:
        items = [it for it in items if it["category"] == a.category]
    pbase = KIT / "data"
    prompts = {r["prompt_id"]: dict(r, path=str(pbase / r["file"])) for r in read_csv(pbase / "prompts.csv")}

    synthesize(a.variants, items, prompts, a.seed)

    import metrics
    models = metrics.Models()
    cols = {"control": Path(a.control), **{v: PILOT / v for v in a.variants}}
    results = {}
    for name, d in cols.items():
        rows = metrics.evaluate_folder(d, items, prompts, models)   # works on any item subset
        results[name] = {r["id"]: r for r in rows}

    print(f"\n{'id':<11}" + "".join(f"{n:>18}" for n in cols))
    for it in items:
        print(f"{it['id']:<11}" + "".join(
            f"{results[n][it['id']]['cer']:>11.3f}/{results[n][it['id']]['sim']:.2f}" for n in cols))
    import numpy as np
    print(f"{'MEAN':<11}" + "".join(
        f"{np.mean([r['cer'] for r in results[n].values()]):>11.3f}/{np.mean([r['sim'] for r in results[n].values()]):.2f}"
        for n in cols))
    for n in a.variants:
        print(f"\n--- {n} transcripts")
        for it in items:
            r = results[n][it["id"]]
            print(f"{it['id']:<11}{r['cer']:.3f}  {r['reasons'] or r['hyp']}")


if __name__ == "__main__":
    main()
