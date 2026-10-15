#!/usr/bin/env python3
"""
Synthesize items with our text frontend on top of the baseline wrapper, and time the run
exactly like baseline/run_baseline.py (model loaded first, one warm-up item excluded).

    # Task 2 / Task 3 control: normalized text, baseline settings
    python code/run_system.py --mode norm  --items data/val_items.csv  --out predictions/val_control
    python code/run_system.py --mode norm  --items data/eval_items.csv --out predictions/t2_norm
    # Task 3 idea: normalized text + rule-based English -> Devanagari
    python code/run_system.py --mode idea  --items data/val_items.csv  --out predictions/val_idea
    python code/run_system.py --mode idea  --items data/eval_items.csv --out predictions/t3_idea
    # Task 4: same as idea, fewer flow-matching steps
    python code/run_system.py --mode idea  --items data/val_items.csv  --out out/val_fast --set nfe_step=16
    # Task 4 (chosen): same as idea, sampler in bfloat16, 32 steps
    python code/run_system.py --mode idea  --items data/val_items.csv  --out out/val_fast_bf16 --bf16

Writes <out>/<id>.wav, and out/<name>_timing.json and out/<name>_texts.csv (kept out of predictions/).
"""
import argparse, csv, json, platform, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(HERE))

from normalize import normalize        # noqa: E402
from translit import translit_text     # noqa: E402

MODES = {
    "baseline": lambda t: t,
    "norm": normalize,
    "idea": lambda t: translit_text(normalize(t)),
}


PEAK = 0.95


def peak_guard(y):
    """Loudness normalization (allowed post-processing): IndicF5 often produces samples above
    full scale, which save_wav hard-clips; enough clipped samples fail the sanity check. Scale the
    whole clip down when its peak exceeds PEAK; quieter clips are left untouched."""
    import numpy as np
    peak = float(np.abs(y).max()) if len(y) else 0.0
    return y * (PEAK / peak) if peak > PEAK else y


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=sorted(MODES), required=True)
    ap.add_argument("--prompts", default=str(KIT / "data" / "prompts.csv"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", help="comma-separated subset of item ids")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="decoding overrides")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--bf16", action="store_true",
                    help="Task 4: run the flow-matching sampler under bfloat16 autocast (weights unchanged)")
    a = ap.parse_args()

    items = read_csv(a.items)
    if a.ids:
        keep = set(a.ids.split(","))
        items = [it for it in items if it["id"] in keep]
    frontend = MODES[a.mode]
    for it in items:
        it["model_text"] = frontend(it["text"])
    pbase = Path(a.prompts).resolve().parent
    prompts = {r["prompt_id"]: dict(r, path=str(pbase / r["file"])) for r in read_csv(a.prompts)}
    overrides = {k: parse_value(v) for k, v in (kv.split("=", 1) for kv in a.set)}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    side = KIT / "out"
    side.mkdir(exist_ok=True)
    with open(side / f"{out.name}_texts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "model_text"], extrasaction="ignore")
        w.writeheader()
        w.writerows(items)

    import torch
    from baseline.tts import get_engine, save_wav
    t0 = time.time()
    engine = get_engine()
    load_s = time.time() - t0
    if a.bf16:
        # bfloat16 compute for the sampler only; fp16 overflows (NaN audio), bf16 keeps fp32's range
        sample = engine.ema.sample
        def sample_bf16(*args, **kw):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return sample(*args, **kw)
        engine.ema.sample = sample_bf16
    if not a.no_warmup and items:
        p = prompts[items[0]["prompt_id"]]
        engine("नमस्ते, यह एक छोटा सा परीक्षण है।", p["path"], p["transcript"], seed=a.seed, **overrides)

    per_item, total = {}, 0.0
    for i, it in enumerate(items, 1):
        p = prompts[it["prompt_id"]]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.time()
        y = engine(it["model_text"], p["path"], p["transcript"], seed=a.seed, **overrides)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t
        total += dt
        save_wav(out / f"{it['id']}.wav", peak_guard(y))
        per_item[it["id"]] = round(dt, 3)
        print(f"[{i}/{len(items)}] {it['id']}: {dt:.2f}s synth, {len(y) / 16000:.2f}s audio", flush=True)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    timing = {"total_synth_s": round(total, 3), "n_items": len(items), "model_load_s": round(load_s, 1),
              "seed": a.seed, "mode": a.mode, "overrides": overrides, "bf16": a.bf16, "gpu": gpu,
              "python": platform.python_version(), "items_file": str(a.items), "per_item_s": per_item}
    tpath = side / f"{out.name}_timing.json"
    tpath.write_text(json.dumps(timing, indent=1), encoding="utf-8")
    print(f"\nDone: {len(items)} items in {total:.1f}s of synthesis on {gpu}. Timing in {tpath}")


if __name__ == "__main__":
    main()
