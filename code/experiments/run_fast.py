"""Task 4 candidates: the idea with fewer steps and/or fp16/bf16 autocast around the sampler.
Times exactly like code/run_system.py (model loaded first, one warm-up excluded).
    python run_fast.py --out out/t4c_nfe12 --nfe 12
    python run_fast.py --out out/t4c_bf16_nfe16 --nfe 16 --half bf16 --ids val_hd_01,val_hg_04
"""
import argparse, json, sys, time
from pathlib import Path

KIT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KIT), str(KIT / "code")]
from normalize import normalize
from translit import translit_text
from run_system import peak_guard, read_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--half", choices=["fp16", "bf16"], help="autocast the sampler (fp16 gives NaN audio here)")
    ap.add_argument("--ids")
    ap.add_argument("--items", default=str(KIT / "data" / "val_items.csv"))
    a = ap.parse_args()
    items = read_csv(a.items)
    if a.ids:
        keep = set(a.ids.split(","))
        items = [it for it in items if it["id"] in keep]
    prompts = {r["prompt_id"]: dict(r, path=str(KIT / "data" / r["file"])) for r in read_csv(KIT / "data" / "prompts.csv")}

    import torch
    from baseline.tts import get_engine, save_wav
    eng = get_engine()
    if a.half:
        f = eng.ema.sample
        dtype = torch.float16 if a.half == "fp16" else torch.bfloat16
        def sample16(*args, **kw):
            with torch.autocast("cuda", dtype=dtype):
                return f(*args, **kw)
        eng.ema.sample = sample16
    ov = {"nfe_step": a.nfe}
    p = prompts[items[0]["prompt_id"]]
    eng("नमस्ते, यह एक छोटा सा परीक्षण है।", p["path"], p["transcript"], seed=0, **ov)   # warm-up

    out = KIT / a.out
    out.mkdir(parents=True, exist_ok=True)
    per, total = {}, 0.0
    for it in items:
        p = prompts[it["prompt_id"]]
        torch.cuda.synchronize(); t = time.time()
        y = eng(translit_text(normalize(it["text"])), p["path"], p["transcript"], seed=0, **ov)
        torch.cuda.synchronize(); dt = time.time() - t
        total += dt; per[it["id"]] = round(dt, 3)
        save_wav(out / f"{it['id']}.wav", peak_guard(y))
        print(f"{it['id']}: {dt:.2f}s", flush=True)
    timing = {"total_synth_s": round(total, 3), "n_items": len(items), "nfe_step": a.nfe, "half": a.half,
              "gpu": torch.cuda.get_device_name(0), "per_item_s": per}
    (KIT / "out" / f"{out.name}_timing.json").write_text(json.dumps(timing, indent=1))
    print(f"Done: {len(items)} items in {total:.1f}s")


if __name__ == "__main__":
    main()
