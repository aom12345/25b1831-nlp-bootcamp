#!/usr/bin/env python3
"""
Checks normalize() against the `spoken` column of the validation items, using the same
canonicalization as metrics.py. Hinglish items are skipped (their references transliterate the
English words, which Task 2 deliberately leaves in Latin), but they must come back unchanged.
Also checks that no digit, ₹ or % survives in any of the 64 items and that every Latin word in
the output was already in the input.

    python code/test_normalize.py
"""
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from normalize import normalize, int_words, read_csv, KIT  # noqa: E402
from metrics import canon, alternatives                     # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
fails = 0


def check(name, ok, detail=""):
    global fails
    if not ok:
        fails += 1
        print(f"FAIL {name}\n     {detail}")


# number spot checks
for n, want in [(0, "शून्य"), (100, "एक सौ"), (101, "एक सौ एक"), (1000, "एक हज़ार"),
                (125000, "एक लाख पच्चीस हज़ार"), (12951, "बारह हज़ार नौ सौ इक्यावन"),
                (10 ** 7, "एक करोड़"), (125000000, "बारह करोड़ पचास लाख")]:
    got = int_words(n)
    check(f"int_words({n})", canon(got) == canon(want), f"{got!r} != {want!r}")

# validation references
for it in read_csv(KIT / "data" / "val_items.csv"):
    out = normalize(it["text"])
    if it["category"] == "hinglish":
        check(it["id"], out == re.sub(r"\s+", " ", it["text"]).strip(), out)
        continue
    best = min(alternatives(it["spoken"]), key=lambda r: canon(r) != canon(out))
    check(it["id"], canon(out) == canon(best), f"got  {out}\n     want {best}")

# all 64 items: no digits / ₹ / %, Latin only where the input had that word
for name in ("val_items.csv", "eval_items.csv"):
    for it in read_csv(KIT / "data" / name):
        out = normalize(it["text"])
        check(f"{it['id']} digits", not re.search(r"[0-9०-९₹%]", out), out)
        new_latin = set(re.findall(r"[A-Za-z]+", out)) - set(re.findall(r"[A-Za-z]+", it["text"]))
        check(f"{it['id']} latin", not new_latin, str(new_latin))
        leftover_caps = re.findall(r"\b[A-Z]{2,}\b", out)
        check(f"{it['id']} acronyms", not leftover_caps, str(leftover_caps))

print("all checks passed" if not fails else f"{fails} check(s) failed")
sys.exit(1 if fails else 0)
