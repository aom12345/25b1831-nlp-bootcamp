#!/usr/bin/env python3
"""
How well does IndicF5's length estimate fit the speech it has to produce?

IndicF5 fixes the output length before generating:
    gen_seconds = ref_seconds * gen_chars / ref_chars          (chars = non-space code points)
This script compares that with a syllable-based need:
    need_seconds = gen_syllables / (ref_syllables / ref_seconds)   (the speaker's own syllable rate)
ratio = allocated / need.  1.0 = right;  < 1 = rushed (risk of skipped words);  > 1 = drawn out.

    python code/duration_analysis.py                  # val items, three text versions
"""
import csv, re, sys, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
sys.path.insert(0, str(HERE))
from normalize import normalize          # noqa: E402
from translit import translit_text       # noqa: E402

CONS = re.compile(r"[क-हक़-य़]़?")
IND_VOWEL = re.compile(r"[ऄ-औॲ]")
HALANT = "्"


def dev_syllables(word):
    """Akshara count: independent vowels + consonants that are not followed by a halant."""
    w = unicodedata.normalize("NFC", word)
    n = len(IND_VOWEL.findall(w))
    for m in CONS.finditer(w):
        nxt = w[m.end():m.end() + 1]
        if nxt != HALANT:
            n += 1
    # Hindi drops the inherent vowel of a word-final bare consonant (schwa deletion)
    if len(w) > 1 and CONS.fullmatch(w[-1]) and len(re.findall(r"[ा-ौॢॣ]", w)) + n > 1:
        n -= 1
    return max(n, 1)


def latin_syllables(word):
    w = word.lower()
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1                                # silent final e
    return max(n, 1)


def syllables(text):
    total = 0
    for tok in re.findall(r"[A-Za-z]+|[ऀ-ॿ]+", text):
        total += latin_syllables(tok) if tok[0].isascii() else dev_syllables(tok)
    return total


def chars(text):
    return len("".join(text.split()))


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import soundfile as sf
    prompts = {r["prompt_id"]: r for r in read_csv(KIT / "data" / "prompts.csv")}
    ref = {}
    for pid, r in prompts.items():
        dur = sf.info(str(KIT / "data" / r["file"])).duration
        ref[pid] = dict(dur=dur, chars=chars(r["transcript"]) + 1,     # wrapper appends ". "
                        syl=syllables(r["transcript"]))
        ref[pid]["chars_per_syl"] = ref[pid]["chars"] / ref[pid]["syl"]
    print("Reference clips: characters per syllable (the exchange rate the formula assumes)")
    for pid in sorted(ref):
        x = ref[pid]
        print(f"  {pid}: {x['dur']:.1f}s  {x['chars']} chars  {x['syl']} syl  "
              f"{x['chars_per_syl']:.2f} chars/syl  {x['syl'] / x['dur']:.1f} syl/s")

    items = read_csv(KIT / "data" / "val_items.csv")
    versions = {"raw (Latin, digits)": lambda t: t, "Task 2 (normalized)": normalize,
                "Task 2 + translit": lambda t: translit_text(normalize(t))}
    rows = []
    for it in items:
        r = ref[it["prompt_id"]]
        for vname, fn in versions.items():
            t = fn(it["text"])
            alloc = r["dur"] * chars(t) / r["chars"]
            need = syllables(t) / (r["syl"] / r["dur"])
            rows.append(dict(id=it["id"], cat=it["category"], version=vname, text=t, chars=chars(t),
                             syl=syllables(t), alloc=alloc, need=need, ratio=alloc / need))

    print("\nMean allocated/needed time, by category (1.00 = right, <1 rushed, >1 drawn out)")
    cats = ["hindi", "hinglish", "hard", "long"]
    print(f"{'version':<24}" + "".join(f"{c:>10}" for c in cats))
    for v in versions:
        vals = []
        for c in cats:
            rs = [x["ratio"] for x in rows if x["version"] == v and x["cat"] == c]
            vals.append(sum(rs) / len(rs))
        print(f"{v:<24}" + "".join(f"{x:>10.2f}" for x in vals))

    print("\nPer English word: characters per syllable, Latin spelling vs our Devanagari spelling")
    words = sorted({w for it in items if it["category"] == "hinglish" for w in re.findall(r"[A-Za-z]+", it["text"])})
    print(f"{'word':<10}{'latin c/s':>10}{'deva':>12}{'deva c/s':>10}")
    for w in words:
        d = translit_text(w)
        print(f"{w:<10}{len(w) / latin_syllables(w):>10.2f}{d:>12}{chars(d) / dev_syllables(d):>10.2f}")

    print("\nItems furthest from 1.00 (Task 2 + translit)")
    tr = sorted((x for x in rows if x["version"] == "Task 2 + translit"), key=lambda x: abs(x["ratio"] - 1), reverse=True)
    for x in tr[:8]:
        print(f"  {x['id']:<10} {x['cat']:<9} ratio {x['ratio']:.2f}  alloc {x['alloc']:.1f}s  need {x['need']:.1f}s")

    out = KIT / "out" / "duration_analysis.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-item table: {out}")


if __name__ == "__main__":
    main()
