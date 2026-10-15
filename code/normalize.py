#!/usr/bin/env python3
"""
Task 2: rule-based text frontend for IndicF5 (no models).

normalize(text) rewrites digits, rupee amounts, percentages, dates, times and ALL-CAPS
abbreviations into Hindi words in Devanagari, as they would be spoken. English words written
in Latin script are left unchanged (Task 3 deals with their pronunciation).

    python code/normalize.py                     # writes predictions/t2_normalized.csv (64 rows)
    python code/normalize.py --text "₹2,340.50"  # normalize one string

All rules are general; the few examples that shaped them come from data/val_items.csv only.
"""
import argparse, csv, re, sys, unicodedata
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------------------------
# Number words. Hindi 0-99 is irregular, so it is a table.
# ---------------------------------------------------------------------------------------------
_0_99 = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ "
    "दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
    "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस "
    "तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस "
    "चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास "
    "पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ "
    "साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर "
    "सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उनासी "
    "अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी "
    "नब्बे इक्यानवे बानवे तिरानवे चौरानवे पचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे"
).split()
assert len(_0_99) == 100

DIGIT_WORDS = _0_99[:10]
MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर",
          "अक्टूबर", "नवंबर", "दिसंबर"]
LETTER_NAMES = dict(zip(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "ए बी सी डी ई एफ जी एच आई जे के एल एम एन ओ पी क्यू आर एस टी यू वी डब्ल्यू एक्स वाई ज़ेड".split()))


def int_words(n: int) -> str:
    """Indian-system reading: करोड़, लाख, हज़ार, सौ."""
    if n < 100:
        return _0_99[n]
    out = []
    crore, n = divmod(n, 10 ** 7)
    lakh, n = divmod(n, 10 ** 5)
    thousand, n = divmod(n, 1000)
    hundred, rest = divmod(n, 100)
    if crore:
        out += [int_words(crore), "करोड़"]
    if lakh:
        out += [_0_99[lakh], "लाख"]
    if thousand:
        out += [_0_99[thousand], "हज़ार"]
    if hundred:
        out += [_0_99[hundred], "सौ"]
    if rest:
        out.append(_0_99[rest])
    return " ".join(out)


def year_words(y: int) -> str:
    """Years 1100-1999 are read in hundreds (उन्नीस सौ निन्यानवे); others as plain numbers."""
    if 1100 <= y <= 1999 and y % 1000 >= 100:
        hi, lo = divmod(y, 100)
        return f"{_0_99[hi]} सौ" + (f" {_0_99[lo]}" if lo else "")
    return int_words(y)


def digits_words(s: str) -> str:
    return " ".join(DIGIT_WORDS[int(c)] for c in s if c.isdigit())


def number_words(s: str) -> str:
    """A bare digit string (commas allowed) -> words. Leading zeros or very long -> digit by digit."""
    s = s.replace(",", "")
    if len(s) > 1 and s.startswith("0"):
        return digits_words(s)
    if len(s) > 9:                       # phone/account-like: read digit by digit
        return digits_words(s)
    n = int(s)
    if len(s) == 4:
        return year_words(n)
    return int_words(n)


def decimal_words(int_part: str, frac: str) -> str:
    return f"{number_words(int_part)} दशमलव {digits_words(frac)}"


# ---------------------------------------------------------------------------------------------
# Patterns (applied in this order)
# ---------------------------------------------------------------------------------------------
NUM = r"\d{1,3}(?:,\d{2})*,\d{3}|\d+"          # 1,25,000 / 125000 / 12
# NFC keeps nukta letters decomposed (ज़ = ज + ़), so inputs and patterns are both NFC.
SCALE = "(?:" + "|".join(unicodedata.normalize("NFC", w) for w in ("हज़ार", "हजार", "लाख", "करोड़", "करोड", "अरब")) + ")"

RE_RUPEE = re.compile(rf"(?:₹|\bRs\.?|रु\.)\s*({NUM})(?:\.(\d{{1,2}}))?(\s*{SCALE})?")
RE_DATE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})\b")
RE_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
RE_PERCENT = re.compile(rf"({NUM})(?:\.(\d+))?\s*%")
RE_DECIMAL = re.compile(rf"({NUM})\.(\d+)")
RE_RANGE = re.compile(r"\b(\d+)\s*[-–]\s*(\d+)\b")
RE_INT = re.compile(NUM)
RE_ACRONYM = re.compile(r"\b(?:[A-Z]\.){2,}|\b[A-Z]{2,6}\b")
RE_WORD_HYPHEN_NUM = re.compile(r"(?<=[^\W\d_])-(?=\d)")     # चंद्रयान-3 -> चंद्रयान 3
RE_LETTER_DIGIT = re.compile(r"(?<=\d)(?=[A-Z])|(?<=[A-Z])(?=\d)")  # 4G -> 4 G
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _rupee(m):
    whole, paise, scale = m.group(1), m.group(2), (m.group(3) or "").strip()
    if scale:                                     # ₹38 लाख -> अड़तीस लाख रुपये
        num = decimal_words(whole, paise) if paise else number_words(whole)
        return f"{num} {scale} रुपये"
    out = f"{int_words(int(whole.replace(',', '')))} रुपये"
    if paise and int(paise):
        out += f" {_0_99[int(paise.ljust(2, '0'))]} पैसे"
    return out


def _date(m):
    d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m.group(0)
    year = int(y) + 2000 if len(y) == 2 else int(y)
    return f"{int_words(d)} {MONTHS[mo - 1]} {year_words(year)}"


def _time(m):
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 24 or mi > 59:
        return m.group(0)
    return f"{int_words(h)} बजे" if mi == 0 else f"{int_words(h)} बजकर {int_words(mi)} मिनट"


def _percent(m):
    num = decimal_words(m.group(1), m.group(2)) if m.group(2) else number_words(m.group(1))
    return f"{num} प्रतिशत"


def _acronym(m):
    letters = [c for c in m.group(0) if c.isalpha()]
    return " ".join(LETTER_NAMES[c] for c in letters)


def normalize(text: str) -> str:
    s = unicodedata.normalize("NFC", text).translate(DEV_DIGITS)
    s = RE_WORD_HYPHEN_NUM.sub(" ", s)
    s = RE_LETTER_DIGIT.sub(" ", s)
    s = RE_RUPEE.sub(_rupee, s)
    s = RE_DATE.sub(_date, s)
    s = RE_TIME.sub(_time, s)
    s = RE_PERCENT.sub(_percent, s)
    s = RE_DECIMAL.sub(lambda m: decimal_words(m.group(1), m.group(2)), s)
    s = RE_RANGE.sub(lambda m: f"{number_words(m.group(1))} से {number_words(m.group(2))}", s)
    s = RE_INT.sub(lambda m: number_words(m.group(0)), s)
    s = RE_ACRONYM.sub(_acronym, s)
    s = s.replace("&", " और ").replace("₹", " रुपये ").replace("%", " प्रतिशत ")
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------------------------
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="normalize one string and print it")
    ap.add_argument("--out", default=str(KIT / "predictions" / "t2_normalized.csv"))
    a = ap.parse_args()
    if a.text is not None:
        sys.stdout.reconfigure(encoding="utf-8")
        print(normalize(a.text))
        return
    rows = []
    for split, name in (("val", "val_items.csv"), ("eval", "eval_items.csv")):
        for it in read_csv(KIT / "data" / name):
            rows.append({"id": it["id"], "split": split, "text": it["text"],
                         "normalized_text": normalize(it["text"])})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split", "text", "normalized_text"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
