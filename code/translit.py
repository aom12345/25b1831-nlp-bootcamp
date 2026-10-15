#!/usr/bin/env python3
"""
Task 3: rule-based English -> Devanagari spelling, the way an Indian speaker says the word.

No lookup table and no model: the word is scanned left to right with spelling rules
(digraphs, silent letters, magic-e, open/closed syllables, Indian-English vowels, retroflex
t/d), turned into a phoneme sequence, and rendered into Devanagari (matras, halant clusters,
anusvara before stops). The rules were developed on English words from the validation items
only (see DEV_WORDS) and on general English phonics; they are meant to apply to unseen words.

    python code/translit.py                 # score the rules on DEV_WORDS
    python code/translit.py walking proper  # convert words
    translit_text("आज office में internet slow है")   # convert every Latin word in a sentence
"""
import re, sys

# ---------------------------------------------------------------------------------------------
# Phoneme inventory -> Devanagari
# ---------------------------------------------------------------------------------------------
CONS = {"k": "क", "g": "ग", "ch": "च", "j": "ज", "t": "ट", "d": "ड", "th": "थ", "dh": "द",
        "n": "न", "p": "प", "b": "ब", "m": "म", "y": "य", "r": "र", "l": "ल", "v": "व",
        "s": "स", "sh": "श", "h": "ह", "f": "फ़", "z": "ज़"}
VOW = {"a": ("अ", ""), "aa": ("आ", "ा"), "i": ("इ", "ि"), "ii": ("ई", "ी"), "u": ("उ", "ु"),
       "uu": ("ऊ", "ू"), "e": ("ए", "े"), "ai": ("ऐ", "ै"), "o": ("ओ", "ो"), "au": ("ऑ", "ॉ")}
NASAL = "N"                       # anusvara
HALANT = "्"
STOPS = {"k", "g", "ch", "j", "t", "d", "th", "dh", "p", "b", "s", "sh"}
VL = "aeiou"
# Function words whose "th" is voiced (general English, not taken from any item list).
VOICED_TH = {"the", "this", "that", "these", "those", "they", "them", "then", "than", "there",
             "their", "though", "with", "other", "mother", "father", "brother", "weather"}


def _is_v(w, i):
    """Is w[i] a vowel letter? y counts as a vowel after a consonant."""
    if i < 0 or i >= len(w):
        return False
    if w[i] in VL:
        return True
    return w[i] == "y" and i > 0 and w[i - 1] not in VL


def _n_vowel_groups(w):
    return len(re.findall(r"[aeiou]+|(?<=[^aeiou])y", w))


def _context(w, j):
    """What follows a vowel that ends at j: ('end'|'magic'|'open'|'closed', consonant letters)."""
    rest = w[j:]
    if not rest:
        return "end", ""
    m = re.match(r"[^aeiouy]*", rest)
    cons = m.group(0)
    after = rest[len(cons):]
    # magic e: one consonant, then e at the end (optionally e+s / e+d)
    if len(cons) == 1 and cons not in "wx" and re.fullmatch(r"e[sd]?", after):
        return "magic", cons
    for dg in ("ch", "sh", "th", "ph", "ck", "ng", "gh"):
        cons = cons.replace(dg, "#")
    if after == "" and not cons:
        return "end", ""
    if len(cons) == 1 and after:
        return "open", cons
    return "closed", cons


def to_phonemes(word):
    w = re.sub(r"[^a-z]", "", word.lower())
    out, i, n = [], 0, len(w)
    mono = _n_vowel_groups(w) <= 1

    def at(s, k=None):
        return w.startswith(s, i if k is None else k)

    while i < n:
        c = w[i]
        prev = w[i - 1] if i else ""
        # ---------- multi-letter units ----------
        if at("tion") or at("ssion"):
            out += ["sh", "a", "n"]; i += 4 + at("ssion"); continue
        if at("sion"):
            out += (["zh"] if _is_v(w, i - 1) else ["sh"]) + ["a", "n"]; i += 4; continue
        if at("ture"):
            out += ["ch", "a", "r"]; i += 4; continue
        if (at("cial") or at("tial")) and i + 4 == n:
            out += ["sh", "a", "l"]; i += 4; continue
        if at("ous") and i + 3 == n:
            out += ["a", "s"]; i += 3; continue
        # unstressed final syllables of longer words: doctor, hotel, total, seven, album
        m = re.match(r"(or|el|al|en|um|us)$", w[i:])
        if m and not mono and i > 0 and not _is_v(w, i - 1):
            out += ["a", m.group(0)[1]]; i += 2; continue
        # unstressed com-/con- prefix: company, computer, control
        if i == 0 and re.match(r"co[mn][^aeiouy]", w) and not mono:
            out += ["k", "a"]; i += 2; continue
        if at("eigh"):
            out += ["e"]; i += 4; continue
        if at("igh"):
            out += ["aa", "i"]; i += 3; continue
        if at("ough"):
            out += ["au"]; i += 4; continue
        if c == "l" and at("le") and i + 2 == n and i > 0 and not _is_v(w, i - 1):
            out += ["a", "l"]; i += 2; continue
        if at("ed") and i + 2 == n and i >= 2 and re.search("[aeiouy]", w[:i - 1]):
            out += (["e", "d"] if prev in "td" else ["d"]); i += 2; continue
        if at("es") and i + 2 == n and i >= 2 and re.search("[aeiouy]", w[:i - 1]):
            out += (["i", "z"] if prev in "sxz" or w[i - 2:i] in ("sh", "ch") else ["s"]); i += 2; continue
        # ---------- vowels ----------
        if c in VL or (c == "y" and i > 0 and not _is_v(w, i - 1) and w[i - 1] != "y"):
            k = i
            # r-coloured vowels before a consonant or the end
            m = re.match(r"(ar|or|er|ir|ur|ear|eer|our|oor)(?![aeiouy])|(are|ire|ere|ore|ure)$", w[i:])
            if m:
                g = m.group(0)
                tail = {"ar": ["aa", "r"], "or": ["au", "r"], "er": ["a", "r"], "ir": ["a", "r"],
                        "ur": ["a", "r"], "ear": ["ii", "y", "a", "r"], "eer": ["ii", "y", "a", "r"],
                        "our": ["aa", "u", "a", "r"], "oor": ["au", "r"], "are": ["e", "y", "a", "r"],
                        "ire": ["aa", "i", "y", "a", "r"], "ere": ["ii", "y", "a", "r"],
                        "ore": ["au", "r"], "ure": ["y", "uu", "r"]}[g]
                if g == "ure" and prev in "rlj":
                    tail = ["uu", "r"]
                out += tail; i += len(g); continue
            two = w[i:i + 2]
            end2 = i + 2 == n
            if two == "ee":
                out += ["ii"]; i += 2; continue
            if two == "ea":
                out += ["ii"]; i += 2; continue
            if two == "oo":
                out += ["uu"]; i += 2; continue
            if two == "ou":
                out += (["uu"] if prev == "y" else ["aa", "u"]); i += 2; continue
            if two == "ow":
                out += (["o"] if end2 and not mono else ["aa", "u"] if end2 else ["aa", "u"])
                if end2 and mono and prev in "lnr" or end2 and i >= 2 and not _is_v(w, i - 2) and i >= 2 and w[i - 2] not in VL and prev in "lr":
                    out[-2:] = ["o"]
                i += 2; continue
            if two in ("ai", "ay"):
                out += ["e"]; i += 2; continue
            if two == "ey" and end2 and not mono:
                out += ["ii"]; i += 2; continue
            if two in ("ei", "ey"):
                out += ["e"]; i += 2; continue
            if two == "oa":
                out += ["o"]; i += 2; continue
            if two in ("oi", "oy"):
                out += ["au", "y"]; i += 2; continue
            if two in ("au", "aw"):
                out += ["au"]; i += 2; continue
            if two == "ew" or (two == "ue" and end2):
                out += (["uu"] if prev in "rlj" or w[max(0, i - 2):i] == "ch" else ["y", "uu"]); i += 2; continue
            if two == "ui":
                out += ["uu"]; i += 2; continue
            if two == "ie":
                out += (["aa", "i"] if end2 and mono else ["ii"]); i += 2; continue
            if two == "ia":
                out += ["i", "y", "aa"]; i += 2; continue
            if two == "io":
                out += ["i", "y", "o"]; i += 2; continue
            if two == "eo":
                out += (["i", "y", "o"] if end2 else ["ii"]); i += 2; continue
            # a before l+consonant / ll: walk, talk, all, call, small, salt
            if c == "a" and re.match(r"a(lk|ll|lt)", w[i:]):
                out += ["au"]; i += 1 + (w[i + 1:i + 3] == "lk"); continue
            ctx, cons = _context(w, i + 1)
            if c == "e" and ctx == "end":
                if mono or i == 0:
                    out += ["ii"]
                i += 1; continue                       # silent final e
            if c == "a":
                if ctx == "magic":
                    v = ["e"]
                elif ctx == "end":
                    v = ["aa"]
                elif ctx == "closed" and re.match(r"(ss$|st|sk|sp|ft|th)", w[i + 1:]):
                    v = ["aa"]                           # class, fast, ask, after, path
                elif prev == "w" and re.match(r"(t|sh|tch|nt|s)", w[i + 1:]):
                    v = ["au"]                           # water, wash, watch, want
                elif re.match(r"[ts]ion", w[i + 1:]) or ctx == "open" and re.match(r"[^aeiou](er|le|ble|tion|sion|ing$|y$)", w[i + 1:]):
                    v = ["e"]                            # paper, table, nation, making, lady
                elif prev == "w" and re.match(r"(t|sh|tch|nt|s)", w[i + 1:]):
                    v = ["au"]                           # water, wash, watch, want
                else:
                    v = ["ai"]                           # cat, message, cancel, family
            elif c == "e":
                v = ["ii"] if ctx == "magic" else ["e"]
            elif c in "iy":
                if ctx == "magic" and not mono and re.fullmatch(r"(c|v|t)es?", w[i + 1:]):
                    v = ["i"]                            # office, notice, active, favourite
                elif ctx == "magic" or (ctx == "end" and mono) or re.match(r"(nd|ld)$", w[i + 1:]):
                    v = ["aa", "i"]                      # time, my, fly, find, child
                elif ctx == "end":
                    v = ["ii"]                           # taxi, happy, city
                elif c == "y" and ctx == "open" and mono:
                    v = ["aa", "i"]
                else:
                    v = ["i"]
            elif c == "o":
                if ctx in ("magic", "end"):
                    v = ["o"]                            # phone, mobile, hello, video
                elif ctx == "closed":
                    if re.match(r"n$", w[i + 1:]) and not mono:
                        v = ["a"]                        # person, lemon, button
                    else:
                        v = ["au"]                       # office, shop, doctor, online
                else:
                    v = ["o"]                            # hotel, total, photo
            else:  # u
                if ctx == "closed" and not re.match(r"(ll|sh|t$)", w[i + 1:]):
                    v = ["a"]                            # bus, lunch, cut, club
                elif ctx == "closed":
                    v = ["u"]                            # full, push, put
                elif prev in ("r", "l", "j") or w[max(0, i - 2):i] in ("ch", "sh") or prev in "stdz":
                    v = ["uu"]                           # rule, june, student
                else:
                    v = ["y", "uu"]                      # music, computer, unit
            out += v; i += 1; continue
        # ---------- consonants ----------
        if at("tch"):
            out += ["ch"]; i += 3; continue
        if at("sch") or at("chr"):
            out += ["s", "k"] if at("sch") else ["k"]; i += 2 + at("sch"); continue
        if at("ch"):
            out += ["ch"]; i += 2; continue
        if at("sh"):
            out += ["sh"]; i += 2; continue
        if at("ph"):
            out += ["f"]; i += 2; continue
        if at("th"):
            out += ["dh" if w in VOICED_TH else "th"]; i += 2; continue
        if at("gh"):
            out += ["g"] if i == 0 else []; i += 2; continue
        if at("ck"):
            out += ["k"]; i += 2; continue
        if at("qu"):
            out += ["k", "v"]; i += 2; continue
        if at("wh"):
            out += ["h"] if at("who") else ["v"]; i += 2; continue
        if i == 0 and (at("kn") or at("wr") or at("gn") or at("ps")):
            i += 1; continue
        if at("mb") and i + 2 == n:
            out += ["m"]; i += 2; continue
        if at("gn") and i + 2 == n:
            out += ["n"]; i += 2; continue
        if at("dg"):
            out += ["j"]; i += 2; continue
        if at("ng") and w[i + 2:i + 3] in ("e", "i", "y") and i + 3 < n:
            out += ["n", "j"]; i += 2; continue           # engine, change, danger
        if at("ng"):
            out += [NASAL, "g"]; i += 2; continue
        if at("nk"):
            out += [NASAL, "k"]; i += 2; continue
        if c == prev and c not in VL:              # doubled letters: ss, pp, ll, tt, ...
            i += 1; continue
        if c == "c":
            out += ["s"] if w[i + 1:i + 2] in ("e", "i", "y") else ["k"]
        elif c == "g":
            out += ["j"] if w[i + 1:i + 2] in ("e", "i", "y") and i > 0 else ["g"]
        elif c == "x":
            out += ["z"] if i == 0 else ["k", "s"]
        elif c == "q":
            out += ["k"]
        elif c == "w":
            out += ["v"]
        elif c == "y":
            out += ["y"]
        elif c in CONS:
            out += [c]
        i += 1
    return out


def render(ph):
    """Phonemes -> Devanagari. Consonant clusters get a halant; n/m before a stop -> anusvara."""
    s = []
    for k, p in enumerate(ph):
        nxt = ph[k + 1] if k + 1 < len(ph) else None
        if p == NASAL:
            s.append("ं")
            continue
        if p in ("n", "m") and nxt in STOPS and k > 0 and ph[k - 1] in VOW:
            s.append("ं")
            continue
        if p in VOW:
            after_cons = k > 0 and (ph[k - 1] in CONS or ph[k - 1] == "zh")
            if after_cons:
                s.append(VOW[p][1])
            else:
                s.append(VOW[p][0])
            continue
        s.append(CONS.get(p, "ज़" if p == "zh" else ""))
        if nxt is not None and (nxt in CONS or nxt == "zh"):
            s.append(HALANT)
    return "".join(s)


def translit_word(word: str) -> str:
    return render(to_phonemes(word))


RE_LATIN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def translit_text(text: str) -> str:
    """Replace every Latin-script word in `text` with its rule-based Devanagari spelling."""
    return RE_LATIN.sub(lambda m: translit_word(m.group(0)), text)


# ---------------------------------------------------------------------------------------------
# Development words: ONLY from validation items (the 17 Latin words of the hinglish items, with
# the spellings of their `spoken` references) plus English loanwords that the validation text and
# validation prompt transcripts already write in Devanagari.
# ---------------------------------------------------------------------------------------------
DEV_WORDS = {
    # val hinglish items (text -> spoken)
    "office": "ऑफिस", "internet": "इंटरनेट", "slow": "स्लो", "new": "न्यू", "address": "एड्रेस",
    "message": "मैसेज", "flight": "फ्लाइट", "cancel": "कैंसिल", "hotel": "होटल", "online": "ऑनलाइन",
    "class": "क्लास", "walking": "वॉकिंग", "proper": "प्रॉपर", "sleep": "स्लीप", "weekend": "वीकेंड",
    "movie": "मूवी", "shopping": "शॉपिंग",
    # loanwords already in Devanagari in val texts / val prompt transcripts
    "doctor": "डॉक्टर", "station": "स्टेशन", "platform": "प्लेटफ़ॉर्म", "train": "ट्रेन",
    "mobile": "मोबाइल", "phone": "फ़ोन", "screen": "स्क्रीन", "mood": "मूड", "school": "स्कूल",
    "bill": "बिल", "company": "कंपनी", "pass": "पास", "rafting": "राफ्टिंग", "market": "मार्केट",
}


def _score():
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from metrics import canon, _edit_distance
    tot_e = tot_n = exact = soft = 0
    nohal = lambda x: canon(x).replace(HALANT, "")
    for en, ref in DEV_WORDS.items():
        got = translit_word(en)
        e, r = _edit_distance(canon(ref), canon(got)), canon(ref)
        tot_e += e; tot_n += len(r); exact += e == 0; soft += nohal(ref) == nohal(got)
        print(f"{'ok ' if e == 0 else 'XX '}{en:<10} {got:<12} {ref:<12} {' '.join(to_phonemes(en))}")
    print(f"\nexact {exact}/{len(DEV_WORDS)}   same ignoring halant {soft}/{len(DEV_WORDS)}   "
          f"char error {tot_e / tot_n:.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1:
        for w in sys.argv[1:]:
            print(w, translit_word(w))
    else:
        _score()
