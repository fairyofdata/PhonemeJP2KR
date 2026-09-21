"""Deterministic Hangul → katakana transliteration (display only).

The katakana line shows the acoustic channel the way a Japanese reader
would spell it. It is a *notation* of that channel, not a channel of its
own, and it deliberately loses what this system measures: lenis /
aspirated / tense onsets share one kana row, ㅓ and ㅗ both become オ,
ㅡ and ㅜ both become ウ, and ㄴ and ㅇ codas both become ン. Showing that
loss is the point — it is why the learner cannot hear the difference.

It transliterates the *surface* jamo the scorer compared (after G2P), one
syllable at a time, so every kana fragment maps back onto the alignment.
Voicing follows the IPA rule in src/g2p.py: a lenis onset is voiced after
a vowel or a sonorant coda (도시 → トシ, 그리고 → クリゴ).

Vowel / coda / glide choices follow common Japanese-textbook practice;
they are a fixed table, so the same input always yields the same kana.
"""

_VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")

# Korean vowel → (glide, base vowel)
_VOWEL_KEY = {
    "ㅏ": ("", "a"), "ㅐ": ("", "e"), "ㅓ": ("", "o"), "ㅔ": ("", "e"),
    "ㅗ": ("", "o"), "ㅜ": ("", "u"), "ㅡ": ("", "u"), "ㅣ": ("", "i"),
    "ㅑ": ("y", "a"), "ㅒ": ("y", "e"), "ㅕ": ("y", "o"), "ㅖ": ("y", "e"),
    "ㅛ": ("y", "o"), "ㅠ": ("y", "u"),
    "ㅘ": ("w", "a"), "ㅙ": ("w", "e"), "ㅚ": ("w", "e"), "ㅞ": ("w", "e"),
    "ㅝ": ("w", "o"), "ㅟ": ("w", "i"),
    "ㅢ": ("ui", ""),
}

# kana row → a i u e o
_ROWS = {
    "": "アイウエオ", "k": "カキクケコ", "g": "ガギグゲゴ", "s": "サシスセソ",
    "n": "ナニヌネノ", "h": "ハヒフヘホ", "b": "バビブベボ", "p": "パピプペポ",
    "m": "マミムメモ", "r": "ラリルレロ",
}
_ROWS_2 = {  # rows whose kana need two characters
    "t": ["タ", "ティ", "トゥ", "テ", "ト"], "d": ["ダ", "ディ", "ドゥ", "デ", "ド"],
    "ch": ["チャ", "チ", "チュ", "チェ", "チョ"], "j": ["ジャ", "ジ", "ジュ", "ジェ", "ジョ"],
}
_SMALL_Y = {"a": "ャ", "u": "ュ", "o": "ョ", "e": "ェ"}
_SMALL_W = {"a": "ァ", "i": "ィ", "e": "ェ", "o": "ォ"}
# the kana a y-/w-glide attaches to
_Y_BASE = {"t": "テ", "d": "デ", "ch": "チ", "j": "ジ", "": ""}
_W_BASE = {"t": "ト", "d": "ド", "ch": "チュ", "j": "ジュ", "": "ウ"}

# onset → (voiceless row, row when voiced between sonorants)
_ONSET_ROW = {
    "ㄱ": ("k", "g"), "ㄲ": ("k", "k"), "ㅋ": ("k", "k"),
    "ㄷ": ("t", "d"), "ㄸ": ("t", "t"), "ㅌ": ("t", "t"),
    "ㅂ": ("p", "b"), "ㅃ": ("p", "p"), "ㅍ": ("p", "p"),
    "ㅈ": ("ch", "j"), "ㅉ": ("ch", "ch"), "ㅊ": ("ch", "ch"),
    "ㅅ": ("s", "s"), "ㅆ": ("s", "s"),
    "ㄴ": ("n", "n"), "ㄹ": ("r", "r"), "ㅁ": ("m", "m"), "ㅎ": ("h", "h"),
    "": ("", ""),
}
_CODA_KANA = {"ㄱ": "ク", "ㄴ": "ン", "ㄷ": "ッ", "ㄹ": "ル", "ㅁ": "ム",
              "ㅂ": "プ", "ㅇ": "ン", "": ""}
_SONORANT_CODAS = {"ㄴ", "ㄹ", "ㅁ", "ㅇ", ""}
_COLUMN = "aiueo"


def _row_kana(row: str, vowel: str) -> str:
    col = _COLUMN.index(vowel)
    return _ROWS[row][col] if row in _ROWS else _ROWS_2[row][col]


def _core(row: str, vowel_jamo: str) -> str:
    glide, v = _VOWEL_KEY[vowel_jamo]
    if glide == "":
        return _row_kana(row, v)
    if glide == "ui":
        return _row_kana(row, "u") + "イ"
    if glide == "y":
        if row == "":
            return {"a": "ヤ", "u": "ユ", "o": "ヨ", "e": "イェ"}[v]
        if row in ("ch", "j") and v != "e":
            return _row_kana(row, v)            # 챠 → チャ
        base = _Y_BASE.get(row, _row_kana(row, "i") if row else "")
        return base + _SMALL_Y[v]
    # w-glide
    if row == "" and v == "a":
        return "ワ"
    base = _W_BASE.get(row, _row_kana(row, "u"))
    return base + _SMALL_W[v]


def syllable_kana(onset: str, vowel: str, coda: str, after_sonorant: bool) -> str:
    """One surface syllable → katakana. ``onset`` is "" for a silent ㅇ."""
    voiceless, voiced = _ONSET_ROW.get(onset, ("", ""))
    return _core(voiced if after_sonorant else voiceless, vowel) + _CODA_KANA.get(coda, "")


def syllables(jamo_pos) -> list:
    """[(jamo, source index)] → [(source index, onset, vowel, coda)].

    Jamo from one source character form one syllable; onset ㅇ is absent
    from the scoring unit, so a syllable whose first jamo is a vowel has
    an empty onset.
    """
    out = []
    for jamo, pos in jamo_pos:
        if not out or out[-1][0] != pos:
            out.append([pos, "", "", ""])
        syl = out[-1]
        if jamo in _VOWELS:
            syl[2] = jamo
        elif syl[2]:
            syl[3] = jamo
        else:
            syl[1] = jamo
    return [tuple(s) for s in out]


def kana_by_position(jamo_pos) -> dict:
    """{source index: katakana} for a jamo_positions() sequence.

    Voicing context runs across the sequence and resets only where source
    characters are not adjacent (a space or punctuation), so slicing the
    result by word gives the same kana as the full line.
    """
    kana, after_sonorant, prev = {}, False, None
    for pos, onset, vowel, coda in syllables(jamo_pos):
        if not vowel:        # malformed (no vowel) — nothing sensible to write
            continue
        if prev is not None and pos != prev + 1:
            after_sonorant = False
        kana[pos] = syllable_kana(onset, vowel, coda, after_sonorant)
        after_sonorant, prev = coda in _SONORANT_CODAS, pos
    return kana


def to_kana(text: str) -> str:
    """Orthographic Korean → katakana of its surface pronunciation."""
    from .g2p import jamo_positions

    by_pos = kana_by_position(jamo_positions(text))
    out, prev = [], None
    for i in sorted(by_pos):
        if prev is not None and i != prev + 1:
            out.append(" ")
        out.append(by_pos[i])
        prev = i
    return "".join(out)
