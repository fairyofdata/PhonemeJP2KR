"""Phone-level comparison: a produced IPA string against the target.

The Hangul channel has a blind spot by construction: the ASR hypothesis
is Hangul, and it passes the same G2P as the target, so a learner who
skips a phonological rule (춥고도 read as [춥고도] instead of [춥꼬도])
is re-derived into the standard form and never flagged. An IPA string —
from a phoneme recognizer or typed in by hand — never goes through G2P,
so it can be compared with the target's surface IPA as produced.

Pipeline:
    tokenize()        IPA text → phone units (tɕʰ, k͈, jʌ, ɰi are one unit)
                      after normalizing allophones and notation variants
    target_phones()   target text → surface phones via the G2P, each with
                      its word index, coda flag, and — by aligning with a
                      letter-by-letter reading of the same text — the
                      phonological rule that produced it, if any
    compare()         Levenshtein alignment with articulatory costs, then
                      L1 tags: the Hangul-channel taxonomy plus
                      rule_*_missed when the learner produced the
                      letter-by-letter phone instead of the rule's output

Normalization merges what the comparison should not count: lenis
voicing (ɡ d b dʑ are allophones), ɕ before i/j, the liquid [ɾ]/[l]/[ɭ],
coda unrelease ([p̚] vs [p]), length, stress and syllable marks.
"""

import re
from dataclasses import dataclass, field

from .g2p import (
    _IPA_CODA, _IPA_ONSET, _IPA_VOWEL, _apply_liaison, _neutralize_codas, _tokenize,
    group_syllables, ipa_segments, jamo_positions,
)

# --- inventory and normalization --------------------------------------------

_VOWELS = ["ja", "jɛ", "jʌ", "je", "jo", "ju", "wa", "wɛ", "wʌ", "we", "wi", "ɰi",
           "a", "e", "ʌ", "o", "u", "ɯ", "i"]
_CONSONANTS = ["tɕʰ", "tɕ͈", "tɕ", "pʰ", "p͈", "tʰ", "t͈", "kʰ", "k͈", "s͈",
               "p", "t", "k", "s", "h", "m", "n", "ŋ", "L"]
_UNITS = sorted(_VOWELS + _CONSONANTS, key=len, reverse=True)
VOWELS = set(_VOWELS)

# notation variants → the inventory above (applied to the raw string, in order)
_REWRITES = [
    ("͡", ""), ("ː", ""), ("ˈ", ""), ("ˌ", ""), (".", ""), (" ", ""), ("̚", ""),
    ("ʨ", "tɕ"), ("ʥ", "tɕ"), ("dʑ", "tɕ"), ("dz", "tɕ"), ("tʃ", "tɕ"),
    ("dʒ", "tɕ"), ("ʃ", "s"), ("z", "s"),
    ("ɡ", "k"), ("g", "k"), ("d", "t"), ("b", "p"),
    ("*", "͈"), ("ʼ", "͈"), ("ˀ", "͈"),
    ("ɾ", "L"), ("ɭ", "L"), ("ɹ", "L"), ("r", "L"), ("l", "L"),
    ("ɛ", "e"), ("ə", "ʌ"), ("ɔ", "ʌ"), ("ɐ", "a"), ("ɨ", "ɯ"), ("ʉ", "u"),
    ("ɯᵝ", "ɯ"), ("ᵝ", ""), ("ʲ", "j"), ("ç", "h"), ("ɸ", "h"), ("ɦ", "h"),
    ("ɴ", "ŋ"), ("ɲ", "n"),
]


def normalize(text: str) -> str:
    for a, b in _REWRITES:
        text = text.replace(a, b)
    return re.sub(r"(?<!t)ɕ", "s", text)   # [ɕ] is /s/ before i/j; tɕ stays


def tokenize(text: str) -> list:
    """IPA text → phone units; characters outside the inventory are skipped."""
    s, out, i = normalize(text), [], 0
    while i < len(s):
        unit = next((u for u in _UNITS if s.startswith(u, i)), None)
        if unit:
            out.append(unit)
            i += len(unit)
        else:
            i += 1
    return out


# phoneme-recognizer tokens (espeak-style, multilingual) → our notation.
# Applied per token, so a "t" followed by an "s" token stays two phones.
_TOKEN_MAP = {
    "kh": "kʰ", "ph": "pʰ", "th": "tʰ", "tɕh": "tɕʰ", "tsh": "tɕʰ", "ts.h": "tɕʰ",
    "tʃʰ": "tɕʰ", "tS": "tɕ", "dZ": "tɕ", "ts": "tɕ", "ts.": "tɕ", "tʃ": "tɕ",
    "dʒ": "tɕ", "dʑ": "tɕ", "ʑ": "tɕ", "ʒ": "tɕ", "dz": "tɕ",
    # geminate/long stops are how this inventory comes closest to tense
    "kː": "k͈", "pː": "p͈", "tː": "t͈", "sː": "s͈", "tʃː": "tɕ͈", "dʒː": "tɕ͈",
    "dzː": "tɕ͈", "ɡː": "k͈", "bː": "p͈", "dː": "t͈",
    "x": "h", "χ": "h", "v": "w", "β": "w", "ʋ": "w", "y": "u", "ɪ": "i", "ʊ": "u",
    "ɑ": "a", "æ": "e", "ɣ": "k", "N": "ŋ", "ʔ": "", "S": "s",
}


def from_model_tokens(tokens) -> str:
    """Decoded phoneme-recognizer tokens → an IPA string compare() accepts."""
    out = []
    for tok in tokens:
        tok = re.sub(r"[1-5.\^\[]", "", tok) if tok not in _TOKEN_MAP else tok
        out.append(_TOKEN_MAP.get(tok, tok))
    return "".join(out)


def _family(p: str) -> str:
    """Place/manner family: phonation variants share one (k kʰ k͈ → k)."""
    return p.replace("ʰ", "").replace("͈", "")


# --- target phones ------------------------------------------------------------

@dataclass
class Phone:
    sym: str
    word: int = 0
    coda: bool = False
    # phones a reader who skips a rule would produce here → that rule
    missed: dict = field(default_factory=dict)


def _word_index(text: str) -> dict:
    idx, w, in_word = {}, -1, False
    for i, ch in enumerate(text):
        if ch.isspace():
            in_word = False
        else:
            if not in_word:
                w, in_word = w + 1, True
            idx[i] = w
    return idx


def _surface_phones(text: str, linked=()) -> list:
    words = _word_index(text)
    seq = jamo_positions(text, linked)
    segs = ipa_segments(seq)
    codas = set()
    k = 0
    for pos, cho, jung, jong in group_syllables(seq):
        k += bool(cho) + bool(jung)
        if jong:
            codas.add(k)
            k += 1
    out = []
    for n, ((_, pos), seg) in enumerate(zip(seq, segs)):
        for unit in tokenize(seg):
            out.append(Phone(unit, words.get(pos, 0), n in codas))
    return out


def _spelled_phones(text: str, liaison: bool, linked=()) -> list:
    """The text read without phonological rules: syllable by syllable with
    coda neutralization only, optionally with liaison (같이 → [가티])."""
    out = []
    for w, word in enumerate(_tokenize(text, linked)):
        if liaison:
            word = _apply_liaison(word)
        for cho, jung, jong in _neutralize_codas(word):
            out += [Phone(u, w) for u in tokenize(_IPA_ONSET[cho])]
            out += [Phone(u, w) for u in tokenize(_IPA_VOWEL[jung])]
            out += [Phone(u, w, coda=True) for u in tokenize(_IPA_CODA.get(jong, ""))]
    return out


def _rule_kind(target: Phone, spelled: Phone) -> str:
    t, s = target.sym, spelled.sym
    if _family(t) == _family(s) and "͈" in t and "͈" not in s:
        return "tensification"
    if _family(t) == _family(s) and "ʰ" in t and "ʰ" not in s:
        return "aspiration"
    if t in ("m", "n", "ŋ") and s not in ("m", "n", "ŋ"):
        return "nasalization"
    if t == "L" and s == "n":
        return "lateralization"
    if _family(t) == "tɕ" and _family(s) == "t":
        return "palatalization"
    return None


def target_phones(text: str, linked=()) -> list:
    """Surface phones of the target, each annotated with the phones a reader
    who skipped a rule would produce there (and which rule that was).

    ``linked`` are the word boundaries read as one phrase, as chosen by
    src/scoring.py, so both channels compare against the same target.
    """
    surface = _surface_phones(text, linked)
    syms = [p.sym for p in surface]
    liaised = _spelled_phones(text, liaison=True, linked=linked)
    for op, i, j in _align(syms, [p.sym for p in liaised]):
        if op == "sub" and (kind := _rule_kind(surface[i], liaised[j])):
            surface[i].missed[liaised[j].sym] = kind
    unlinked = _spelled_phones(text, liaison=False, linked=linked)
    for op, i, j in _align(syms, [p.sym for p in unlinked]):
        if op == "sub" and unlinked[j].coda and not surface[i].coda:
            surface[i].missed.setdefault(unlinked[j].sym, "liaison")
    return surface


# --- alignment ------------------------------------------------------------------

def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    av, bv = a in VOWELS, b in VOWELS
    if av != bv:
        return 1.2
    if av:
        return 0.6
    if _family(a) == _family(b):
        return 0.5
    nasal_of = {"p": "m", "t": "n", "k": "ŋ"}
    if {a, b} <= {"m", "n", "ŋ"} or nasal_of.get(_family(a)) == b or nasal_of.get(_family(b)) == a:
        return 0.6
    if {a, b} == {"L", "n"}:
        return 0.6
    return 1.0


def _align(ref: list, hyp: list) -> list:
    """Weighted Levenshtein → [(op, ref index | None, hyp index | None)]."""
    n, m = len(ref), len(hyp)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = float(i)
    for j in range(1, m + 1):
        dp[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + _sub_cost(ref[i - 1], hyp[j - 1]))
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(dp[i][j] - dp[i - 1][j - 1] - _sub_cost(ref[i - 1], hyp[j - 1])) < 1e-9:
            out.append(("match" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(dp[i][j] - dp[i - 1][j] - 1) < 1e-9:
            out.append(("del", i - 1, None))
            i -= 1
        else:
            out.append(("ins", None, j - 1))
            j -= 1
    out.reverse()
    return out


# --- comparison -------------------------------------------------------------------

@dataclass
class IpaReport:
    score: int
    target: list                       # list[Phone]
    produced: list                     # list[str]
    pairs: list = field(default_factory=list)   # [(op, ref idx, hyp idx)]
    error_tags: list = field(default_factory=list)


def _tag(ref: Phone, hyp: str, op: str) -> str:
    r = ref.sym if ref else ""
    if op == "ins":
        return "vowel_epenthesis" if hyp in ("ɯ", "u", "o") else "insertion"
    if op == "del":
        return "coda_deletion" if ref.coda and r not in VOWELS else "deletion"
    if hyp in ref.missed:
        return f"rule_{ref.missed[hyp]}_missed"
    pair = {r, hyp}
    if pair == {"ʌ", "o"}:
        return "vowel_ʌ_o_confusion"
    if pair == {"jʌ", "jo"}:
        return "vowel_jʌ_jo_confusion"
    if pair == {"ɯ", "u"}:
        return "vowel_ɯ_u_confusion"
    if r == "ɰi":
        return "diphthong_ɰi_monophthongization"
    if ref.coda and pair <= {"m", "n", "ŋ"}:
        return "nasal_coda_confusion"
    if ref.coda and pair <= {"p", "t", "k"}:
        return "stop_coda_confusion"
    if _family(r) == _family(hyp) and r not in VOWELS:
        return "laryngeal_confusion"
    return "substitution"


def compare(target_text: str, produced_ipa: str, linked=()) -> IpaReport:
    """Align a produced IPA string with the target's surface phones."""
    target = target_phones(target_text, linked)
    produced = tokenize(produced_ipa)
    if not target:
        return IpaReport(0, target, produced)
    pairs = _align([p.sym for p in target], produced)
    tags = []
    for op, i, j in pairs:
        if op == "match":
            continue
        ref = target[i] if i is not None else None
        hyp = produced[j] if j is not None else ""
        tags.append({"tag": _tag(ref, hyp, op), "unit": "ipa",
                     "ref": ref.sym if ref else "", "hyp": hyp})
    misses = sum(op != "match" for op, _, _ in pairs)
    score = max(0, round(100 * (1 - misses / max(len(target), len(produced)))))
    return IpaReport(score, target, produced, pairs, tags)


_VOICED = {"p": "b", "t": "d", "k": "ɡ", "tɕ": "dʑ"}
_SONORANTS = {"m", "n", "ŋ", "L"}
_FRONT = {"i", "ja", "jʌ", "jo", "ju", "je", "wi"}


def render(phones: list) -> list:
    """Normalized phones → display IPA, one string per phone.

    Puts back the allophony normalization removed, with the G2P's rules:
    lenis voicing between voiced sounds, [ɕ] before i/j, the liquid as [ɾ]
    before a vowel and [ɭ] elsewhere, and unreleased stops before a
    consonant or at the end — so the channel reads like the others.
    """
    out = []
    for k, p in enumerate(phones):
        prev = phones[k - 1] if k else ""
        nxt = phones[k + 1] if k + 1 < len(phones) else ""
        voiced_before = prev in VOWELS or prev in _SONORANTS
        if p in _VOICED and voiced_before and nxt in VOWELS:
            out.append(_VOICED[p])
        elif p in ("s", "s͈") and nxt in _FRONT:
            out.append("ɕ" if p == "s" else "ɕ͈")
        elif p == "L":
            out.append("ɾ" if nxt in VOWELS else "ɭ")
        elif p in ("p", "t", "k") and nxt not in VOWELS:
            out.append(p + "̚")
        else:
            out.append(p)
    return out


def _render_by_word(phones: list, words: list) -> list:
    out = []
    for w in dict.fromkeys(words):
        out += render([p for p, pw in zip(phones, words) if pw == w])
    return out


def split_by_word(report: IpaReport, n_words: int) -> list:
    """Per target word: produced phones, a [ref, hyp, op] diff, and tags.

    Insertions follow the previous word, as in src/words.py.
    """
    words = [{"phones": [], "diff": [], "tags": []} for _ in range(n_words)]
    # word of every produced phone, so display allophony resets per word
    # (as the G2P does at a space)
    hyp_word, cur = {}, 0
    for op, i, j in report.pairs:
        if i is not None:
            cur = report.target[i].word
        if j is not None:
            hyp_word[j] = cur
    ref_disp = _render_by_word([p.sym for p in report.target], [p.word for p in report.target])
    hyp_disp = _render_by_word(report.produced, [hyp_word[j] for j in range(len(report.produced))])
    cur, t = 0, iter(report.error_tags)
    for op, i, j in report.pairs:
        if i is not None:
            cur = report.target[i].word
        ref = ref_disp[i] if i is not None else ""
        hyp = hyp_disp[j] if j is not None else ""
        if hyp:
            words[cur]["phones"].append(hyp)
        words[cur]["diff"].append([ref, hyp, op])
        if op != "match":
            words[cur]["tags"].append(next(t))
    return words


def target_ipa_by_word(report: IpaReport, n_words: int) -> list:
    words = [[] for _ in range(n_words)]
    for p in report.target:
        words[p.word].append(p.sym)
    return ["".join(w) for w in words]
