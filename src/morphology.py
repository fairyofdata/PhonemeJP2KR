"""Morphology-conditioned pronunciation adjustments via Kiwipiepy.

The core G2P pipeline (src/g2p.py) is deliberately context-free: it
cannot see morpheme boundaries. Rules that depend on them are applied
here first, by rewriting the orthography into a *pronunciation spelling*
(발음 표기) — e.g. 꽃잎 → 꽃닢 — which the phonological pipeline then
derives normally (꽃닢 → 꼳닢 → [꼰닙]).

Boundary detection is single-sourced from Kiwipiepy's POS analysis.
Compounds that Kiwi lumps into one token are taught their internal
boundary through ``add_pre_analyzed_word`` (declarative lexicon below) —
never by pattern-matching characters in running text.

Rules implemented (표준발음법):
    §29        ㄴ-insertion at compound/derivation boundaries  (꽃잎 → 꼰닙)
    §24–25     verb/adjective stem-final tensification          (신다 → 신따)
    §10–11 단서 stem coda overrides                             (밟다 → 밥따, 읽고 → 일꼬)
    §15        liaison blocking before lexical morphemes        (맛없다 → 마덥따)
    §15 다만    lexical liaison exceptions                       (맛있다 → 마싣따)
    §20 다만    ㄴ+ㄹ → [ㄴㄴ] in Sino-Korean derivation          (의견란 → 의견난)

Determinism: Kiwi's analysis is deterministic for a fixed model version,
so kiwipiepy is pinned in requirements.txt. Without kiwipiepy installed,
src/g2p.py falls back to the context-free pipeline (documented
degradation; morphology-dependent words revert to their pre-morphology
outputs).
"""

from kiwipiepy import Kiwi

from .g2p import CODA_NEUTRAL, TENSE, compose, decompose, is_hangul_syllable

# --- declarative lexicons ---------------------------------------------------

# Compounds Kiwi analyzes as a single token; teach it the internal boundary
# so the ㄴ-insertion rule below can see it. Parts are (lexeme, tag) with
# optional explicit surface span (start, end) for 사잇소리 fusions where the
# lexeme differs from its surface spelling (나뭇잎 = 나무 over span 0-2).
_COMPOUND_BOUNDARIES = {
    "꽃잎": [("꽃", "NNG"), ("잎", "NNG")],
    "나뭇잎": [("나무", "NNG", 0, 2), ("잎", "NNG", 2, 3)],
    "깻잎": [("깨", "NNG", 0, 1), ("잎", "NNG", 1, 2)],
    "풀잎": [("풀", "NNG"), ("잎", "NNG")],
    "콩잎": [("콩", "NNG"), ("잎", "NNG")],
    "물약": [("물", "NNG"), ("약", "NNG")],
    "알약": [("알", "NNG"), ("약", "NNG")],
    "식용유": [("식용", "NNG"), ("유", "NNG")],
    "색연필": [("색", "NNG"), ("연필", "NNG")],
    "솜이불": [("솜", "NNG"), ("이불", "NNG")],
    "담요": [("담", "NNG"), ("요", "NNG")],
    "서울역": [("서울", "NNP"), ("역", "NNG")],
    "값어치": [("값", "NNG"), ("어치", "XSN")],
}

# §15 다만: 맛있다/멋있다 may keep their coda's original value in liaison
_LIAISON_EXCEPTIONS = {("맛", "있"), ("멋", "있")}

# §10 단서: 밟- keeps [ㅂ] before a consonant-initial ending
_STEM_CODA_OVERRIDES = {"밟": "ㅂ"}

# §24 (ㄴ, ㄵ, ㅁ, ㄻ) and §25 (ㄼ, ㄾ): stem-final codas that tensify endings
_STEM_TENSING_CODAS = {"ㄴ", "ㄵ", "ㅁ", "ㄻ", "ㄼ", "ㄾ"}

_N_INSERTION_VOWELS = {"ㅣ", "ㅑ", "ㅕ", "ㅛ", "ㅠ", "ㅒ", "ㅖ"}


def _is_bound(tag: str) -> bool:
    """Phonologically bound forms — liaison applies as within one word.

    Josa (J*), endings (E*), the copula 이- (VCP), and auxiliary verbs
    (VX) do not constitute the kind of lexical boundary that triggers
    ㄴ-insertion or liaison blocking. Treating VCP as lexical is exactly
    the bug that turned 학생입니다 into *[학쌩님니다].
    """
    return tag[0] in "JE" or tag in ("VCP", "VX")


def _hosts_n_insertion(tag: str) -> bool:
    """Morphemes that can host §29 ㄴ-insertion: nominals and nominal affixes."""
    return tag[0] == "N" or tag in ("XR", "XSN", "XPN", "MM")


def _is_lexical(tag: str) -> bool:
    """실질형태소 for §15 liaison blocking (nouns, roots, lexical verb stems,
    and vowel-initial nominal suffixes like 어치 — 값어치 → [가버치])."""
    return tag[0] in "NV" and not _is_bound(tag) or tag in ("XR", "MAG", "XSN")


_kiwi = None


def _get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        kiwi = Kiwi()
        for form, parts in _COMPOUND_BOUNDARIES.items():
            analyzed, pos = [], 0
            for part in parts:
                if len(part) == 4:
                    analyzed.append(part)
                    pos = part[3]
                else:
                    p_form, p_tag = part
                    analyzed.append((p_form, p_tag, pos, pos + len(p_form)))
                    pos += len(p_form)
            kiwi.add_pre_analyzed_word(form, analyzed, score=100.0)
        _kiwi = kiwi
    return _kiwi


def _apply_boundary_rules(chars, t1, t2):
    """Rewrite the two syllables flanking one morpheme boundary in place."""
    i1, i2 = t2.start - 1, t2.start
    c1, c2 = chars[i1], chars[i2]
    if not (is_hangul_syllable(c1) and is_hangul_syllable(c2)):
        return
    cho1, jung1, jong1 = decompose(c1)
    cho2, jung2, jong2 = decompose(c2)

    if (t1.form, t2.form) in _LIAISON_EXCEPTIONS:
        return

    t1_is_stem = t1.tag[0] == "V" and t1.tag not in ("VCP",)
    t2_is_ending = t2.tag[0] == "E"

    # §10 단서: 밟- + consonant → coda ㅂ (밟다 → 밥따)
    if t1_is_stem and c1 in _STEM_CODA_OVERRIDES and cho2 != "ㅇ":
        jong1 = _STEM_CODA_OVERRIDES[c1]
        chars[i1] = compose(cho1, jung1, jong1)

    # §11 단서: stem ㄺ + ending ㄱ → ㄹ + ㄲ (읽고 → 일꼬)
    if t1_is_stem and t2_is_ending and jong1 == "ㄺ" and cho2 == "ㄱ":
        chars[i1] = compose(cho1, jung1, "ㄹ")
        chars[i2] = compose("ㄲ", jung2, jong2)
        return

    # §24–25: stem-final sonorant(+cluster) coda tensifies the ending onset
    if (t1_is_stem and t2_is_ending
            and jong1 in _STEM_TENSING_CODAS and cho2 in TENSE):
        chars[i2] = compose(TENSE[cho2], jung2, jong2)
        return

    if _is_bound(t2.tag):
        return

    # §29: ㄴ-insertion (꽃|잎 → 꽃닢, 한|여름 → 한녀름)
    if jong1 and _hosts_n_insertion(t2.tag) and cho2 == "ㅇ" and jung2 in _N_INSERTION_VOWELS:
        chars[i2] = compose("ㄴ", jung2, jong2)
        return

    # §20 다만: ㄴ + ㄹ-initial Sino-Korean suffix syllable → [ㄴㄴ] (의견란 → 의견난)
    if jong1 == "ㄴ" and cho2 == "ㄹ" and len(t2.form) == 1 and _hosts_n_insertion(t2.tag):
        chars[i2] = compose("ㄴ", jung2, jong2)
        return

    # §15: coda neutralizes before a vowel-initial lexical morpheme
    # (맛없다 → 맏없다 → [마덥따]) instead of plain liaison
    if jong1 and _is_lexical(t2.tag) and cho2 == "ㅇ":
        chars[i1] = compose(cho1, jung1, CODA_NEUTRAL.get(jong1, jong1))


def apply(text: str) -> str:
    """Orthography → pronunciation spelling (length/offsets preserved).

    The whole text is analyzed in one pass so Kiwi can use sentence
    context for POS disambiguation (신고/NNG vs 신-고/VV+EC). Rules only
    fire between strictly adjacent tokens, so whitespace and punctuation
    still act as boundaries.
    """
    if not any(is_hangul_syllable(ch) for ch in text):
        return text
    chars = list(text)
    tokens = _get_kiwi().tokenize(text, split_complex=True)
    for t1, t2 in zip(tokens, tokens[1:]):
        if t1.start + t1.len == t2.start and t2.start > 0:
            _apply_boundary_rules(chars, t1, t2)
    return "".join(chars)
