"""Deterministic Korean grapheme-to-phoneme (G2P) conversion.

Implements the major phonological rules of Standard Korean (표준발음법)
as a pure-Python rule pipeline, then maps the resulting surface form to
IPA with basic allophony. Being rule-based, the output is fully
deterministic and reproducible — unlike LLM-generated transcriptions.

Pipeline order (applied to decomposed jamo syllables):
    0. Morphological Preprocessing (Kiwipiepy rules)
    1. ㅎ-cluster rules      (aspiration, ㅎ-deletion)        표준발음법 12항
    2. Palatalization        (굳이 → 구지, 같이 → 가치)        표준발음법 17항
    3. Liaison               (한국어 → 한구거)                 표준발음법 13-14항
    4. Coda neutralization   (7종성: 꽃 → 꼳)                  표준발음법 8-11항
    5. Post-obstruent tensification (학교 → 학꾜)              표준발음법 23항
    6. Nasal/liquid assimilation (합니다 → 함니다, 신라 → 실라) 표준발음법 18-20항

Now handles previously known limitations using morphological POS tagging:
    - ㄴ-insertion in compounds (꽃잎 → [꼰닙])
    - Context-dependent 겹받침 choice (밟다 → [밥따], 읽고 → [일꼬])
    - Vocative/semantic liaison exceptions (맛없다 → [마덥따])
    - Verb/Adjective stem tensification (신다 → [신따], 넓게 → [널께])
"""

try:
    from kiwipiepy import Kiwi
    _kiwi = Kiwi()
except ImportError:
    _kiwi = None

CHOSEONG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNGSEONG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONGSEONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ",
             "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ",
             "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

HANGUL_BASE = 0xAC00

# --- rule tables -----------------------------------------------------------

ASPIRATE = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ"}
TENSE = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}

# coda → (remaining coda, onset released on liaison)
CODA_SPLIT = {
    "ㄳ": ("ㄱ", "ㅆ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("ㄴ", ""),
    "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"), "ㄼ": ("ㄹ", "ㅂ"),
    "ㄽ": ("ㄹ", "ㅆ"), "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"),
    "ㅀ": ("ㄹ", ""), "ㅄ": ("ㅂ", "ㅆ"),
}

# 7-coda neutralization (평파열음화 + 자음군 단순화)
CODA_NEUTRAL = {
    "ㄱ": "ㄱ", "ㄲ": "ㄱ", "ㅋ": "ㄱ", "ㄳ": "ㄱ", "ㄺ": "ㄱ",
    "ㄴ": "ㄴ", "ㄵ": "ㄴ", "ㄶ": "ㄴ",
    "ㄷ": "ㄷ", "ㅅ": "ㄷ", "ㅆ": "ㄷ", "ㅈ": "ㄷ", "ㅊ": "ㄷ", "ㅌ": "ㄷ", "ㅎ": "ㄷ",
    "ㄹ": "ㄹ", "ㄼ": "ㄹ", "ㄽ": "ㄹ", "ㄾ": "ㄹ", "ㅀ": "ㄹ",
    "ㅁ": "ㅁ", "ㄻ": "ㅁ",
    "ㅂ": "ㅂ", "ㅍ": "ㅂ", "ㄿ": "ㅂ", "ㅄ": "ㅂ",
    "ㅇ": "ㅇ", "": "",
}

# codas whose obstruent element aspirates a following ㅎ (입학 → 이팍)
CODA_H_FUSION = {
    "ㄱ": ("", "ㅋ"), "ㄺ": ("ㄹ", "ㅋ"),
    "ㄷ": ("", "ㅌ"), "ㅅ": ("", "ㅌ"), "ㅆ": ("", "ㅌ"), "ㅊ": ("", "ㅌ"), "ㅌ": ("", "ㅌ"),
    "ㅈ": ("", "ㅊ"), "ㄵ": ("ㄴ", "ㅊ"),
    "ㅂ": ("", "ㅍ"), "ㄼ": ("ㄹ", "ㅍ"),
}

_IPA_ONSET = {
    "ㄱ": "k", "ㄲ": "k͈", "ㄴ": "n", "ㄷ": "t", "ㄸ": "t͈", "ㄹ": "ɾ",
    "ㅁ": "m", "ㅂ": "p", "ㅃ": "p͈", "ㅅ": "s", "ㅆ": "s͈", "ㅇ": "",
    "ㅈ": "tɕ", "ㅉ": "tɕ͈", "ㅊ": "tɕʰ", "ㅋ": "kʰ", "ㅌ": "tʰ",
    "ㅍ": "pʰ", "ㅎ": "h",
}
_IPA_VOICED = {"ㄱ": "ɡ", "ㄷ": "d", "ㅂ": "b", "ㅈ": "dʑ"}
_IPA_VOWEL = {
    "ㅏ": "a", "ㅐ": "ɛ", "ㅑ": "ja", "ㅒ": "jɛ", "ㅓ": "ʌ", "ㅔ": "e",
    "ㅕ": "jʌ", "ㅖ": "je", "ㅗ": "o", "ㅘ": "wa", "ㅙ": "wɛ", "ㅚ": "we",
    "ㅛ": "jo", "ㅜ": "u", "ㅝ": "wʌ", "ㅞ": "we", "ㅟ": "wi", "ㅠ": "ju",
    "ㅡ": "ɯ", "ㅢ": "ɰi", "ㅣ": "i",
}
_IPA_CODA = {"ㄱ": "k̚", "ㄴ": "n", "ㄷ": "t̚", "ㄹ": "ɭ", "ㅁ": "m",
             "ㅂ": "p̚", "ㅇ": "ŋ", "": ""}
_FRONT_GLIDE_VOWELS = {"ㅣ", "ㅑ", "ㅕ", "ㅛ", "ㅠ", "ㅖ", "ㅒ", "ㅟ"}
_SONORANT_CODAS = {"ㄴ", "ㄹ", "ㅁ", "ㅇ", ""}


def is_hangul_syllable(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7A3


def decompose(ch: str):
    """Hangul syllable → [choseong, jungseong, jongseong] (jongseong may be '')."""
    code = ord(ch) - HANGUL_BASE
    cho, rem = divmod(code, 21 * 28)
    jung, jong = divmod(rem, 28)
    return [CHOSEONG[cho], JUNGSEONG[jung], JONGSEONG[jong]]


def compose(cho: str, jung: str, jong: str = "") -> str:
    code = (HANGUL_BASE
            + CHOSEONG.index(cho) * 21 * 28
            + JUNGSEONG.index(jung) * 28
            + JONGSEONG.index(jong))
    return chr(code)


def _tokenize(text: str):
    """Split text into word-chunks of decomposed syllables; drop punctuation.

    Phonological rules never apply across a word boundary here, which is a
    simplification (real speech links across spaces) but keeps behaviour
    predictable for sentence-level scoring.
    """
    words, current = [], []
    for ch in text:
        if is_hangul_syllable(ch):
            current.append(decompose(ch))
        else:
            if current:
                words.append(current)
                current = []
    if current:
        words.append(current)
    return words


# --- rule steps (each operates on one word: list of [cho, jung, jong]) -----

def _apply_h_rules(syls):
    for i in range(len(syls) - 1):
        cur, nxt = syls[i], syls[i + 1]
        coda, onset = cur[2], nxt[0]
        # coda ㅎ (incl. ㄶ, ㅀ) + lenis onset → aspirated onset (좋다 → 조타)
        if coda in ("ㅎ", "ㄶ", "ㅀ"):
            base = {"ㅎ": "", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}[coda]
            if onset in ASPIRATE:
                cur[2], nxt[0] = base, ASPIRATE[onset]
            elif onset == "ㅅ":
                cur[2], nxt[0] = base, "ㅆ"
            elif onset == "ㅇ":  # ㅎ-deletion before vowel (좋아 → 조아)
                cur[2] = base
            elif onset == "ㄴ" and coda == "ㅎ":  # 놓는 → 논는 (via ㄷ)
                cur[2] = "ㄷ"
        # obstruent coda + ㅎ onset → fused aspirate (입학 → 이팍)
        elif onset == "ㅎ" and coda in CODA_H_FUSION:
            cur[2], nxt[0] = CODA_H_FUSION[coda]
    return syls


def _apply_palatalization(syls):
    for i in range(len(syls) - 1):
        cur, nxt = syls[i], syls[i + 1]
        if nxt[0] == "ㅇ" and nxt[1] == "ㅣ":
            if cur[2] == "ㄷ":
                cur[2], nxt[0] = "", "ㅈ"
            elif cur[2] == "ㅌ":
                cur[2], nxt[0] = "", "ㅊ"
            elif cur[2] == "ㄾ":
                cur[2], nxt[0] = "ㄹ", "ㅊ"
    return syls


def _apply_liaison(syls):
    for i in range(len(syls) - 1):
        cur, nxt = syls[i], syls[i + 1]
        coda = cur[2]
        if nxt[0] != "ㅇ" or coda in ("", "ㅇ"):
            continue
        if coda in CODA_SPLIT:
            remain, released = CODA_SPLIT[coda]
            cur[2] = remain
            if released:
                nxt[0] = released
        else:
            cur[2] = ""
            nxt[0] = coda
    return syls


def _neutralize_codas(syls):
    for syl in syls:
        syl[2] = CODA_NEUTRAL.get(syl[2], syl[2])
    return syls


def _apply_tensification(syls):
    for i in range(len(syls) - 1):
        cur, nxt = syls[i], syls[i + 1]
        if cur[2] in ("ㄱ", "ㄷ", "ㅂ") and nxt[0] in TENSE:
            nxt[0] = TENSE[nxt[0]]
    return syls


def _apply_assimilation(syls):
    """Nasalization and lateralization, iterated to a fixpoint."""
    changed = True
    while changed:
        changed = False
        for i in range(len(syls) - 1):
            cur, nxt = syls[i], syls[i + 1]
            coda, onset = cur[2], nxt[0]
            # lateralization: ㄴㄹ / ㄹㄴ → ㄹㄹ (신라 → 실라)
            if (coda, onset) in (("ㄴ", "ㄹ"), ("ㄹ", "ㄴ")):
                cur[2], nxt[0] = "ㄹ", "ㄹ"
                changed = True
            # onset ㄹ after non-ㄹ consonant → ㄴ (심리 → 심니, 독립 →…)
            elif onset == "ㄹ" and coda in ("ㅁ", "ㅇ", "ㄱ", "ㄷ", "ㅂ"):
                nxt[0] = "ㄴ"
                changed = True
            # obstruent coda + nasal onset → nasal coda (합니다 → 함니다)
            elif onset in ("ㄴ", "ㅁ") and coda in ("ㄱ", "ㄷ", "ㅂ"):
                cur[2] = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}[coda]
                changed = True
    return syls


def _word_to_surface(syls):
    syls = _apply_h_rules(syls)
    syls = _apply_palatalization(syls)
    syls = _apply_liaison(syls)
    syls = _neutralize_codas(syls)
    syls = _apply_tensification(syls)
    syls = _apply_assimilation(syls)
    return syls


def _preprocess_morphology(text: str) -> str:
    """Apply morphology-dependent phonological rules using Kiwipiepy."""
    if _kiwi is None:
        return text
        
    words = text.split(" ")
    out_words = []
    
    for word in words:
        if not word:
            out_words.append("")
            continue
            
        chars = list(word)
        
        # Patch for known compound roots that Kiwi lumps into single tokens
        for i in range(len(chars) - 1):
            if is_hangul_syllable(chars[i]) and decompose(chars[i])[2]:  # preceding char has coda
                if chars[i+1] == "잎": chars[i+1] = "닢"
                elif chars[i+1] == "약": chars[i+1] = "냑"
                
        tokens = _kiwi.tokenize("".join(chars), split_complex=True)
        
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            
            # Ensure tokens are adjacent in the original string
            if t1.start + t1.len != t2.start:
                continue
                
            c1_idx = t1.start + t1.len - 1
            c2_idx = t2.start
            
            char1 = chars[c1_idx]
            char2 = chars[c2_idx]
            
            if not (is_hangul_syllable(char1) and is_hangul_syllable(char2)):
                continue
                
            cho1, jung1, jong1 = decompose(char1)
            cho2, jung2, jong2 = decompose(char2)
            
            is_t1_verb = t1.tag.startswith('V')
            # For phonology, Suffixes (XS) often act as substantives (e.g., 식용유 -> 시굥뉴)
            # so we only consider Josa and Eomi as formal.
            is_t2_formal = t2.tag.startswith('J') or t2.tag.startswith('E')
            is_t2_eomi = t2.tag.startswith('E')
            
            # 1. ㄴ-insertion (ㄴ 첨가) - Rule 29
            # Preceding ends in consonant, following is substantive starting with 이,야,여,요,유
            if jong1 and not is_t2_formal and cho2 == "ㅇ" and jung2 in ("ㅣ", "ㅑ", "ㅕ", "ㅛ", "ㅠ", "ㅒ", "ㅖ"):
                chars[c2_idx] = compose("ㄴ", jung2, jong2)
                cho2 = "ㄴ"
                
            # 2. Verb/Adjective Stem Tensification (어간 경음화) - Rules 24, 25
            if is_t1_verb and is_t2_eomi and cho2 in TENSE:
                if jong1 in ("ㄴ", "ㄵ", "ㅁ", "ㄻ", "ㄼ", "ㄾ"):
                    chars[c2_idx] = compose(TENSE[cho2], jung2, jong2)
                elif jong1 == "ㄺ" and cho2 == "ㄱ":
                    chars[c1_idx] = compose(cho1, jung1, "ㄹ")
                    chars[c2_idx] = compose("ㄲ", jung2, jong2)
                    
            # 2-b. 밟다 / 넓다 special coda (Rule 10 exception)
            if is_t1_verb and jong1 == "ㄼ":
                if char1 == "밟":
                    chars[c1_idx] = compose(cho1, jung1, "ㅂ")
                    
            # 3. Meaningful Liaison Exception (의미 경계 연음 예외) - Rule 15
            # t1 ends in consonant, t2 is substantive starting with ㅏ,ㅓ,ㅗ,ㅜ,ㅟ
            if jong1 and not is_t2_formal and cho2 == "ㅇ" and jung2 in ("ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅟ"):
                neutral_coda = CODA_NEUTRAL.get(jong1, jong1)
                chars[c1_idx] = compose(cho1, jung1, neutral_coda)
                
        out_words.append("".join(chars))
        
    return " ".join(out_words)


# --- public API -------------------------------------------------------------

def to_surface(text: str) -> str:
    """Orthographic text → surface pronunciation in Hangul (감사합니다 → 감사함니다)."""
    text = _preprocess_morphology(text)
    words = [_word_to_surface(w) for w in _tokenize(text)]
    return " ".join("".join(compose(*s) for s in w) for w in words)


def to_jamo_sequence(text: str):
    """Orthographic text → flat list of surface-form jamo (scoring unit).

    Spaces and punctuation are excluded so the score is insensitive to
    tokenization differences between ASR outputs.
    """
    text = _preprocess_morphology(text)
    seq = []
    for word in (_word_to_surface(w) for w in _tokenize(text)):
        for cho, jung, jong in word:
            if cho != "ㅇ":
                seq.append(cho)
            seq.append(jung)
            if jong:
                seq.append(jong)
    return seq


def to_ipa(text: str) -> str:
    """Orthographic text → broad IPA with basic allophony.

    Allophony implemented: intervocalic lenis voicing (k→ɡ etc.) and
    ㅅ/ㅆ palatalization before front-glide vowels (s→ɕ).
    """
    text = _preprocess_morphology(text)
    words = [_word_to_surface(w) for w in _tokenize(text)]
    out_words = []
    for word in words:
        parts = []
        prev_voiced = False  # previous phone is a vowel or sonorant coda
        for cho, jung, jong in word:
            onset = _IPA_ONSET[cho]
            if prev_voiced and cho in _IPA_VOICED:
                onset = _IPA_VOICED[cho]
            if cho in ("ㅅ", "ㅆ") and jung in _FRONT_GLIDE_VOWELS:
                onset = "ɕ" if cho == "ㅅ" else "ɕ͈"
            parts.append(onset + _IPA_VOWEL[jung] + _IPA_CODA[jong])
            prev_voiced = jong in _SONORANT_CODAS
        out_words.append("".join(parts))
    return " ".join(out_words)
