"""Phoneme-level scoring via jamo alignment.

Both the target sentence and the ASR hypothesis are converted to their
surface pronunciation with the deterministic G2P (src/g2p.py), decomposed
into jamo, and aligned with Levenshtein dynamic programming. The score is
1 − PER (phoneme error rate), and the alignment trace feeds two consumers:

    1. the UI (per-phoneme diff highlighting), and
    2. a rule-based classifier that tags known Japanese-L1 interference
       patterns (vowel epenthesis, coda drop, laryngeal confusion, …),
       giving the LLM structured evidence instead of raw strings.
"""

from dataclasses import dataclass, field

from .g2p import char_times, ipa_segments, jamo_positions, linkable_boundaries, word_gaps

# lenis / aspirated / tense triads share place & manner of articulation
_LARYNGEAL_SETS = [
    {"ㄱ", "ㄲ", "ㅋ"}, {"ㄷ", "ㄸ", "ㅌ"}, {"ㅂ", "ㅃ", "ㅍ"},
    {"ㅈ", "ㅉ", "ㅊ"}, {"ㅅ", "ㅆ"},
]
_VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_EPENTHETIC_VOWELS = {"ㅡ", "ㅜ", "ㅗ"}  # typical CV-repair vowels for JP speakers
_CODA_LIKE = {"ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅇ"}
_NASAL_CODAS = {"ㄴ", "ㅁ", "ㅇ"}   # JP 撥音 ん has no stable place of its own
_STOP_CODAS = {"ㄱ", "ㄷ", "ㅂ"}    # unreleased [k̚ t̚ p̚]; JP 促音 っ copies place


@dataclass
class AlignedPair:
    op: str          # "match" | "sub" | "del" | "ins"
    ref: str         # target jamo ("" for insertions)
    hyp: str         # produced jamo ("" for deletions)
    start_time: float = None
    end_time: float = None
    ref_pos: int = None  # index of the source character in the target text
    hyp_pos: int = None  # index of the source character in the hypothesis
    ref_ipa: str = ""    # IPA of the target jamo in context ([ʌ], [k͈], …)
    hyp_ipa: str = ""


@dataclass
class ScoreReport:
    score: int                       # 0–100, round(100 * (1 - PER))
    distance: int
    ref_len: int
    pairs: list = field(default_factory=list)   # list[AlignedPair]
    error_tags: list = field(default_factory=list)
    linked: list = field(default_factory=list)  # boundaries scored as one phrase


def align_jamo(ref, hyp, ref_pos=None, hyp_pos=None):
    """Levenshtein alignment with backtrace → list[AlignedPair].

    ``ref_pos``/``hyp_pos`` optionally give each jamo's source-character
    index; they ride along on the pairs and never affect the alignment.
    """
    ref_pos = ref_pos or [None] * len(ref)
    if hyp and isinstance(hyp[0], tuple):
        hyp_chars = [h[0] for h in hyp]
        hyp_times = [(h[1], h[2]) for h in hyp]
    else:
        hyp_chars = hyp
        hyp_times = [(None, None) for _ in hyp]

    hyp_pos = hyp_pos or [None] * len(hyp_chars)
    n, m = len(ref), len(hyp_chars)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp_chars[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,        # deletion
                           dp[i][j - 1] + 1,        # insertion
                           dp[i - 1][j - 1] + cost)  # match/sub

    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp_chars[j - 1] else 1):
            op = "match" if ref[i - 1] == hyp_chars[j - 1] else "sub"
            start_time, end_time = hyp_times[j - 1]
            pairs.append(AlignedPair(op, ref[i - 1], hyp_chars[j - 1], start_time, end_time,
                                     ref_pos[i - 1], hyp_pos[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append(AlignedPair("del", ref[i - 1], "", None, None, ref_pos[i - 1]))
            i -= 1
        else:
            start_time, end_time = hyp_times[j - 1]
            pairs.append(AlignedPair("ins", "", hyp_chars[j - 1], start_time, end_time,
                                     None, hyp_pos[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs, dp[n][m]


def _same_laryngeal_family(a, b):
    return any(a in s and b in s for s in _LARYNGEAL_SETS)


def _is_coda(pairs, idx):
    """True if the target jamo at pairs[idx] is a syllable coda.

    Onset ㅇ is dropped from the jamo sequence, so on the target side a
    consonant is a coda exactly when it follows a vowel and is not
    followed by one (V C C… or V C at the end). Insertions carry no
    target jamo and are skipped when looking at neighbours.

    Word boundaries are not in the sequence, so a coda before a
    vowel-initial next word (산 아래) reads as an onset; that only costs a
    missed coda tag (it falls back to "substitution"), never a false one.
    """
    refs = [p.ref for p in pairs]
    prev = next((r for r in reversed(refs[:idx]) if r), "")
    nxt = next((r for r in refs[idx + 1:] if r), "")
    return prev in _VOWELS and nxt not in _VOWELS


def classify_pair(pairs, idx):
    """Tag one non-matching pair; None for a match.

    Returns {"tag", "ref", "hyp"[, "timestamp"]}, suitable for direct
    serialization into the LLM prompt.
    """
    p = pairs[idx]
    if p.op == "match":
        return None
    tag_dict = {"ref": p.ref, "hyp": p.hyp}
    if p.ref_ipa or p.hyp_ipa:
        tag_dict["ref_ipa"], tag_dict["hyp_ipa"] = p.ref_ipa, p.hyp_ipa
    if p.start_time is not None:
        tag_dict["timestamp"] = round(p.start_time, 2)

    if p.op == "ins" and p.hyp in _EPENTHETIC_VOWELS:
        tag_dict["tag"] = "vowel_epenthesis"
    elif p.op == "del" and p.ref in _CODA_LIKE and p.ref not in _VOWELS:
        prev_is_vowel = idx > 0 and pairs[idx - 1].ref in _VOWELS
        tag_dict["tag"] = "coda_deletion" if prev_is_vowel else "consonant_deletion"
    elif p.op == "sub" and _same_laryngeal_family(p.ref, p.hyp):
        tag_dict["tag"] = "laryngeal_confusion"
    elif p.op == "sub" and {p.ref, p.hyp} <= {"ㅓ", "ㅗ"}:
        tag_dict["tag"] = "vowel_ʌ_o_confusion"
    elif p.op == "sub" and {p.ref, p.hyp} <= {"ㅕ", "ㅛ"}:
        tag_dict["tag"] = "vowel_jʌ_jo_confusion"
    elif p.op == "sub" and {p.ref, p.hyp} <= {"ㅡ", "ㅜ"}:
        tag_dict["tag"] = "vowel_ɯ_u_confusion"
    elif p.op == "sub" and p.ref == "ㅢ":
        tag_dict["tag"] = "diphthong_ɰi_monophthongization"
    elif (p.op == "sub" and {p.ref, p.hyp} <= _NASAL_CODAS
          and _is_coda(pairs, idx)):
        tag_dict["tag"] = "nasal_coda_confusion"
    elif (p.op == "sub" and {p.ref, p.hyp} <= _STOP_CODAS
          and _is_coda(pairs, idx)):
        tag_dict["tag"] = "stop_coda_confusion"
    elif p.op == "sub":
        tag_dict["tag"] = "substitution"
    elif p.op == "ins":
        tag_dict["tag"] = "insertion"
    else:
        tag_dict["tag"] = "deletion"
    return tag_dict


def classify_errors(pairs):
    """Tag alignment errors with known Japanese-L1 interference patterns."""
    return [t for t in (classify_pair(pairs, i) for i in range(len(pairs))) if t]


def _attach_ipa(pairs, ref_ipa, hyp_ipa):
    """Give each pair the in-context IPA of its two jamo (display/evidence)."""
    r = h = 0
    for p in pairs:
        if p.op != "ins":
            p.ref_ipa, r = ref_ipa[r], r + 1
        if p.op != "del":
            p.hyp_ipa, h = hyp_ipa[h], h + 1


def _boundary_window(target_text: str, boundary: int) -> set:
    """The two characters a linked boundary can change: the last of the
    first word and the first of the second."""
    space = next(i for i, b in word_gaps(target_text).items() if b == boundary)
    return {space - 1, space + 1}


def _boundary_errors(target_text: str, actual_text: str, linked, boundary: int) -> int:
    """Non-matching pairs at one boundary, with the given reading of all of them."""
    ref_seq = jamo_positions(target_text, linked)
    hyp_seq = jamo_positions(actual_text)
    pairs, _ = align_jamo([j for j, _ in ref_seq], [j for j, _ in hyp_seq],
                          [i for _, i in ref_seq], [i for _, i in hyp_seq])
    window = _boundary_window(target_text, boundary)
    in_window = [p.ref_pos in window for p in pairs]
    errors = 0
    for k, p in enumerate(pairs):
        if p.op == "match":
            continue
        if p.op == "ins":       # insertions have no target position of their own
            before = next((in_window[i] for i in range(k - 1, -1, -1) if pairs[i].op != "ins"), False)
            after = next((in_window[i] for i in range(k + 1, len(pairs)) if pairs[i].op != "ins"), False)
            errors += before or after
        else:
            errors += in_window[k]
    return errors


def choose_linked_boundaries(target_text: str, actual_text: str) -> list:
    """Which word boundaries the speaker read as one phrase.

    Where pausing or linking changes the standard pronunciation, both
    readings are correct (표준발음법 §15, §18 붙임, §29 붙임2), so each
    boundary is judged on its own: the reading with fewer errors *at that
    boundary* wins, and a tie keeps the pausing (default) reading. The
    comparison is deliberately local — picking whichever whole-sentence
    reading scores higher would let errors elsewhere decide it.
    """
    chosen = []
    for b in linkable_boundaries(target_text):
        if (_boundary_errors(target_text, actual_text, chosen + [b], b)
                < _boundary_errors(target_text, actual_text, chosen, b)):
            chosen.append(b)
    return chosen


def score_pronunciation(target_text: str, actual_text: str, char_timestamps=None,
                        linked=None) -> ScoreReport:
    """Compare target vs ASR hypothesis at the jamo level after G2P.

    ``linked`` are the word boundaries to read as one phrase; by default
    they are chosen per boundary (see choose_linked_boundaries).
    """
    if linked is None:
        linked = choose_linked_boundaries(target_text, actual_text)
    ref_seq = jamo_positions(target_text, linked)
    hyp_seq = jamo_positions(actual_text)
    if not ref_seq:
        return ScoreReport(score=0, distance=0, ref_len=0)
    ref = [j for j, _ in ref_seq]
    if char_timestamps is None:
        hyp = [j for j, _ in hyp_seq]
    else:
        times = char_times(actual_text, char_timestamps)
        hyp = [(j, *times.get(i, (None, None))) for j, i in hyp_seq]
    pairs, distance = align_jamo(ref, hyp, [i for _, i in ref_seq], [i for _, i in hyp_seq])
    _attach_ipa(pairs, ipa_segments(ref_seq), ipa_segments(hyp_seq))
    per = distance / max(len(ref), len(hyp))
    score = max(0, round(100 * (1 - per)))
    return ScoreReport(
        score=score,
        distance=distance,
        ref_len=len(ref),
        pairs=pairs,
        error_tags=classify_errors(pairs),
        linked=list(linked),
    )
