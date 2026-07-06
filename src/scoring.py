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

from .g2p import to_jamo_sequence

# lenis / aspirated / tense triads share place & manner of articulation
_LARYNGEAL_SETS = [
    {"ㄱ", "ㄲ", "ㅋ"}, {"ㄷ", "ㄸ", "ㅌ"}, {"ㅂ", "ㅃ", "ㅍ"},
    {"ㅈ", "ㅉ", "ㅊ"}, {"ㅅ", "ㅆ"},
]
_VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_EPENTHETIC_VOWELS = {"ㅡ", "ㅜ", "ㅗ"}  # typical CV-repair vowels for JP speakers
_CODA_LIKE = {"ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅇ"}


@dataclass
class AlignedPair:
    op: str          # "match" | "sub" | "del" | "ins"
    ref: str         # target jamo ("" for insertions)
    hyp: str         # produced jamo ("" for deletions)


@dataclass
class ScoreReport:
    score: int                       # 0–100, round(100 * (1 - PER))
    distance: int
    ref_len: int
    pairs: list = field(default_factory=list)   # list[AlignedPair]
    error_tags: list = field(default_factory=list)


def align_jamo(ref, hyp):
    """Levenshtein alignment with backtrace → list[AlignedPair]."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,        # deletion
                           dp[i][j - 1] + 1,        # insertion
                           dp[i - 1][j - 1] + cost)  # match/sub

    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            op = "match" if ref[i - 1] == hyp[j - 1] else "sub"
            pairs.append(AlignedPair(op, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append(AlignedPair("del", ref[i - 1], ""))
            i -= 1
        else:
            pairs.append(AlignedPair("ins", "", hyp[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs, dp[n][m]


def _same_laryngeal_family(a, b):
    return any(a in s and b in s for s in _LARYNGEAL_SETS)


def classify_errors(pairs):
    """Tag alignment errors with known Japanese-L1 interference patterns.

    Returns a list of dicts: {"tag": ..., "ref": ..., "hyp": ...} suitable
    for direct serialization into the LLM prompt.
    """
    tags = []
    for idx, p in enumerate(pairs):
        if p.op == "match":
            continue
        if p.op == "ins" and p.hyp in _EPENTHETIC_VOWELS:
            # e.g. 밥 → 바브: mora-timed L1 repairs closed syllables with a vowel
            tags.append({"tag": "vowel_epenthesis", "ref": "", "hyp": p.hyp})
        elif p.op == "del" and p.ref in _CODA_LIKE and p.ref not in _VOWELS:
            prev_is_vowel = idx > 0 and pairs[idx - 1].ref in _VOWELS
            tag = "coda_deletion" if prev_is_vowel else "consonant_deletion"
            tags.append({"tag": tag, "ref": p.ref, "hyp": ""})
        elif p.op == "sub" and _same_laryngeal_family(p.ref, p.hyp):
            # 평음/경음/격음 confusion — JP has only a voiced/voiceless contrast
            tags.append({"tag": "laryngeal_confusion", "ref": p.ref, "hyp": p.hyp})
        elif p.op == "sub" and {p.ref, p.hyp} <= {"ㅓ", "ㅗ"}:
            tags.append({"tag": "vowel_ʌ_o_confusion", "ref": p.ref, "hyp": p.hyp})
        elif p.op == "sub" and {p.ref, p.hyp} <= {"ㅡ", "ㅜ"}:
            tags.append({"tag": "vowel_ɯ_u_confusion", "ref": p.ref, "hyp": p.hyp})
        elif p.op == "sub" and {p.ref, p.hyp} == {"ㄴ", "ㅇ"}:
            tags.append({"tag": "nasal_coda_confusion", "ref": p.ref, "hyp": p.hyp})
        elif p.op == "sub":
            tags.append({"tag": "substitution", "ref": p.ref, "hyp": p.hyp})
        elif p.op == "ins":
            tags.append({"tag": "insertion", "ref": "", "hyp": p.hyp})
        else:
            tags.append({"tag": "deletion", "ref": p.ref, "hyp": ""})
    return tags


def score_pronunciation(target_text: str, actual_text: str) -> ScoreReport:
    """Compare target vs ASR hypothesis at the jamo level after G2P."""
    ref = to_jamo_sequence(target_text)
    hyp = to_jamo_sequence(actual_text)
    if not ref:
        return ScoreReport(score=0, distance=0, ref_len=0)
    pairs, distance = align_jamo(ref, hyp)
    per = distance / max(len(ref), len(hyp))
    score = max(0, round(100 * (1 - per)))
    return ScoreReport(
        score=score,
        distance=distance,
        ref_len=len(ref),
        pairs=pairs,
        error_tags=classify_errors(pairs),
    )


def render_diff_markdown(pairs) -> str:
    """Two-row jamo diff for the UI: target row vs produced row."""
    ref_row, hyp_row = [], []
    for p in pairs:
        ref_cell = p.ref or "·"
        hyp_cell = p.hyp or "·"
        if p.op != "match":
            hyp_cell = f"**{hyp_cell}**"
        ref_row.append(ref_cell)
        hyp_row.append(hyp_cell)
    header = "| " + " | ".join(ref_row) + " |"
    sep = "|" + "---|" * len(ref_row)
    body = "| " + " | ".join(hyp_row) + " |"
    return "\n".join([header, sep, body])
