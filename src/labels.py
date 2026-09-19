"""Learner-facing (Japanese) names for the L1 error tags in src/scoring.py.

Tag ids stay stable for the LLM prompt, the database and the experiments;
only the UI shows these names.
"""

TAG_LABELS = {
    "vowel_epenthesis": "母音の挿入",
    "coda_deletion": "パッチムの脱落",
    "consonant_deletion": "子音の脱落",
    "laryngeal_confusion": "平音・激音・濃音の混同",
    "vowel_ʌ_o_confusion": "ㅓ と ㅗ の混同",
    "vowel_jʌ_jo_confusion": "ㅕ と ㅛ の混同",
    "vowel_ɯ_u_confusion": "ㅡ と ㅜ の混同",
    "diphthong_ɰi_monophthongization": "ㅢ の単母音化",
    "nasal_coda_confusion": "鼻音パッチム ㄴ・ㅁ・ㅇ の混同",
    "stop_coda_confusion": "閉鎖音パッチム ㄱ・ㄷ・ㅂ の混同",
    "substitution": "別の音への置き換え",
    "insertion": "余分な音",
    "deletion": "音の脱落",
}


def tag_label(tag: str) -> str:
    return TAG_LABELS.get(tag, tag)


def describe_error(err: dict) -> str:
    """'ㅓ と ㅗ の混同 · ㅓ→ㅗ' — label plus the concrete jamo evidence."""
    ref, hyp = err.get("ref", ""), err.get("hyp", "")
    if ref and hyp:
        evidence = f"{ref}→{hyp}"
    elif ref:
        evidence = f"{ref} が脱落"
    elif hyp:
        evidence = f"{hyp} を挿入"
    else:
        evidence = ""
    label = tag_label(err.get("tag", ""))
    return f"{label} · {evidence}" if evidence else label
