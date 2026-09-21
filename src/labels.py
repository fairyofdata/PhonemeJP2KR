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


# --- per-error explanations (fixed templates; no LLM) ------------------------

_LARYNGEAL_KIND = {
    "ㄱ": "平音", "ㄷ": "平音", "ㅂ": "平音", "ㅈ": "平音", "ㅅ": "平音",
    "ㅋ": "激音", "ㅌ": "激音", "ㅍ": "激音", "ㅊ": "激音",
    "ㄲ": "濃音", "ㄸ": "濃音", "ㅃ": "濃音", "ㅉ": "濃音", "ㅆ": "濃音",
}
_LARYNGEAL_HOW = {
    "平音": "息を強く出さない音", "激音": "息を強く出す音", "濃音": "喉を締めて息を出さない音",
}
_CODA_PLACE = {"ㄱ": "舌の奥", "ㄷ": "舌先", "ㅂ": "両唇",
               "ㄴ": "舌先", "ㅁ": "両唇", "ㅇ": "舌の奥"}
_VOWEL_HOW = {
    "ㅓ": "唇を丸めずに口を縦に開ける [ʌ]", "ㅗ": "唇を丸める [o]",
    "ㅕ": "唇を丸めない [jʌ]", "ㅛ": "唇を丸める [jo]",
    "ㅡ": "唇を横に引く [ɯ]", "ㅜ": "唇を丸める [u]",
}


def explain_error(err: dict) -> str:
    """One Japanese sentence for one tag, built only from the tag and its jamo.

    States what the recognizer produced, never why the learner did it —
    the evidence is an ASR transcript, not an articulatory measurement.
    """
    tag, ref, hyp = err.get("tag", ""), err.get("ref", ""), err.get("hyp", "")
    if tag == "laryngeal_confusion":
        a, b = _LARYNGEAL_KIND.get(ref, ""), _LARYNGEAL_KIND.get(hyp, "")
        return (f"{ref}（{a}：{_LARYNGEAL_HOW.get(a, '')}）が {hyp}（{b}）として"
                "認識されました。日本語の清音・濁音の区別では捉えられない違いです。")
    if tag == "stop_coda_confusion":
        return (f"パッチム {ref}（{_CODA_PLACE[ref]}で閉じる）が {hyp}（{_CODA_PLACE[hyp]}で閉じる）"
                "として認識されました。促音「っ」のように次の音に合わせず、閉じる位置を保ちます。")
    if tag == "nasal_coda_confusion":
        return (f"パッチム {ref}（{_CODA_PLACE[ref]}で閉じる）が {hyp}（{_CODA_PLACE[hyp]}で閉じる）"
                "として認識されました。日本語の「ん」ではどちらも同じ音になります。")
    if tag in ("vowel_ʌ_o_confusion", "vowel_jʌ_jo_confusion", "vowel_ɯ_u_confusion"):
        return (f"{ref}（{_VOWEL_HOW[ref]}）が {hyp}（{_VOWEL_HOW[hyp]}）として認識されました。"
                "カタカナでは同じ文字になる組み合わせです。")
    if tag == "diphthong_ɰi_monophthongization":
        return f"語頭の ㅢ [ɰi] が {hyp} として認識されました。唇を横に引いたまま ㅣ へ移ります。"
    if tag == "vowel_epenthesis":
        return (f"本来ない母音 {hyp} が認識されました。パッチムの後に母音を足さず、"
                "口の形だけで音を止めます。")
    if tag == "coda_deletion":
        return f"パッチム {ref} が認識されませんでした。"
    if tag in ("consonant_deletion", "deletion"):
        return f"{ref} が認識されませんでした。"
    if tag == "insertion":
        return f"本来ない音 {hyp} が認識されました。"
    if tag == "substitution":
        return f"{ref} が {hyp} として認識されました。"
    return describe_error(err)
