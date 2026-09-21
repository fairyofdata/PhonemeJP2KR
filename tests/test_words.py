"""Word-level alignment and rule-based katakana (display/export layer)."""

from src.kana import to_kana
from src.labels import TAG_LABELS, explain_error
from src.scoring import score_pronunciation
from src.words import word_view

# the README demo take (sentence 8, record 6 in the author's history)
TARGET = "화려한 도시를 그리며 찾아왔네 그 곳은 춥고도 험한 곳"
WHISPER = "하려한 도시르 그리며 자두자 왔네 그곳은 주부고도 홈한 곳"
WAV2VEC = "하류한도시로그리임용들증하는데그고은죽고도홍한고"


def _by_word():
    return {w["target"]: w for w in word_view(TARGET, WHISPER, WAV2VEC)}


def test_unspaced_acoustic_output_is_cut_into_target_words():
    words = word_view(TARGET, WHISPER, WAV2VEC)
    assert [w["target"] for w in words] == TARGET.split()
    assert "".join(w["acoustic"] for w in words) == WAV2VEC
    assert [w["acoustic"] for w in words][-3:] == ["죽고도", "홍한", "고"]


def test_whisper_segments_keep_their_own_spacing():
    w = _by_word()
    assert w["찾아왔네"]["heard"] == "자두자 왔네"
    assert w["그"]["heard"] == "그" and w["곳은"]["heard"] == "곳은"   # 그곳은 split back


def test_word_tags_are_exactly_the_scored_tags():
    words = word_view(TARGET, WHISPER, WAV2VEC)
    per_word = [(t["tag"], t["ref"], t["hyp"]) for w in words for t in w["acoustic_errors"]]
    scored = [(t["tag"], t["ref"], t["hyp"])
              for t in score_pronunciation(TARGET, WAV2VEC).error_tags]
    assert per_word == scored


def test_demo_words_carry_the_expected_l1_tags():
    w = _by_word()
    assert [t["tag"] for t in w["춥고도"]["acoustic_errors"]] == [
        "laryngeal_confusion", "stop_coda_confusion"]
    assert [t["tag"] for t in w["험한"]["acoustic_errors"]] == [
        "vowel_ʌ_o_confusion", "nasal_coda_confusion"]
    # Whisper heard an epenthetic vowel the acoustic channel does not show
    assert "vowel_epenthesis" in [t["tag"] for t in w["춥고도"]["heard_errors"]]
    # tensification is invisible: 죽고도 passes the same G2P → [죽꼬도]
    assert not any(t["ref"] == "ㄲ" for t in w["춥고도"]["acoustic_errors"])
    assert w["춥고도"]["surface"] == "춥꼬도"


def test_a_syllable_straddling_a_boundary_goes_to_its_vowel_word():
    w = _by_word()
    assert w["그리며"]["acoustic"] == "그리임용"
    assert w["찾아왔네"]["acoustic"] == "들증하는데"


def test_word_katakana_concatenates_to_the_sentence_katakana():
    words = word_view(TARGET, WHISPER, WAV2VEC)
    assert "".join(w["katakana"] for w in words) == to_kana(WAV2VEC)


def test_katakana_merges_the_contrasts_the_system_measures():
    assert to_kana("거") == to_kana("고")                          # ㅓ / ㅗ
    assert to_kana("그") == to_kana("구")                          # ㅡ / ㅜ
    assert to_kana("달") == to_kana("탈") == to_kana("딸")          # lenis / aspirated / tense
    assert to_kana("산") == to_kana("상")                          # ㄴ / ㅇ coda


def test_katakana_voices_lenis_onsets_between_sonorants():
    assert to_kana("감사합니다") == "カムサハムニダ"
    assert to_kana("도시") == "トシ"
    assert to_kana("그리고") == "クリゴ"
    assert to_kana("화려한 도시") == "ファリョハン トシ"   # a space resets voicing


def test_every_tag_has_a_japanese_explanation():
    for tag in TAG_LABELS:
        err = {"tag": tag, "ref": "ㅂ", "hyp": "ㄱ"}
        if tag.startswith("vowel_") and tag != "vowel_epenthesis":
            err = {"tag": tag, "ref": "ㅓ", "hyp": "ㅗ"}
            if "jʌ" in tag:
                err = {"tag": tag, "ref": "ㅕ", "hyp": "ㅛ"}
            elif "ɯ" in tag:
                err = {"tag": tag, "ref": "ㅡ", "hyp": "ㅜ"}
        elif tag == "nasal_coda_confusion":
            err = {"tag": tag, "ref": "ㅁ", "hyp": "ㅇ"}
        text = explain_error(err)
        assert text and tag not in text


def test_empty_hypotheses_do_not_crash():
    words = word_view("안녕하세요", "", "")
    assert words[0]["acoustic"] == "" and words[0]["katakana"] == ""


# --- CTC forced alignment (admin text input) ---------------------------------

from src.ctc import forced_align  # noqa: E402

NEG = -30.0


def _frames(seq, blank=0, n_tokens=4):
    """One-hot-ish log-probs: frame t strongly predicts seq[t]."""
    return [[0.0 if k == tok else NEG for k in range(n_tokens)] for tok in seq]


def test_forced_align_finds_each_token_span():
    # frames: blank a a blank b blank blank c
    lp = _frames([0, 1, 1, 0, 2, 0, 0, 3])
    assert forced_align(lp, [1, 2, 3], blank=0) == [(1, 3), (4, 5), (7, 8)]


def test_forced_align_separates_repeated_tokens_with_a_blank():
    lp = _frames([1, 0, 1])
    assert forced_align(lp, [1, 1], blank=0) == [(0, 1), (2, 3)]


def test_forced_align_rejects_text_longer_than_audio():
    import pytest
    with pytest.raises(ValueError):
        forced_align(_frames([1, 2]), [1, 2, 3], blank=0)
