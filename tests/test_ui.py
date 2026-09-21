"""UI markup helpers: syllable grouping, escaping, audio sniffing."""

from src.scoring import score_pronunciation
from src.ui import (
    audio_mime, channel_rows_html, diff_html, error_list_html, score_hero_html,
    syllable_groups, target_html, word_detail_html, word_grid_html,
)


def _syllables(target, hyp):
    pairs = score_pronunciation(target, hyp).pairs
    return ["".join(p.ref for p in g) for g in syllable_groups(pairs)]


def test_syllable_groups_follow_target_syllables():
    # surface 서우레서: liaison moves ㄹ into the next syllable's onset
    assert _syllables("서울에서", "서울에서") == ["ㅅㅓ", "ㅜ", "ㄹㅔ", "ㅅㅓ"]
    assert _syllables("밥", "밥") == ["ㅂㅏㅂ"]
    assert _syllables("학교", "학교") == ["ㅎㅏㄱ", "ㄲㅛ"]


def test_syllable_groups_keep_insertions_with_preceding_syllable():
    groups = syllable_groups(score_pronunciation("밥", "바브").pairs)
    assert len(groups) == 1
    assert [p.op for p in groups[0]].count("ins") == 1


def test_diff_marks_only_syllables_with_errors():
    html = diff_html(score_pronunciation("서울", "소울").pairs)
    assert html.count('class="pc-syl bad"') == 1
    assert html.count('class="pc-syl"') == 1


def test_user_and_model_text_is_escaped():
    evil = "<script>alert(1)</script>"
    assert "<script>" not in target_html(evil, evil, evil)
    res = {"target": evil, "target_surface": evil, "whisper_text": evil, "whisper_ipa": "",
           "wav2vec_text": evil, "actual_ipa": ""}
    word = {"index": 0, "target": evil, "surface": evil, "heard": evil, "acoustic": evil,
            "katakana": evil, "acoustic_errors": [{"tag": evil, "ref": evil, "hyp": evil}],
            "heard_errors": []}
    assert "<script>" not in channel_rows_html(res, [word])
    assert "<script>" not in word_grid_html([word])
    assert "<script>" not in word_detail_html(word)
    assert "<script>" not in error_list_html([{"tag": evil, "ref": evil, "hyp": ""}])


def test_score_hero_shows_band_range_and_delta():
    ref = {"p10": 68, "median": 81, "n_faithful": 483}
    band = {"band": "indeterminate", "low": 68, "high": 80}
    html = score_hero_html(76, band, "判定保留ゾーン", "…", ref, previous=70)
    assert "68–80" in html and ">76<" in html and "+6" in html
    assert "前回" not in score_hero_html(76, band, "x", "…", ref)


def test_audio_mime_sniffing():
    assert audio_mime(b"RIFF\x00\x00\x00\x00WAVE") == "audio/wav"
    assert audio_mime(b"ID3\x04") == "audio/mpeg"
    assert audio_mime(b"fLaC\x00") == "audio/flac"
    assert audio_mime(b"\x1aE\xdf\xa3") == "audio/webm"
