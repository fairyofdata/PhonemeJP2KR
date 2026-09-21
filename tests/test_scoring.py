"""Scoring, alignment, and L1-error classification tests."""

from src.scoring import align_jamo, classify_errors, score_pronunciation


def test_perfect_match_scores_100():
    report = score_pronunciation("감사합니다", "감사합니다")
    assert report.score == 100
    assert report.error_tags == []


def test_surface_equivalent_orthography_scores_100():
    # ASR may emit the surface spelling; phonemically identical → full score
    report = score_pronunciation("감사합니다", "감사함니다")
    assert report.score == 100


def test_score_is_deterministic():
    a = score_pronunciation("안녕하세요", "안녕하세유")
    b = score_pronunciation("안녕하세요", "안녕하세유")
    assert a.score == b.score
    assert 0 < a.score < 100


def test_empty_hypothesis():
    report = score_pronunciation("감사합니다", "")
    assert report.score == 0


def test_empty_target():
    report = score_pronunciation("", "감사합니다")
    assert report.score == 0


def test_alignment_ops():
    pairs, dist = align_jamo(list("abc"), list("adc"))
    assert dist == 1
    assert [p.op for p in pairs] == ["match", "sub", "match"]


def test_vowel_epenthesis_detection():
    # 밥 → 바브: mora-timed repair inserts ㅡ after the coda
    report = score_pronunciation("밥", "바브")
    assert any(t["tag"] == "vowel_epenthesis" for t in report.error_tags)


def test_coda_deletion_detection():
    report = score_pronunciation("밥", "바")
    assert any(t["tag"] == "coda_deletion" for t in report.error_tags)


def test_laryngeal_confusion_detection():
    # 딸 (tense) mispronounced as 달 (lenis)
    report = score_pronunciation("딸", "달")
    assert any(t["tag"] == "laryngeal_confusion" for t in report.error_tags)


def test_vowel_confusion_detection():
    # 서울 pronounced as 소울 (ㅓ → ㅗ)
    report = score_pronunciation("서울", "소울")
    assert any(t["tag"] == "vowel_ʌ_o_confusion" for t in report.error_tags)


def _tags(target, hyp):
    return [t["tag"] for t in score_pronunciation(target, hyp).error_tags]


def test_glide_vowel_confusion_detection():
    # 여기 → 요기: the ʌ/o merger carried over to the j-onglide (ㅕ → ㅛ)
    assert "vowel_jʌ_jo_confusion" in _tags("여기", "요기")


def test_ui_monophthongization_detection():
    # word-initial 의 must stay [ɰi]; JP has no ɰ-glide → 으 / 이
    assert "diphthong_ɰi_monophthongization" in _tags("의사", "이사")
    assert "diphthong_ɰi_monophthongization" in _tags("의사", "으사")


def test_permitted_ui_variants_not_penalized():
    # 표준발음법 5항: 희망 → [히망] (required), 회의 → [회이] (permitted)
    assert score_pronunciation("희망", "히망").score == 100
    assert score_pronunciation("회의", "회이").score == 100
    assert score_pronunciation("가져", "가저").score == 100


def test_nasal_coda_confusion_includes_m():
    assert "nasal_coda_confusion" in _tags("산", "상")
    assert "nasal_coda_confusion" in _tags("감", "간")


def test_nasal_onset_substitution_is_not_a_coda_error():
    # 나무 → 마무: onset ㄴ/ㅁ swap is not the 撥音 coda pattern
    assert "nasal_coda_confusion" not in _tags("나무", "마무")


def test_stop_coda_confusion_detection():
    # 밥 → 박: unreleased coda place lost (JP 促音 has no place of its own)
    assert "stop_coda_confusion" in _tags("밥", "박")
    # coda before a consonant: 입구 → 익구 (surface 입꾸 → 익꾸)
    assert "stop_coda_confusion" in _tags("입구", "익구")


def test_stop_onset_substitution_is_not_a_coda_error():
    assert "stop_coda_confusion" not in _tags("바다", "가다")


# --- CTC char-timestamp threading (forced-alignment path) -------------------

def test_timestamps_thread_through_to_error_tags():
    # 밥 → 바브: the epenthetic ㅡ comes from the 2nd syllable's time span
    char_timestamps = [("바", 0.0, 0.2), ("브", 0.2, 0.4)]
    report = score_pronunciation("밥", "바브", char_timestamps)
    epenthesis = [t for t in report.error_tags if t["tag"] == "vowel_epenthesis"]
    assert epenthesis and epenthesis[0]["timestamp"] == 0.2


def test_no_timestamps_keeps_plain_tags():
    report = score_pronunciation("밥", "바브")
    assert all("timestamp" not in t for t in report.error_tags)


def test_timestamped_jamo_sequence_matches_plain_sequence():
    from src.g2p import to_jamo_sequence

    text = "감사합니다"
    char_timestamps = [(ch, i * 0.1, (i + 1) * 0.1) for i, ch in enumerate(text)]
    tagged = to_jamo_sequence(text, char_timestamps)
    plain = to_jamo_sequence(text)
    # same jamo content, each carrying its source syllable's time span
    assert [t[0] for t in tagged] == plain
    assert all(isinstance(t, tuple) and len(t) == 3 for t in tagged)
    # 합 is the 3rd syllable → its jamo inherit the (0.2, 0.3) span
    ham_jamos = [t for t in tagged if t[0] in ("ㅎ",)]
    assert ham_jamos[0][1] == 0.2


def test_timestamps_survive_spacing_differences():
    # timestamps are matched against non-space chars only
    char_timestamps = [("감", 0.0, 0.1), ("사", 0.1, 0.2), ("합", 0.2, 0.3),
                       ("니", 0.3, 0.4), ("다", 0.4, 0.5)]
    report = score_pronunciation("감사합니다", "감사 합니다", char_timestamps)
    assert report.score == 100


def test_drills_cover_only_classifier_tags():
    # every drill must train a tag the classifier can actually emit
    import inspect

    from src import scoring
    from src.drills import DRILLS

    source = inspect.getsource(scoring.classify_pair)
    for drill in DRILLS:
        assert f'"{drill["tag"]}"' in source, drill["tag"]
        assert drill["sentences"]


def test_every_classifier_tag_has_a_ui_label():
    import inspect
    import re

    from src import scoring
    from src.labels import TAG_LABELS

    emitted = set(re.findall(r'tag_dict\["tag"\] = "([^"]+)"',
                             inspect.getsource(scoring.classify_pair)))
    assert emitted and emitted <= set(TAG_LABELS)
