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
