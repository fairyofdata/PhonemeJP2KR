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
