"""Edge-noise stripping: isolated leading/trailing syllables from the recorder."""

from src.preprocess import strip_edge_noise


def stamps(spec):
    """[(char, start, end), ...] from (char, start) pairs with 0.2 s durations."""
    return [(c, t, t + 0.2) for c, t in spec]


def test_isolated_leading_noise_is_dropped():
    # "크" decoded from a button click, 0.6 s before the sentence starts
    ts = stamps([("크", 0.0), ("감", 0.8), ("사", 1.0), ("합", 1.2), ("니", 1.4), ("다", 1.6)])
    cleaned = strip_edge_noise("감사합니다", "크감사합니다", ts)
    assert cleaned.text == "감사합니다"
    assert cleaned.removed_leading == "크" and cleaned.removed_trailing == ""
    assert cleaned.score_gain > 0


def test_isolated_trailing_noise_is_dropped():
    ts = stamps([("감", 0.0), ("사", 0.2), ("합", 0.4), ("니", 0.6), ("다", 0.8), ("음", 1.6)])
    cleaned = strip_edge_noise("감사합니다", "감사합니다음", ts)
    assert cleaned.text == "감사합니다"
    assert cleaned.removed_trailing == "음"


def test_noise_at_both_ends():
    ts = stamps([("크", 0.0), ("감", 0.8), ("사", 1.0), ("합", 1.2), ("니", 1.4), ("다", 1.6),
                 ("어", 2.4)])
    cleaned = strip_edge_noise("감사합니다", "크감사합니다어", ts)
    assert cleaned.text == "감사합니다"
    assert cleaned.removed_leading == "크" and cleaned.removed_trailing == "어"


def test_epenthesis_without_a_pause_is_kept():
    # 밥 → 바브: the inserted vowel is continuous speech and a real L1 error
    ts = stamps([("바", 0.0), ("브", 0.2)])
    cleaned = strip_edge_noise("밥", "바브", ts)
    assert cleaned.text == "바브" and not cleaned.changed


def test_pause_before_a_correct_syllable_is_kept():
    # hesitation before the last word must not remove correct speech
    ts = stamps([("감", 0.0), ("사", 0.2), ("합", 0.4), ("니", 0.6), ("다", 1.4)])
    cleaned = strip_edge_noise("감사합니다", "감사합니다", ts)
    assert cleaned.text == "감사합니다" and not cleaned.changed


def test_noise_touching_the_speech_is_left_alone():
    ts = stamps([("크", 0.0), ("감", 0.25), ("사", 0.45), ("합", 0.65), ("니", 0.85), ("다", 1.05)])
    cleaned = strip_edge_noise("감사합니다", "크감사합니다", ts)
    assert not cleaned.changed


def test_without_timestamps_nothing_is_stripped():
    cleaned = strip_edge_noise("감사합니다", "크감사합니다", [])
    assert cleaned.text == "크감사합니다" and not cleaned.changed


def test_spaces_do_not_count_as_speech():
    ts = stamps([("크", 0.0), (" ", 0.4), ("감", 0.8), ("사", 1.0), ("합", 1.2),
                 ("니", 1.4), ("다", 1.6)])
    cleaned = strip_edge_noise("감사합니다", "크 감사합니다", ts)
    assert cleaned.text == "감사합니다"
