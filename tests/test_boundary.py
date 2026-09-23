"""Word boundaries read as one phrase (표준발음법 §15, §18 붙임, §29 붙임2)."""

import pytest

from src.g2p import linkable_boundaries, to_surface
from src.ipa import compare as ipa_compare
from src.ipa import tokenize
from src.scoring import score_pronunciation
from src.words import word_view

# (text, pausing reading, reading as one phrase)
CASES = [
    ("노란 옷이", "노란 오시", "노라노시"),      # §13 liaison across the gap
    ("꽃 위", "꼳 위", "꼬뒤"),                  # §15 representative sound + liaison
    ("밭 아래", "받 아래", "바다래"),            # §15
    ("늪 앞", "늡 압", "느밥"),                  # §15
    ("밥 먹어", "밥 머거", "밤머거"),            # §18 붙임 nasalization
    ("옷 입어", "옫 이버", "온니버"),            # §29 붙임2 ㄴ-insertion
]
# the standard's own examples for the two 붙임
STANDARD = [
    ("옷 맞추다", "온맏추다"), ("밥 먹는다", "밤멍는다"), ("책 넣는다", "챙넌는다"),
    ("값 매기다", "감매기다"), ("한 일", "한닐"), ("할 일", "할릴"), ("먹은 엿", "머근녇"),
]


@pytest.mark.parametrize("text,paused,phrase", CASES)
def test_both_readings_are_fixed(text, paused, phrase):
    assert to_surface(text) == paused
    assert to_surface(text, (0,)) == phrase
    assert linkable_boundaries(text) == [0]


@pytest.mark.parametrize("text,phrase", STANDARD)
def test_standard_examples_for_the_two_addenda(text, phrase):
    assert to_surface(text, (0,)) == phrase


def test_linking_only_where_it_changes_the_pronunciation():
    assert linkable_boundaries("그 곳은 험한 곳") == []
    assert linkable_boundaries("밥 먹어 옷 입어") == [0, 2]


@pytest.mark.parametrize("text,said", [("밥 먹어", "밤머거"), ("옷 입어", "온니버")])
def test_a_correct_linked_reading_scores_clean(text, said):
    report = score_pronunciation(text, said)
    assert report.error_tags == [] and report.score == 100
    assert report.linked == [0]


@pytest.mark.parametrize("text,said", [("밥 먹어", "밥 머거"), ("옷 입어", "옫 이버")])
def test_the_pausing_reading_still_scores_clean(text, said):
    report = score_pronunciation(text, said)
    assert report.error_tags == [] and report.score == 100
    assert report.linked == []


def test_each_boundary_is_judged_on_its_own():
    report = score_pronunciation("밥 먹어 옷 입어", "밥 머거 온니버")
    assert report.linked == [2] and report.error_tags == []


def test_an_error_elsewhere_does_not_link_a_boundary():
    # 머거 → 머고 is not at the boundary, so the boundary stays paused
    report = score_pronunciation("밥 먹어", "밥 머고")
    assert report.linked == []
    assert [t["tag"] for t in report.error_tags] == ["vowel_ʌ_o_confusion"]


def test_the_ipa_channel_compares_against_the_same_reading():
    assert [p.sym for p in _phones("밥 먹어", (0,))] == tokenize("pammʌɡʌ")
    assert ipa_compare("밥 먹어", "pammʌɡʌ", (0,)).error_tags == []
    # without the linked reading the same phones would look like an error
    assert ipa_compare("밥 먹어", "pammʌɡʌ").error_tags != []


def _phones(text, linked):
    from src.ipa import target_phones
    return target_phones(text, linked)


def test_word_view_marks_the_linked_boundary():
    words = word_view("밥 먹어", "밤머거", "밤머거", "pammʌɡʌ")
    assert [w["surface"] for w in words] == ["밤", "머거"]
    assert words[0]["linked_next"] and words[1]["linked_prev"]
    assert words[0]["linked_surface"] == "밤머거"
    assert all(w["acoustic_errors"] == [] and w["phone_errors"] == [] for w in words)


# --- regression: the README demo take (record 10) must not move --------------

DEMO = "화려한 도시를 그리며 찾아왔네 그 곳은 춥고도 험한 곳"
DEMO_ACOUSTIC = "하료한 도시루루 구리묘 차즈아왔네 구 고순 추부고도 호무한 곳"


def test_the_demo_take_is_unaffected_by_boundary_linking():
    report = score_pronunciation(DEMO, DEMO_ACOUSTIC)
    assert report.linked == []
    assert to_surface(DEMO) == "화려한 도시를 그리며 차자완네 그 고슨 춥꼬도 험한 곧"
    assert report.score == 76 and len(report.error_tags) == 13
    assert [w["surface"] for w in word_view(DEMO, DEMO_ACOUSTIC, DEMO_ACOUSTIC)] == [
        "화려한", "도시를", "그리며", "차자완네", "그", "고슨", "춥꼬도", "험한", "곧"]


def test_the_grid_marks_the_link_and_does_not_flag_the_phrase_reading():
    from src.ui import word_grid_html

    html = word_grid_html(word_view("밥 먹어", "밤머거", "밤머거"))
    assert html.count("pc-wlink") == 1 and "‿" in html
    assert "<u>" not in html          # 밤 is the right reading at this boundary
