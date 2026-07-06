"""G2P rule verification against Standard Korean pronunciation (표준발음법)."""

import pytest

from src.g2p import to_surface, to_ipa, to_jamo_sequence


@pytest.mark.parametrize("orthography, surface", [
    # nasalization (비음화)
    ("감사합니다", "감사함니다"),
    ("입니다", "임니다"),
    ("국물", "궁물"),
    ("놓는", "논는"),
    # lateralization (유음화)
    ("신라", "실라"),
    ("설날", "설랄"),
    # relayed nasalization (ㄹ 비음화 연쇄: 독립 → 동닙)
    ("독립", "동닙"),
    ("심리", "심니"),
    # aspiration via ㅎ (격음화)
    ("좋다", "조타"),
    ("많다", "만타"),
    ("입학", "이팍"),
    ("축하", "추카"),
    # ㅎ-deletion (ㅎ탈락)
    ("좋아요", "조아요"),
    ("많아", "마나"),
    ("싫어", "시러"),
    # palatalization (구개음화)
    ("같이", "가치"),
    ("굳이", "구지"),
    # liaison (연음)
    ("한국어", "한구거"),
    ("음악", "으막"),
    ("앉아", "안자"),
    ("없어요", "업써요"),
    ("읽어", "일거"),
    # coda neutralization (받침 중화)
    ("부엌", "부억"),
    ("옷", "옫"),
    ("있다", "읻따"),
    # tensification (경음화)
    ("학교", "학꾜"),
    ("국밥", "국빱"),
    ("읽다", "익따"),
    ("맛있다", "마싣따"),
    # multi-word sentences (word-internal rules only)
    ("만나서 반갑습니다", "만나서 반갑씀니다"),
    ("안녕하세요", "안녕하세요"),
])
def test_surface_forms(orthography, surface):
    assert to_surface(orthography) == surface


def test_ipa_basic():
    # 감사합니다 → 감사함니다 → kamsahamnida (intervocalic voicing of ㄷ)
    ipa = to_ipa("감사합니다")
    assert ipa.startswith("kamsa")
    assert "mnida" in ipa


def test_ipa_aspiration_and_tension():
    assert "tʰ" in to_ipa("좋다")       # 조타
    assert "k͈" in to_ipa("학교")        # 학꾜
    assert "t͈" in to_ipa("있다")        # 읻따


def test_ipa_palatal_s():
    assert "ɕ" in to_ipa("시간")         # ㅅ + ㅣ → ɕ


def test_ipa_vowel_inventory():
    assert "ʌ" in to_ipa("서울")
    assert "ɯ" in to_ipa("그늘")


def test_jamo_sequence_ignores_spacing_and_punct():
    assert to_jamo_sequence("감사합니다!") == to_jamo_sequence("감사 합니다")


def test_non_hangul_input():
    assert to_surface("hello!") == ""
    assert to_jamo_sequence("...") == []
