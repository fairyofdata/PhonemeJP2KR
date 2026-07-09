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
    # morphology-conditioned: liaison exceptions (§15 & §15 다만)
    ("맛없다", "마덥따"),
    ("맛있다", "마싣따"),   # §15 다만 lexical exception (마싣따/마딛따 both standard)
    ("멋있다", "머싣따"),
    ("헛웃음", "허두슴"),
    ("값어치", "가버치"),
    # morphology-conditioned: ㄴ-insertion (§29)
    ("꽃잎", "꼰닙"),
    ("나뭇잎", "나문닙"),
    ("풀잎", "풀립"),
    ("물약", "물략"),
    ("색연필", "생년필"),
    ("식용유", "시굥뉴"),
    ("한여름", "한녀름"),
    ("맨입", "맨닙"),
    ("솜이불", "솜니불"),
    ("서울역", "서울력"),
    # morphology-conditioned: stem tensification & coda choice (§24-25, §10-11 단서)
    ("신다", "신따"),
    ("안고", "안꼬"),
    ("넓게", "널께"),
    ("읽고", "일꼬"),
    ("밟다", "밥따"),
    ("밟아", "발바"),      # before a vowel the ㄼ cluster liaisons normally
    # morphology-conditioned: ㄴ+ㄹ → [ㄴㄴ] in Sino-Korean derivation (§20 다만)
    ("의견란", "의견난"),
    # regression guards: boundaries that must NOT trigger morphology rules
    ("학생입니다", "학쌩임니다"),   # copula 이- is bound: no ㄴ-insertion
    ("책입니다", "채김니다"),
    ("절약", "저략"),              # Sino-Korean single morphemes: plain liaison
    ("선약", "서냑"),
    ("밀약", "미략"),
    ("한류", "할류"),              # lateralization intact where §20 다만 does not apply
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
