"""IPA channel (src/ipa.py): comparison without the G2P, rule-missed tags."""

import pytest

from src.ipa import compare, from_model_tokens, render, tokenize
from src.scoring import score_pronunciation
from src.words import word_view


def _tags(target, ipa):
    return [(t["tag"], t["ref"], t["hyp"]) for t in compare(target, ipa).error_tags]


def test_multi_character_phones_are_single_units():
    assert tokenize("tɕʰup̚k͈odo") == ["tɕʰ", "u", "p", "k͈", "o", "t", "o"]
    assert tokenize("hwaɾjʌ") == ["h", "wa", "L", "jʌ"]


def test_allophones_and_notation_variants_are_merged():
    assert tokenize("dʑa") == tokenize("tɕa")          # lenis voicing
    assert tokenize("ɕi") == tokenize("si")            # ɕ before i
    assert tokenize("kaɭ") == tokenize("kal") == tokenize("kaɾ")
    assert tokenize("k a ː . t") == ["k", "a", "t"]    # spaces, length, syllable marks


@pytest.mark.parametrize("target,produced,rule", [
    ("춥고도", "tɕʰupkodo", "tensification"),     # [춥꼬도] read as spelled
    ("합니다", "hapnida", "nasalization"),        # [함니다]
    ("신라", "sinla", "lateralization"),          # [실라]
    ("좋다", "tɕota", "aspiration"),              # [조타]
    ("같이", "katʰi", "palatalization"),          # [가치], read with liaison only
    ("곳은", "kotɯn", "liaison"),                 # [고슨]
])
def test_skipped_rules_are_named(target, produced, rule):
    assert f"rule_{rule}_missed" in [t for t, _, _ in _tags(target, produced)]


def test_a_correct_surface_reading_has_no_tags():
    target = "화려한 도시를 그리며 찾아왔네 그 곳은 춥고도 험한 곳"
    ipa = "hwaɾjʌhan toɕiɾɯɭ kɯɾimjʌ tɕʰadʑawanne kɯ kosɯn tɕʰup̚k͈odo hʌmhan kot̚"
    assert compare(target, ipa).error_tags == [] and compare(target, ipa).score == 100


def test_liaison_of_an_affricate_is_not_palatalization():
    # 찾아 → [차자]: the liaison baseline already has [tɕ]; no false rule tag
    assert _tags("찾아", "tɕʰatɕa") == []


def test_ipa_channel_sees_what_the_hangul_channel_cannot():
    # 차즈아왔네 is re-derived by the G2P into [차즈아완네]; the IPA [t̚n] is not
    target = "찾아왔네"
    assert "rule" not in str(score_pronunciation(target, "차즈아왔네").error_tags)
    assert ("rule_nasalization_missed", "n", "t") in _tags(target, "tɕʰadʑɯawat̚ne")


def test_display_restores_allophony_per_word():
    assert "".join(render(tokenize("tɕʰubuɡodo"))) == "tɕʰubuɡodo"
    assert "".join(render(tokenize("toɕiɾuɾu"))) == "toɕiɾuɾu"
    assert "".join(render(tokenize("kot"))) == "kot̚"


def test_recognizer_tokens_map_to_our_notation():
    assert from_model_tokens(["tɕh", "a", "kː", "o"]) == "tɕʰak͈o"
    assert from_model_tokens(["t", "s", "u"]) == "tsu"          # two tokens stay two phones
    assert from_model_tokens(["ts", "u"]) == "tɕu"


def test_word_view_carries_the_ipa_channel_per_word():
    target = "찾아왔네 춥고도"
    words = word_view(target, target, "차즈아왔네 추부고도", "tɕʰadʑɯawat̚ne tɕʰubuɡodo")
    assert [w["phones"] for w in words] == ["tɕʰadʑɯawat̚ne", "tɕʰubuɡodo"]
    assert "rule_tensification_missed" in [t["tag"] for t in words[1]["phone_errors"]]
    assert words[0]["target_ipa"] == "tɕʰadʑawanne"
    assert words[1]["acoustic_ipa"] == "tɕʰubuɡodo"
