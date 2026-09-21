"""Word-level view of the jamo alignment (display and export only).

Wav2Vec2-CTC output carries no spaces, so it cannot be split into words
on its own. The scorer's jamo alignment already pairs every target jamo
with a produced one, and both sides know the character they came from
(src/g2p.py: jamo_positions). So:

    1. each pair belongs to the target word of its target jamo;
       an insertion belongs to the word of the pair before it
       (the first word, if it opens the sentence);
    2. each produced character belongs to the word that holds its vowel —
       every syllable has exactly one, so a character whose consonants
       straddle a word boundary (liaison) is still assigned once;
    3. a word's produced segment is its characters in order.

Error tags per word come from scoring.classify_pair over the very pairs
the score was computed from, so the word view never disagrees with the
score. The same procedure is run for Whisper for display; Whisper's tags
describe what a listener model heard and never enter the score.
"""

from .g2p import is_hangul_syllable, to_surface
from .kana import kana_by_position
from .scoring import classify_pair, score_pronunciation

_VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")


def _word_spans(text: str) -> list:
    """[(start, end)] of whitespace-separated words in ``text``."""
    spans, start = [], None
    for i, ch in enumerate(text + " "):
        if ch.isspace():
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    return spans


def _pair_words(pairs, word_of_pos) -> list:
    """Target-word index of every pair (rule 1)."""
    out, cur = [], None
    for p in pairs:
        if p.ref_pos is not None:
            cur = word_of_pos[p.ref_pos]
        out.append(cur)
    first = next((w for w in out if w is not None), 0)
    return [first if w is None else w for w in out]


def split_by_word(target: str, hyp_text: str, pairs) -> list:
    """Per target word: its pair indices and the produced characters.

    Returns [{"span": (start, end), "pairs": [idx, …], "chars": [pos, …]}].
    """
    spans = _word_spans(target)
    word_of_pos = {i: w for w, (s, e) in enumerate(spans) for i in range(s, e)}
    pair_word = _pair_words(pairs, word_of_pos)

    char_word = {}
    for p, w in zip(pairs, pair_word):          # rule 2: the vowel decides
        if p.hyp_pos is not None and p.hyp in _VOWELS:
            char_word[p.hyp_pos] = w
    for p, w in zip(pairs, pair_word):          # vowel-less leftovers
        if p.hyp_pos is not None:
            char_word.setdefault(p.hyp_pos, w)

    words = [{"span": s, "pairs": [], "chars": []} for s in spans]
    for idx, w in enumerate(pair_word):
        words[w]["pairs"].append(idx)
    for pos in sorted(char_word):
        words[char_word[pos]]["chars"].append(pos)
    return words


def _segment(text: str, chars: list) -> str:
    """Produced characters of one word, keeping any space between them."""
    return text[chars[0]:chars[-1] + 1].strip() if chars else ""


def _tags(pairs, idxs) -> list:
    return [t for t in (classify_pair(pairs, i) for i in idxs) if t]


def word_view(target: str, whisper_text: str, wav2vec_text: str) -> list:
    """Align both ASR channels to the target, word by word.

    Returns one dict per target word:
        target        the word as written
        surface       its standard pronunciation (G2P)
        heard         Whisper's segment (display only, not scored)
        acoustic      Wav2Vec2's segment (the scored channel)
        katakana      rule-based katakana of the acoustic segment
        acoustic_errors / heard_errors
                      tags from scoring.classify_pair (no timestamps here;
                      the stored result carries them)
    """
    acoustic = score_pronunciation(target, wav2vec_text)
    heard = score_pronunciation(target, whisper_text)
    a_words = split_by_word(target, wav2vec_text, acoustic.pairs)
    h_words = split_by_word(target, whisper_text, heard.pairs)

    produced = [(p.hyp, p.hyp_pos) for p in acoustic.pairs if p.hyp_pos is not None]
    kana = kana_by_position(produced)
    # to_surface drops words without Hangul, so hand its words out in order
    surfaces = iter(to_surface(target).split())

    out = []
    for i, (a, h) in enumerate(zip(a_words, h_words)):
        s, e = a["span"]
        out.append({
            "index": i,
            "target": target[s:e],
            "surface": (next(surfaces, "")
                        if any(is_hangul_syllable(c) for c in target[s:e]) else ""),
            "heard": _segment(whisper_text, h["chars"]),
            "acoustic": _segment(wav2vec_text, a["chars"]),
            "katakana": "".join(kana.get(c, "") for c in a["chars"]),
            "acoustic_errors": _tags(acoustic.pairs, a["pairs"]),
            "heard_errors": _tags(heard.pairs, h["pairs"]),
        })
    return out
