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

from . import ipa
from .g2p import ipa_segments, is_hangul_syllable, jamo_positions, to_surface
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


def _runs(spans, linked) -> list:
    """Word indices grouped into the chunks the G2P pronounced together."""
    runs = []
    for i in range(len(spans)):
        if i and (i - 1) in linked:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def _surfaces(target: str, spans: list, linked) -> list:
    """Standard pronunciation per word; a linked chunk is split back by
    syllable count (no rule changes how many syllables a word has)."""
    chunks = iter(to_surface(target, linked).split())
    out = [""] * len(spans)
    for run in _runs(spans, linked):
        if not any(is_hangul_syllable(c) for w in run for c in target[slice(*spans[w])]):
            continue
        chunk, at = next(chunks, ""), 0
        for w in run:
            n = sum(is_hangul_syllable(c) for c in target[slice(*spans[w])])
            out[w], at = chunk[at:at + n], at + n
    return out


def _ipa_of(pairs, idxs) -> str:
    return "".join(pairs[i].hyp_ipa for i in idxs if pairs[i].op != "del")


def _ipa_diff(pairs, idxs) -> list:
    return [[pairs[i].ref_ipa, pairs[i].hyp_ipa, pairs[i].op] for i in idxs]


def word_view(target: str, whisper_text: str, wav2vec_text: str,
              phone_ipa: str = None) -> list:
    """Align every channel to the target, word by word.

    Returns one dict per target word:
        target        the word as written
        surface       its standard pronunciation (G2P)
        target_ipa / heard_ipa / acoustic_ipa
                      IPA of each reading (the ASR channels via the G2P,
                      in context); acoustic_ipa_diff marks what differs
        phones, phones_diff, phone_errors
                      the IPA channel (``phone_ipa``, compared without the
                      G2P by src/ipa.py); None when there is no IPA input
        linked_next / linked_prev / linked_surface
                      the boundary after / before this word was scored as
                      one phrase, and that phrase's pronunciation
                      (밥 먹어 → [밤머거])
        heard         Whisper's segment (display only, not scored)
        acoustic      Wav2Vec2's segment (the scored channel)
        katakana      rule-based katakana of the acoustic segment
        acoustic_errors / heard_errors
                      tags from scoring.classify_pair (no timestamps here;
                      the stored result carries them)
    """
    acoustic = score_pronunciation(target, wav2vec_text)
    linked = acoustic.linked                    # one target reading for every channel
    heard = score_pronunciation(target, whisper_text, linked=linked)
    a_words = split_by_word(target, wav2vec_text, acoustic.pairs)
    h_words = split_by_word(target, whisper_text, heard.pairs)

    produced = [(p.hyp, p.hyp_pos) for p in acoustic.pairs if p.hyp_pos is not None]
    kana = kana_by_position(produced)
    spans = [w["span"] for w in a_words]
    surfaces = _surfaces(target, spans, linked)
    # the whole phrase a linked word belongs to, for the UI note
    phrase = {}
    for run in _runs(spans, linked):
        if len(run) > 1:
            for w in run:
                phrase[w] = "".join(surfaces[x] for x in run)
    # per-word target IPA, from the same segments the alignment used
    seq = jamo_positions(target, linked)
    word_of = {i: w for w, (s, e) in enumerate(spans) for i in range(s, e)}
    target_ipas = [""] * len(spans)
    for (_, pos), seg in zip(seq, ipa_segments(seq)):
        target_ipas[word_of[pos]] += seg
    phone_words = None
    if phone_ipa:
        phone_words = ipa.split_by_word(ipa.compare(target, phone_ipa, linked), len(a_words))

    out = []
    for i, (a, h) in enumerate(zip(a_words, h_words)):
        s, e = a["span"]
        pw = phone_words[i] if phone_words else None
        out.append({
            "index": i,
            "target": target[s:e],
            "surface": surfaces[i],
            "target_ipa": target_ipas[i],
            "linked_next": i in linked,
            "linked_prev": (i - 1) in linked,
            "linked_surface": phrase.get(i, ""),
            "heard_ipa": _ipa_of(heard.pairs, h["pairs"]),
            "acoustic_ipa": _ipa_of(acoustic.pairs, a["pairs"]),
            "acoustic_ipa_diff": _ipa_diff(acoustic.pairs, a["pairs"]),
            "phones": "".join(pw["phones"]) if pw else None,
            "phones_diff": pw["diff"] if pw else None,
            "phone_errors": pw["tags"] if pw else [],
            "heard": _segment(whisper_text, h["chars"]),
            "acoustic": _segment(wav2vec_text, a["chars"]),
            "katakana": "".join(kana.get(c, "") for c in a["chars"]),
            "acoustic_errors": _tags(acoustic.pairs, a["pairs"]),
            "heard_errors": _tags(heard.pairs, h["pairs"]),
        })
    return out
