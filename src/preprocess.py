"""Drop recorder noise that the ASR transcribed at the edges of a take.

Button clicks, a chair creak or a throat clear before/after the sentence
are loud enough to survive any silence gate, and the LM-free acoustics
channel happily decodes them into syllables. Those syllables then score
as insertions and get tagged as pronunciation errors.

What separates them from real speech is *time*: they sit alone, before or
after the sentence, with a clear pause between them and the rest. So the
rule here is deliberately narrow — a leading or trailing chunk is dropped
only when all of the following hold:

  1. it is separated from the rest of the take by at least `min_gap`
     seconds of silence (from the CTC character offsets),
  2. it is at most `max_chars` characters long, and
  3. removing it actually raises the score.

Condition 3 is what protects real pronunciation errors: a learner's
epenthetic vowel (밥 → 바브) is attached to the preceding syllable with no
pause, and even if it were, dropping it would not improve the match
against the target. Noise that overlaps the speech itself is not
separable this way and is left alone.
"""

from dataclasses import dataclass, field

from .scoring import score_pronunciation


@dataclass
class CleanedHypothesis:
    text: str
    char_timestamps: list = field(default_factory=list)
    removed_leading: str = ""
    removed_trailing: str = ""
    score_gain: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.removed_leading or self.removed_trailing)

    @property
    def removed(self) -> str:
        return " / ".join(p for p in (self.removed_leading, self.removed_trailing) if p)


def _speech(char_timestamps):
    """Timestamped entries for non-space characters."""
    return [t for t in char_timestamps if t[0].strip()]


def strip_edge_noise(target: str, hyp: str, char_timestamps,
                     min_gap: float = 0.35, max_chars: int = 2) -> CleanedHypothesis:
    """Return the hypothesis with isolated edge noise removed (see module doc)."""
    entries = _speech(char_timestamps or [])
    keep = CleanedHypothesis(text=hyp, char_timestamps=list(char_timestamps or []))
    if len(entries) < 3:
        return keep

    base = score_pronunciation(target, hyp, char_timestamps).score
    best = keep
    for lead in range(0, max_chars + 1):
        for trail in range(0, max_chars + 1):
            if lead == 0 and trail == 0:
                continue
            if lead + trail >= len(entries):
                continue
            if lead and entries[lead][1] - entries[lead - 1][2] < min_gap:
                continue
            if trail and entries[-trail][1] - entries[-trail - 1][2] < min_gap:
                continue
            kept = entries[lead:len(entries) - trail]
            text = "".join(e[0] for e in kept)
            gain = score_pronunciation(target, text, kept).score - base
            if gain > best.score_gain:
                best = CleanedHypothesis(
                    text=text,
                    char_timestamps=kept,
                    removed_leading="".join(e[0] for e in entries[:lead]),
                    removed_trailing="".join(e[0] for e in entries[len(entries) - trail:]),
                    score_gain=gain,
                )
    return best
