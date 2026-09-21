# Decision Record

Load-bearing choices, why they were made, and what they cost. Newest last.
Each entry is meant to be enough to re-open the decision later without
re-deriving the context. Experiment numbers refer to
[EVALUATION.md](EVALUATION.md).

---

## 1. The LLM never measures

**Context.** v1 asked the LLM to transcribe both the target and the
learner's speech into IPA and scored the edit distance between the two
strings.

**Decision.** Measurement (G2P, alignment, scoring, error tagging) is
deterministic Python. The LLM receives the finished evidence and only
interprets it: coaching text. (It also rendered the katakana line until
decision 11 moved that to a rule table.)

**Why.** Experiment 1: on identical input, the v1 scorer moved over a
5-point range (sd 1.45) and produced 4 different IPA transcriptions of
the same target. The deterministic scorer has sd 0.0.

**Consequence.** The quantitative half of the app works with no API key
at all, which the UI states. Feedback quality now depends on the tags we
can produce, not on what the LLM can guess.

---

## 2. Score text similarity over ASR output, not model confidence

**Context.** The classic approach to pronunciation scoring is Goodness of
Pronunciation (Witt & Young, 2000) — a likelihood ratio from the acoustic
model.

**Decision.** Score 1 − PER between the G2P surface form of the target
and of the ASR hypothesis, at the jamo level.

**Why.** Experiment 6b, same 615 clips and same acoustic model: the
alignment score beat CTC-GOP on every axis (상/하 AUC 0.818 vs 0.606,
correlation with human level 0.473 vs 0.153, deviation detection 0.717 vs
0.635).

**Consequence.** The score inherits the ASR's blind spots. Anything that
does not change the transcript — rule application, voicing, intonation —
is invisible (see [L1_TAXONOMY.md](L1_TAXONOMY.md)).

---

## 3. Both sides pass the same G2P

**Decision.** Target and hypothesis are converted to surface form before
alignment, so 감사합니다 and 감사함니다 are the same string.

**Consequence (intended).** Orthographic variation in ASR output stops
being an error.

**Consequence (unintended).** Rule non-application by the learner is
erased as well: a learner who says [합니다] without nasalisation is
transcribed `합니다`, which the G2P maps to [함니다]. This is the single
biggest structural limitation of the current design, and it needs a
pronunciation-faithful (phone-level) recogniser to fix, not more rules.

---

## 4. Morphology as a separate, optional layer

**Decision.** Context-free rules live in `src/g2p.py`; rules that need
morpheme boundaries live in `src/morphology.py` behind Kiwipiepy
(version-pinned), and the engine degrades to the context-free pipeline
when it is absent.

**Why.** Boundary detection is the only part that needs a tagger, and a
naive boundary turns 학생입니다 into \*[학쌩님니다].

**Consequence.** CI runs the full held-out gate with only pytest and
kiwipiepy installed — no torch, no models.

---

## 5. Present the score against a measured reference, not as an absolute

**Context.** Experiment 6 found the score ordinally valid but
uncalibrated: jamo-faithful readings by Japanese-L1 speakers average
79.6, median 81, p10 68.

**Decision.** The result view leads with where the score falls among
faithful readings (≥ median / p10–median / < p10), then shows the raw
score and the change since the previous attempt at the same sentence.
Cut points are read at runtime from `experiments/results/exp6_l2_validation.json`
(`src/reference.py`).

**Why.** Showing "83/100" implied 17 points of learner error that the
data does not support. The retry delta is the reading Experiments 1 and 4
support most directly.

**Consequence.** Re-running Exp 6 after fine-tuning moves the bands
automatically. The bottom band is deliberately not called "error":
P(score < p10 | faithful) = 10% is not P(error | score < p10), and the
deviation-detection AUC is 0.717 — informative, not diagnostic.

---

## 6. Error tags are literature-grounded, and audited for detectability

**Decision.** `src/scoring.py` tags Japanese-L1 interference patterns
drawn from contrastive studies; [L1_TAXONOMY.md](L1_TAXONOMY.md) maps them
against Lee (2022)'s eleven-category inventory and states, per category,
whether this architecture can detect it.

**Why.** An inventory taken from the literature says what *should* occur;
it does not say what this pipeline can *see*. Publishing the gap is
stronger than implying full coverage.

**Consequence.** 8 of 11 categories are implemented. Experiment 7 (script
ready, run pending) will say which of them actually separate proficiency
in real speech rather than firing on ASR noise.

---

## 7. Drop edge noise by timing, never by shape

**Context.** Recorder clicks and throat clears are decoded by the
LM-free channel into syllables and scored as insertions.

**Decision.** `src/preprocess.py` removes a leading/trailing chunk only
when it is separated from the speech by ≥ 0.35 s in the CTC character
offsets *and* removing it raises the score. At most two characters per
edge.

**Why.** An epenthetic vowel (밥 → 바브) looks like a stray syllable but
is continuous with the speech and is a real L1 error. Timing separates
the two; shape does not.

**Consequence.** Noise overlapping the speech is not removable this way.
The UI says what was excluded and a checkbox turns it off.

---

## 8. History stores the whole analysis, and the clip

**Decision.** Each record keeps the full analysis payload plus the
recording (`data/clips/`, most recent 50, pruned on save, deleted with the
record). "Open analysis" rehydrates the result view.

**Why.** Text-only rows could not restore the waveform or the error
markers, which are the parts a learner replays.

**Consequence.** Rows saved before this change reopen as disabled. Disk
use is bounded by the prune limit (`KEEP_CLIPS`).

---

## 9. System fonts, no webfont

**Context.** The header rendered as mangled glyphs in the user's browser
while body text was fine.

**Decision.** No `@import` from Google Fonts; a system CJK stack, named
explicitly on the elements that need it.

**Why.** Streamlit's stylesheet sets Source Sans (no CJK coverage)
directly on markdown content, which beats an inherited app-level font, so
every kanji fell back per glyph. Naming the font on the element fixes it;
dropping the webfont also removes an external fetch the page never needed.

**Consequence.** Typography depends on what the OS ships. Any new custom
element must name the font stack (`FONT` / `FONT_KR` in `src/ui.py`).

---

## 10. ASCII-only launcher

**Decision.** `run_app.bat` prints English, ASCII-only messages; Korean
guidance lives in the READMEs.

**Why.** cmd.exe reads batch files by byte offset. Non-ASCII text
desynchronises the parser, and later lines execute truncated — observed
directly (`REM --- create the virtual…` ran as `reate the virtual…`).

**Consequence.** The launcher is less friendly to read, and correct
everywhere.

---

## 11. Three channels plus one notation; word view from the scorer's alignment

**Context.** The result view showed four equal cards: target, Whisper,
Wav2Vec2 and an LLM-written katakana line. A portfolio review found two
problems: to a Japanese reader the three Korean sentences are unreadable
as wholes, and the katakana card looked like a fourth measurement.

**Decision.**
- There are three channels: the target (standard pronunciation, G2P),
  what was heard (Whisper, not scored), and what was produced
  (Wav2Vec2-CTC, the score's basis). Katakana is a *notation* of the
  third, shown as a sub-line of it and never on the target or Whisper.
- Katakana comes from a fixed Hangul→kana table over the surface jamo
  the scorer compared (`src/kana.py`); the LLM no longer produces it.
- A word view aligns all three channels to the target's words. Wav2Vec2
  output has no spaces, so it is cut through the scorer's own jamo
  alignment: every jamo already knows its source character; a pair
  belongs to the word of its target jamo (insertions to the preceding
  word), and a produced character to the word holding its vowel
  (`src/words.py`). Whisper is aligned the same way.
- Per-word explanations are fixed Japanese templates keyed by tag
  (`labels.explain_error`). Whisper gets the same classifier for display,
  labelled 参考・採点外.

**Why.** Katakana cannot carry what this system measures — lenis /
aspirated / tense, ㅓ/ㅗ, ㅡ/ㅜ and ㄴ/ㅇ codas each collapse to one kana
(unit-tested) — so presenting it as a channel would contradict the
measurement. An LLM transliteration is also not aligned to anything,
so it could not be cut per word, and it varied between runs. Reusing the
scoring alignment, rather than a second aligner, guarantees the word
view never disagrees with the score: the per-word tags concatenate to
exactly the scored tags (test), and the demo take reproduces its stored
22 tags and score 60.

**Consequence.** The word view inherits the scorer's blind spot: the
hypothesis passes the same G2P as the target (decision 3), so a missing
phonological rule is invisible — 죽고도 is read as [죽꼬도], and "no
tensification in 춥고도" cannot be claimed. Whisper tags now exist but
mean something different (what a listener model heard); they never
enter the score or the weak-point profile. Older history records keep
their LLM katakana in the stored text, while the view recomputes the
rule-based line. The demo take is exported as data
([DEMO_TAKE.md](DEMO_TAKE.md)) from the stored record, with no model or
API call, and the export refuses to write if recomputation drifts.
