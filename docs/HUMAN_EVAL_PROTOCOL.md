# Human-Rater Correlation Study — Protocol

**Status: designed, not yet executed** (recordings pending).
This document specifies the full pipeline so that the study can be run
later — by the author, a collaborator, or an AI assistant — without any
additional design work. The analysis half is already implemented and
self-tested: [`experiments/exp5_human_correlation.py`](../experiments/exp5_human_correlation.py).

## 1. Research question and hypothesis

Does the deterministic phoneme score agree with native listeners'
judgments of real Japanese-accented Korean speech?

- **H1**: Spearman ρ between system score and mean native intelligibility
  rating ≥ **0.6** (pre-registered threshold; ρ in the 0.4–0.6 band would
  indicate partial validity requiring error analysis, < 0.4 falsifies the
  scoring approach for real L2 speech).
- Interim evidence from surrogate studies: severity-injection Spearman
  ρ = −0.702 (Exp 4), pairwise ranking accuracy 80% (Exp 2). These used
  TTS perturbations; this study replaces synthetic severity with human
  judgment on genuine L2 speech.

## 2. Materials

Use the 10 sentences from Experiment 2 (`experiments/exp2_error_discrimination.py`,
`PAIRS` list, target column). They are balanced for the documented L1
error surfaces: coda-bearing syllables (받침), the lenis/aspirated/tense
triad, ㅓ/ㅗ and ㅡ/ㅜ contrasts, and nasal codas.
Sentence list may be extended, but keep every speaker recording the
**same** sentence set (fully crossed design).

## 3. Participants

- **Speakers**: ≥ 5 Japanese L1 learners of Korean, mixed proficiency
  (ideally 2 beginner / 2 intermediate / 1 advanced). Each records all 10
  sentences → **≥ 50 clips**. With N = 50 and expected ρ ≈ 0.6, the 95% CI
  half-width is roughly ±0.2 — adequate for the hypothesis test.
- **Raters**: ≥ 3 native Korean speakers, no phonetics training required.

## 4. Recording procedure

- Quiet room; smartphone microphone is acceptable. One sentence per file.
- Any common format (wav/mp3/m4a — the harness normalizes via ffmpeg).
- File naming: `spk{ID}_s{sentence#}.{ext}` (e.g. `spk3_s07.wav`).
- Show the speaker the orthographic sentence only (no audio model, no IPA)
  — we measure their internalized pronunciation, not imitation ability.
- One take per sentence (no retakes) to preserve natural error rates.
- **Consent**: written consent that recordings are used for evaluation and
  may appear in aggregate results; raw audio is never published.

## 5. Rating procedure

- Raters hear clips in **shuffled order, blind** to speaker identity and
  to the system's scores.
- Each clip is rated on **intelligibility** (primary), 1–5:

| Rating | Anchor |
|---|---|
| 5 | Completely natural; could be a native speaker |
| 4 | Fully intelligible; minor accent noticeable |
| 3 | Intelligible with effort; several noticeable errors |
| 2 | Partially intelligible; listener must guess some words |
| 1 | Largely unintelligible |

- Optional secondary scale: accentedness (1–5), recorded in the same CSV
  with rater_id suffixed `_acc` if collected.

## 6. Data layout

```
experiments/data/human_eval/
├── audio/                 # the recordings
│   ├── spk1_s01.wav
│   └── ...
├── manifest.csv           # audio_file,speaker_id,target_text
└── ratings.csv            # audio_file,rater_id,rating
```

Templates with headers and one example row are provided in
`experiments/data/human_eval/`. `target_text` must be the orthographic
sentence (the harness applies G2P itself).

## 7. Analysis (already implemented)

```bash
python experiments/exp5_human_correlation.py
```

The harness computes, and writes to `experiments/results/exp5_human_correlation.json`:

1. **Primary**: Spearman ρ (system score vs mean rating) with percentile
   bootstrap 95% CI (2000 resamples, fixed seed).
2. Pearson r (secondary; linearity is not assumed).
3. **Inter-rater reliability**: mean pairwise Spearman across raters —
   report alongside the primary result; if < 0.5, ratings are too noisy
   to validate against and more raters/anchoring are needed.
4. Per-speaker breakdown (does the score track proficiency ordering?).

Verify the harness anytime without data: `--selftest` synthesizes 5 TTS
clips + dummy ratings and runs the identical code path.

## 8. Interpretation & reporting rules (pre-registered)

- Report the CI, not just the point estimate.
- Report **all** clips; no post-hoc exclusion except unreadable audio
  (document any exclusion).
- The known failure mode to check first if ρ is low: Wav2Vec2
  misrecognizing *correct* speech (see Exp 2/4 failure analyses). Inspect
  the `asr` field of low-scoring clips with high human ratings — if the
  ASR text is garbage on clearly intelligible audio, the bottleneck is
  the acoustic model (motivates fine-tuning), not the scoring method.
  This diagnostic distinction is itself a reportable finding.

## 9. Alternatives considered (and why they were deferred/rejected)

- **AI-Hub L2 Korean speech corpora** (e.g. 외국인 한국어 발화 데이터):
  the most promising no-recruitment path — pre-rated L2 speech at scale.
  Requires AI-Hub account approval and license review; revisit before
  recruiting speakers, since it may replace sections 3–5 entirely.
- **LLM-as-judge on audio** (e.g. Gemini rating clips 1–5): rejected as a
  *validation* instrument — it would validate one unproven model with
  another, and contradicts the project's core design principle that
  measurement must be deterministic and auditable. Could be reported as a
  tertiary comparison only after human ratings exist.
