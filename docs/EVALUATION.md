# Empirical Evaluation

Three experiments validate the system's core claims. All scripts live in
[`experiments/`](../experiments/) and write raw results to
`experiments/results/*.json`. Every number below is reproducible by
re-running the scripts (Exp 1 requires a Gemini API key; Exp 2 requires
the ASR models).

---

## Experiment 1 — Reproducibility: LLM-generated IPA scoring (v1) vs deterministic G2P (v2)

**Question.** The pre-refactor system asked the LLM to transcribe both
sentences to IPA and scored the Levenshtein distance between those strings.
Is that a valid measurement instrument?

**Method.** The identical input pair (target 감사합니다, ASR hypothesis
감사하무니다 — a typical epenthesis error) was scored 10 times by each
method. v1 replicates the original prompt logic verbatim, including
`temperature=0.2`. Script: [`exp1_reproducibility.py`](../experiments/exp1_reproducibility.py).

**Results.**

| Method | mean | sd | range | distinct IPA transcriptions of the *same* target |
|---|---|---|---|---|
| v1 — LLM-generated IPA | 91.9 | 1.45 | 89–93 | **4** (`[kamsahamnida]`, `kamsahamnida`, `kɐm.sɐ.ɦɐm.ni.dɐ`, `[kam.sa.ham.ni.da]`) |
| v2 — deterministic G2P | 92.0 | **0.0** | 92–92 | 1 |

**Interpretation.** v1's score fluctuates over a 5-point range on identical
input because the LLM freely varies bracketing, syllable delimiters, and
vowel symbols (ɐ vs a, ɦ vs h) between calls — the Levenshtein distance then
measures transcription-style noise, not pronunciation. A learner practicing
the same sentence twice could see their "score" move without any change in
their speech. This experiment is the quantitative justification for the
measure-deterministically/interpret-with-LLM architecture.

---

## Experiment 2 — Discriminant validity via synthetic error injection

**Question.** Does the pipeline actually assign lower scores to speech
containing typical Japanese-L1 errors?

**Method.** For 10 target sentences, two clips were synthesized with the
same neural TTS voice: the correct sentence, and an error-injected version
encoding typical L1 patterns (vowel epenthesis, ㅓ→ㅗ, coda repair, …).
Both ran through the full pipeline (ffmpeg → Wav2Vec2 → G2P → jamo
alignment). Script: [`exp2_error_discrimination.py`](../experiments/exp2_error_discrimination.py).

**Results.**

| Metric | Value |
|---|---|
| Pairwise ranking accuracy (correct > error) | **8/10 (80%)** |
| Correct-audio score | 85.8 ± 23.5 |
| Error-audio score | 75.5 ± 17.3 |
| Mean gap | 10.3 points |

**Failure analysis (the interesting part).**

- **괜찮아요** (misranked): Wav2Vec2 transcribed even the *correct* native
  TTS clip as 근로아 (score 25). This is a direct instance of the documented
  confound — the acoustics channel cannot distinguish its own ASR error from
  a pronunciation error. Mitigation on the roadmap: fine-tuning on accented
  Korean and CTC-confidence gating.
- **맛있어요 → 마시소요** (tie): the ASR recognized the error-injected clip
  as the correct sentence, i.e. the acoustic model's implicit lexical bias
  "auto-corrected" the perturbation. Small segmental perturbations near
  high-frequency words can be absorbed by the recognizer.

**Threats to validity.** TTS renders segmental errors with native prosody,
so this is a controlled perturbation study of pipeline sensitivity — not an
evaluation on genuine L2 speech. The natural next step is a human-rater
correlation study (system score vs native-speaker ratings, Spearman ρ) on
recordings of Japanese learners; see Future Work.

---

## Experiment 3 — G2P engine accuracy on held-out data

**Question.** The rule engine was developed against ~30 textbook examples
(now unit tests). Does it generalize to unseen items?

**Method.** A 60-item evaluation set disjoint from the development/test
examples, gold labels from 표준국어대사전 pronunciation fields:
51 items within the engine's rule scope, 9 items requiring morphological
analysis (out of scope by design). Data:
[`experiments/data/g2p_heldout.tsv`](../experiments/data/g2p_heldout.tsv).
This benchmark runs in CI as a regression gate (`--check`).

**Results.**

| Bucket | Accuracy |
|---|---|
| In-scope (core phonological rules) | **51/51 = 100%** |
| Out-of-scope (morphology-dependent) | 0/9 = 0% (expected) |

Out-of-scope failures are exactly the documented limitation categories:
stem-final nasal tensification (앉고→[안꼬]), exceptional cluster resolution
(밟다→[밥따]), ㄴ-insertion in compounds (꽃잎→[꼰닙]), and
semantic-boundary liaison (맛없다→[마덥따]). Fixing these requires a
morphological analyzer, which is on the roadmap.

**Note on library comparison.** A head-to-head with `g2pK` was attempted but
the library cannot be installed on Windows/Python 3.13 (its `python-mecab-ko`
dependency fails to build). A comparison on Linux CI is future work.

---

## Future Work (evaluation)

1. **Human-rater correlation study** — collect recordings from Japanese
   learners, have native Korean speakers rate them, report Spearman ρ
   between system scores and human ratings. This is the decisive validity
   test that synthetic perturbation cannot replace.
2. **GOP baseline** — implement Goodness of Pronunciation (Witt & Young,
   2000) from Wav2Vec2 phoneme posteriors and compare error-detection
   performance against the alignment-based method.
3. **Real L2 corpora** — evaluate on accented-Korean datasets (e.g. AI-Hub
   L2 Korean speech) once licensing permits.
