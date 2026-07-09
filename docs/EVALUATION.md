# Empirical Evaluation

Four executed experiments validate the system's core claims, and a fifth
(the decisive human-rater study) is fully designed and harness-ready.
All scripts live in [`experiments/`](../experiments/) and write raw
results to `experiments/results/*.json`. Every number below is
reproducible by re-running the scripts (Exp 1 requires a Gemini API key;
Exp 2/4 require the ASR models).

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

**Method.** A 73-item evaluation set with gold labels from 표준국어대사전
pronunciation fields, in three buckets: 51 items exercising the
context-free rule pipeline (disjoint from the unit-test examples),
17 items exercising the morphology-conditioned layer (added with
`src/morphology.py`; serves as its regression gate), and 5 items that are
out of scope by design (사잇소리 tensification in native compounds, which
needs semantic compound analysis). Data:
[`experiments/data/g2p_heldout.tsv`](../experiments/data/g2p_heldout.tsv).
The core and morph buckets run in CI as a regression gate (`--check`).

**Results.**

| Bucket | Accuracy |
|---|---|
| Core (context-free phonological rules) | **51/51 = 100%** |
| Morph (morphology-conditioned, Kiwipiepy layer) | **17/17 = 100%** |
| Out-of-scope (semantics-dependent 사잇소리) | 0/5 = 0% (expected) |

The morphology-conditioned bucket covers the categories that were
out-of-scope failures before the Kiwipiepy layer existed: ㄴ-insertion
(꽃잎→[꼰닙]), stem tensification with POS disambiguation in context
(신발을 신고→[신바를 신꼬] vs the noun 신고→[신고]), exceptional cluster
resolution (밟다→[밥따]), liaison blocking (맛없다→[마덥따]), and
Sino-Korean ㄴ+ㄹ→[ㄴㄴ] (의견란→[의견난]). Remaining out-of-scope items
(강가→[강까], 밤길→[밤낄]) require knowing whether a compound is a native
사잇소리 compound — information not present in POS tags.

**Note on library comparison.** A head-to-head with `g2pK` was attempted but
the library cannot be installed on Windows/Python 3.13 (its `python-mecab-ko`
dependency fails to build). A comparison on Linux CI is future work.

---

## Experiment 4 — Graded severity monotonicity (human-rating surrogate)

**Question.** Without human ratings, can we still test whether the score
tracks error *severity*, not just error presence?

**Method.** For 5 base sentences, four TTS clips were synthesized at
severity 0–3, where severity *k* applies the first *k* cumulative error
injections (all attested Japanese-L1 patterns: epenthesis, ㅓ→ㅗ,
laryngeal confusion, coda repair, ŋ-loss). The injected error count is a
controlled ground-truth severity ordinal; a valid scorer must decrease
monotonically. 20 clips, full pipeline.
Script: [`exp4_severity_monotonicity.py`](../experiments/exp4_severity_monotonicity.py).

**Results.**

| Metric | Value |
|---|---|
| Spearman ρ (severity vs score) | **−0.702** [95% bootstrap CI −0.928, −0.325] |
| Mean score by severity 0→3 | 91.8 → 88.2 → 78.0 → 74.8 (strictly decreasing) |
| Monotonic step rate | 13/15 (87%) |
| Perfectly monotone sentences | 4/5 |

**Failure analysis.** The one non-monotone ladder (비빔밥을 먹었어요) had
its *severity-0* clip scored 71 — Wav2Vec2 misrecognized the correct
native TTS audio, the same ASR-error confound documented in Experiment 2.
One ladder (서울에서 만나요) plateaued at 85 for severities 1–3: the
recognizer absorbed the later perturbations, compressing ordinal
resolution at high error densities.

**Interpretation.** The CI excludes zero: the score is a statistically
significant monotone function of controlled error severity. Combined with
Exp 2 this establishes ordinal sensitivity; absolute calibration against
human judgment remains for Exp 5.

---

## Experiment 5 — Human-rater correlation study (designed; awaiting recordings)

The decisive validity test — system score vs native listeners' ratings of
genuine Japanese-accented Korean — cannot run until recordings exist.
Everything except the data is done:

- **Protocol** (materials, ≥5 speakers × 10 sentences, ≥3 native raters,
  anchored 1–5 intelligibility rubric, blinding, consent, pre-registered
  threshold ρ ≥ 0.6 and interpretation rules):
  [docs/HUMAN_EVAL_PROTOCOL.md](HUMAN_EVAL_PROTOCOL.md)
- **Analysis harness** (Spearman/Pearson + bootstrap CIs, inter-rater
  reliability, per-speaker breakdown; CSV in → JSON report out):
  [`exp5_human_correlation.py`](../experiments/exp5_human_correlation.py)
- **Harness verified** end-to-end via `--selftest` (synthesizes 5 TTS
  clips + dummy ratings and runs the identical code path).

An alternative to speaker recruitment — AI-Hub L2 Korean speech corpora —
is documented in the protocol (§9) and should be license-reviewed before
recruiting.

---

## Future Work (evaluation)

1. **Execute Experiment 5** — collect recordings per
   [HUMAN_EVAL_PROTOCOL.md](HUMAN_EVAL_PROTOCOL.md), or substitute an
   AI-Hub L2 corpus after license review; then run the existing harness.
2. **GOP baseline** — implement Goodness of Pronunciation (Witt & Young,
   2000) from Wav2Vec2 phoneme posteriors and compare error-detection
   performance against the alignment-based method.
3. **Accent-aware acoustic model** — fine-tune Wav2Vec2 on
   Japanese-accented Korean to attack the ASR-error confound that appears
   as the dominant failure mode in Exp 2 and Exp 4.
