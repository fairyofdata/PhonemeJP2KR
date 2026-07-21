# Validating the Ruler Before Measuring the Learner: A Deterministic Grapheme-to-Phoneme Pipeline for Pronunciation Assessment of Japanese-L1 Korean Speech, and an Empirical ASR Noise Floor

**Working draft — prepared as a research writing sample.**
Target venues: Interspeech SLaTE (Speech and Language Technology in Education) workshop; *Phonetics and Speech Sciences* (말소리와 음성과학, Korean Society of Speech Sciences); *Journal of Korean Language Education*.

> **Status of this document.** This is a full-length draft assembled from an implemented and unit-tested system (the *Phomene* / PhonemeJP2KR repository) and six reproducible experiments whose raw outputs are committed as JSON. Every quantitative claim below is traceable to a script in `experiments/` and a result file in `experiments/results/`. Sections that state work as *planned* rather than *done* are marked explicitly. The intent is to demonstrate problem formulation and experimental-validation practice, not to claim a finished peer-reviewed result.

---

## Abstract

Computer-assisted pronunciation training (CAPT) systems return a number — a pronunciation score — and both learners and downstream research treat that number as a measurement. Yet the number is only as trustworthy as the instrument that produces it, and in practice the instrument is rarely validated: recent systems increasingly delegate scoring to large language models (LLMs) prompted to "transcribe to IPA and rate the pronunciation," a design that is fluent-looking but, we show, not reproducible. We take the opposite stance — **the measurement must be deterministic and independently validated before it is allowed to interpret a learner** — and build a pronunciation-assessment pipeline for the specific and under-served case of Japanese-first-language (L1) learners of Korean.

The system separates *measurement* from *interpretation*. Measurement is a deterministic Korean grapheme-to-phoneme (G2P) engine implementing Standard Korean pronunciation rules (표준발음법) plus a morphology-conditioned layer, followed by jamo-level (sub-syllabic phoneme) alignment of the target against a language-model-free acoustic transcription. Interpretation — katakana-based first-language-interference feedback — is delegated to an LLM that never touches the score.

We validate the instrument along a ladder of increasing realism. (i) An LLM IPA scorer, re-run on identical input, fluctuates over a 5-point range (SD 1.45) and emits four different IPA transcriptions of the same word, whereas the deterministic scorer has SD 0.0. (ii) The G2P engine reaches 100% (68/68) on held-out Standard-Korean items within its declared scope. (iii) On controlled text-to-speech perturbations the score discriminates injected first-language errors (80% pairwise ranking accuracy) and decreases monotonically with graded error severity (Spearman ρ = −0.702). (iv) On **615 clips of genuine Japanese-accented Korean** from a 252-speaker corpus, the score separates human speech-level ratings with an area under the ROC curve (AUC) of 0.818 and tracks the deviations that native transcribers actually heard (AUC 0.717). Crucially, we quantify the **ASR noise floor**: even on readings that transcribers heard as faithful, the acoustics channel scores 79.6 on average — a direct, honest measurement of the automatic-speech-recognition (ASR) confound that limits absolute calibration. Finally (v), on the same real-speech sample our alignment-based score dominates a same-budget Goodness-of-Pronunciation (GOP) baseline on every axis (speech-level AUC 0.818 vs. 0.606).

The contribution is not a state-of-the-art score but a **method for earning trust in a score**, instantiated for the Japanese↔Korean pair, with the system's own dominant failure mode measured rather than hidden.

**Keywords:** computer-assisted pronunciation training; mispronunciation detection and diagnosis; grapheme-to-phoneme conversion; Korean as a second language; Japanese L1 interference; measurement validity; reproducibility; Goodness of Pronunciation.

---

## 1. Introduction

### 1.1 The problem: a trusted number with an unexamined provenance

A pronunciation-assessment system exists to answer one question for a learner — *how close was I?* — and it answers with a number. That number then propagates: it drives learner feedback, it becomes a dependent variable in CALL (computer-assisted language learning) studies, and increasingly it is used to rank or gate learners. The entire edifice rests on an assumption that is almost never tested in the systems literature: **that the number is a valid and reliable measurement of pronunciation.**

Two developments make re-examining that assumption urgent.

First, the field has a **measurement-validity blind spot**. Much CAPT work reports that a proposed scorer correlates with human judgment on one dataset and stops there, treating the score as a black box whose internal reproducibility, scope of correctness, and failure modes are not separately established. When the score later disagrees with a human, there is no diagnostic vocabulary for *why*, because the instrument was never decomposed.

Second, the **LLM turn** has made this worse while appearing to make it better. It is now trivial to prompt a capable LLM to "transcribe both the reference and the learner's speech to IPA and output a pronunciation score from 0 to 100." The output is grammatical, plausible, and — as we demonstrate in §4.1 — **not reproducible**: the same audio and text, scored twice, can yield different IPA strings and different numbers, because the model freely varies bracketing, syllable delimiters, and vowel symbols between calls. A learner who practices the same sentence twice may see the score move without any change in their speech. An instrument that is not reproducible is not an instrument; it is a random variable wearing the costume of one.

### 1.2 The population: Japanese-L1 learners of Korean

We situate the problem in a specific L1–L2 pair for reasons that are both practical and scientific.

Japanese-L1 learners are the **second-largest population** in Korean-language-education research (Survey of 258 studies, *Journal of Korean Language Education* 26(1), 2015), and Korean is among the most-studied foreign languages in Japan. Yet the pronunciation difficulties of this population are not random noise to be modeled generically — they are **systematic, contrastively predictable transfer effects** with a mature descriptive literature:

- **Mora-timed rhythm and open-syllable repair.** Japanese phonotactics strongly prefer open (CV) syllables, so learners resyllabify Korean CVC by mapping the coda onto the Japanese moraic units — the sokuon /Q/ and the moraic nasal /N/ — and insert an epenthetic vowel to repair the closed syllable (*밥* /pap̚/ → *バプ* [ba.pɯ]). This is documented both in coda-production studies (장향실, 2016; 하호빈·이화진, 2019) and acoustically as mora-timed rhythm transfer, with the largest deviation precisely in nasal-coda closed syllables (Phonetics and Speech Sciences 10(4)).
- **Two-way → three-way laryngeal collapse.** Japanese distinguishes voiced/voiceless; Korean distinguishes a three-way lenis/aspirated/tense contrast (ㄱ/ㅋ/ㄲ). Learners collapse the three-way system.
- **Vowel-inventory mergers.** The absence of /ʌ/ and /ɯ/ in the Japanese vowel system produces systematic mergers (ㅓ↔ㅗ, ㅡ↔ㅜ) (KCI, vowel-system analysis).
- **Redundant nasal place assimilation.** Where Korean nasalization is required, learners redundantly copy the following consonant's place of articulation (이화진, 2018).

This is a gift to a diagnostic system: the space of likely errors is *known a priori* and can be encoded as a structured taxonomy rather than discovered from data. It is also a gift to the *validation* problem, because a system that is supposed to detect these errors can be tested against them directly. Genericity is the enemy here; the Japanese↔Korean specificity is the strength.

### 1.3 Our stance and contributions

We adopt a single governing principle:

> **Measure deterministically; interpret with the LLM. The LLM never produces the number.**

Concretely, the pipeline computes its score with a rule-based Korean G2P engine and a transparent edit-distance alignment — fully deterministic, unit-tested, reproducible — and only *afterward* hands the structured result (aligned phonemes, error tags, timestamps) to an LLM whose job is pedagogical explanation in the learner's L1, never scoring.

We then validate the instrument along a **ladder of increasing realism**, and our contributions map onto its rungs:

1. **A reproducibility critique of LLM-based pronunciation scoring** (§4.1), quantifying the non-determinism that the fluency of LLM output conceals, and using it as the empirical justification for the architecture.
2. **A dependency-light, morphology-aware deterministic Korean G2P engine**, validated at 100% on a held-out Standard-Korean set within a *declared* scope, with the out-of-scope cases (semantically-conditioned 사잇소리) identified rather than silently mis-handled (§4.2).
3. **Construct-validity evidence on controlled perturbations** — discriminant validity and graded-severity monotonicity — that isolates scorer behavior from the confounds of real recordings (§5).
4. **Validation on real Japanese-accented Korean speech at scale** (615 clips, 198 speakers), correlating the score with two independent human signals from a national corpus (§6.1–6.2).
5. **An empirically measured ASR noise floor** (§6.3) — the system's own dominant failure mode, quantified rather than hidden, which becomes the concrete success criterion for future acoustic-model adaptation.
6. **A three-way comparison against the classic GOP baseline** on the same real speech (§6.4), showing the alignment-based method dominates a same-budget likelihood-ratio GOP on every axis.

The through-line is *epistemic*: the value of this work is less "a good scorer" and more "**a defensible procedure for deciding whether to believe a scorer**," demonstrated end-to-end on a specific language pair.

---

## 2. Related Work

### 2.1 Two lineages of pronunciation scoring

Automatic pronunciation assessment has two broad lineages.

**The acoustic-posterior lineage** descends from **Goodness of Pronunciation** (Witt & Young, 2000). GOP scores a phone by the log-posterior of the intended phone given the audio, typically normalized by duration, computed from a forced alignment against an acoustic model. It is *pronunciation-native* — it never decodes text — and it is still the reference baseline in mispronunciation detection and diagnosis (MDD). Its weakness is that the likelihood is entangled with everything else that makes audio unlike the model's training distribution: channel, speaking rate, and global accent. Neural variants recompute GOP from deep acoustic posteriors, but the entanglement is structural, not incidental.

**The text-similarity lineage** scores the *distance between the intended text and a recognized transcription*. Recent Korean work is directly relevant: scoring L2 Korean by morpheme-level similarity between the reference and the ASR transcript tracks native listeners' comprehension **better than off-the-shelf pronunciation-scoring APIs** (형태소 분석기반 외국인 발화 한국어 발음평가 개선 방법, DBpia). This is independent support for the design we adopt — but at the *morpheme* level, which is too coarse to localize a segmental error. We push the same idea down to the **jamo** (sub-syllabic phoneme) level, which is the granularity at which the Japanese-L1 error taxonomy actually operates.

Our §6.4 is, to our knowledge, a rare *same-model, same-data, same-budget* head-to-head between these two lineages on genuine L2 speech.

### 2.2 LLMs as scorers, and the reproducibility gap

The use of general-purpose LLMs for pronunciation and writing assessment is expanding rapidly, motivated by their fluent, human-like rationales. The systems literature has been slower to interrogate the *statistical* properties of these scores. Our §4.1 contributes a small but pointed measurement: the run-to-run variance of an LLM IPA scorer on fixed input, and the mechanism (free variation in transcription style) that produces it. This positions LLMs where we believe they belong in a measurement pipeline — as an **interpretation** layer over a deterministic measurement, not as the measurement.

### 2.3 Korean G2P and the morphology boundary

Korean pronunciation is governed by the well-codified 표준발음법 (Standard Korean Pronunciation), but a faithful surface form requires two kinds of knowledge. Context-free phonological rules (neutralization, tensification, nasalization, liaison) can be applied over the raw syllable stream. A residual class, however, is **morphologically conditioned**: ㄴ-insertion at compound boundaries (꽃잎 → [꼰닙]), stem-final tensification distinguished by part of speech (the verb 신고 → [신꼬] vs. the noun 신고 → [신고]), and lexical liaison blocking (맛없다 → [마덥따] vs. 맛있다 → [마싣따]). Existing Korean G2P libraries (e.g., `g2pK`) address much of this but carry heavy platform dependencies (a MeCab-family analyzer that, in our experience, fails to build on Windows/Python 3.13). Our engine is pure-Python for the context-free core and single-sources morpheme boundaries from one pinned POS tagger, degrading gracefully to the context-free pipeline when the tagger is absent.

### 2.4 Data resources and direction

Fine-tuning an accent-robust Korean acoustic model for this population requires Japanese-L1 Korean speech. The AI-Hub *Foreign-speaker Korean Speech* corpus family includes a Japanese-L1 subset and is the natural source. We note one direction-sensitive pitfall for completeness: NINJAL's C-JAS corpus is *Japanese*-language speech by Chinese/Korean L1 learners — the **reverse** direction — and is therefore not a source for this task, though it would suit a mirror-image Korean→Japanese system.

---

## 3. System Architecture

### 3.1 The measurement/interpretation split

The system is organized around one architectural commitment, summarized in Table 1.

**Table 1. The measurement/interpretation separation.**

| Layer | Component | Property |
|---|---|---|
| **Measurement** | Rule-based G2P (표준발음법) + jamo alignment | Deterministic, unit-tested, reproducible |
| **Perception probe** | Whisper (strong internal LM) vs. Wav2Vec2-CTC (no LM) | The *gap* between them separates intelligibility from acoustics |
| **Interpretation** | LLM, given structured evidence (error tags, IPA, score) | Pedagogical only: katakana rendering + coaching text |

The same audio always yields the same score. If the LLM is unavailable, the full quantitative analysis still renders; only the natural-language coaching is lost. This is the operational meaning of "the LLM never produces the number."

### 3.2 A dual-ASR perception/production probe

The pipeline transcribes each utterance with **two deliberately asymmetric** recognizers:

- **Whisper** carries a strong internal language model, so its output approximates what a native listener *understands* after their brain auto-corrects the signal with context. We call this the **intelligibility channel**.
- **Wav2Vec2 with greedy CTC decoding** has no language model, so its output stays close to the raw phone sequence actually produced. We call this the **acoustics channel**.

The divergence between the two channels operationalizes the **perception/production gap** that L2 learners cannot self-monitor — the phenomenon of *phonological deafness*, in which the L1's perceptual categories filter the acoustic signal before it reaches awareness, so the learner literally cannot hear the difference between what they produced and what they intended. Scoring, described next, is computed against the **acoustics channel**, because the goal is to measure production, not to reward an intelligibility that the recognizer's language model may have manufactured.

### 3.3 The deterministic G2P engine

The engine (`src/g2p.py`) converts orthography to a surface pronunciation form and then to IPA, via an ordered rule pipeline. **Rule order is load-bearing** and follows the feeding/bleeding structure of Korean phonology:

0. **Morphology-conditioned rewrite** (optional; §3.4)
1. ㅎ-cluster rules — aspiration and ㅎ-deletion (표준발음법 §12): 좋다 → [조타], 좋아 → [조아], 입학 → [이팍]
2. Palatalization (§17): 같이 → [가치], 굳이 → [구지]
3. Liaison (§13–14): 한국어 → [한구거], 없어요 → [업써요]
4. Coda neutralization — the seven-coda rule (§8–11): 부엌 → [부억], 꽃 → [꼳]
5. Post-obstruent tensification (§23): 학교 → [학꾜], 국밥 → [국빱]
6. Nasal/liquid assimilation (§18–20): 합니다 → [함니다], 신라 → [실라], 독립 → [동닙]

Liaison (step 3) must precede neutralization (step 4): 없어요 must resyllabify to [업써요] *before* the coda cluster is simplified, or the /s/ is lost. This ordering is not a convenience; it is the phonology, and encoding it correctly is part of what the held-out evaluation in §4.2 tests.

Hangul's algorithmic block structure makes decomposition exact: a syllable at Unicode code point *c* decomposes into (choseong, jungseong, jongseong) by integer arithmetic over the base 0xAC00, with no lookup table and no ambiguity. Surface forms then map to IPA with basic allophony — intervocalic lenis voicing (/k/ → [ɡ]), ㅅ-palatalization before front glides ([s] → [ɕ]) — yielding e.g. `만나서 반갑습니다` → `/mannasʌ panɡap̚s͈ɯmnida/`.

A subtle but important property: **both** the reference sentence and the ASR hypothesis pass through the *same* G2P before comparison. This neutralizes orthographic variation — 감사합니다 and its pronunciation spelling 감사함니다 both surface identically and score 100 against each other — so the score reflects pronunciation distance, not spelling distance.

### 3.4 The morphology layer

Morphologically-conditioned rules are applied first (`src/morphology.py`) by rewriting orthography into a *pronunciation spelling* (발음 표기) that the context-free pipeline then derives normally — e.g. 꽃잎 → 꽃닢 → [꼳닢] → [꼰닙]. Boundary detection is **single-sourced** from one POS tagger (Kiwipiepy, version-pinned for determinism); compounds the tagger lumps into a single token are taught their internal boundary through a small declarative lexicon rather than by pattern-matching characters in running text. Without the tagger installed, the engine degrades to the context-free pipeline in a documented, testable way. This layer resolves exactly the cases that motivate morphology in Korean G2P: §29 ㄴ-insertion, §24–25 stem tensification with POS disambiguation, §10–11 단서 cluster overrides, §15 liaison blocking, and §20 다만 Sino-Korean ㄴ+ㄹ → [ㄴㄴ].

### 3.5 Jamo alignment and the L1 error taxonomy

Both surface forms are decomposed to a flat jamo sequence (spaces and punctuation excluded, so the score is insensitive to tokenization), aligned by Levenshtein dynamic programming with backtrace, and scored:

$$\text{score} = \mathrm{round}\!\left(100 \times \left(1 - \frac{\text{edit distance}}{\max(|\text{ref}|,\,|\text{hyp}|)}\right)\right) = \mathrm{round}(100 \times (1 - \text{PER}))$$

where PER is the phoneme error rate. The alignment trace feeds two consumers: the learner-facing per-phoneme diff, and a **rule-based classifier** that tags each non-matching alignment operation with a known Japanese-L1 interference pattern (Table 2). Because each tag corresponds to a documented contrastive-phonology phenomenon, the LLM receives *structured evidence* — "vowel epenthesis at timestamp 0.62s" — rather than a raw string to speculate over.

**Table 2. The L1 error taxonomy, grounded in contrastive phonology.**

| Tag | Phenomenon | Example | Grounding |
|---|---|---|---|
| `vowel_epenthesis` | Moraic CV-repair after a coda | 밥 → 바브 | 장향실 2016; 하호빈·이화진 2019 |
| `coda_deletion` | Coda drop | 밥 → 바 | 이화진 2021 |
| `laryngeal_confusion` | Three-way laryngeal collapse | 딸 → 달 | (contrastive) |
| `vowel_ʌ_o_confusion` | ㅓ/ㅗ merger (no /ʌ/ in Japanese) | 서울 → 소울 | KCI vowel-system |
| `vowel_ɯ_u_confusion` | ㅡ/ㅜ merger (no /ɯ/ in Japanese) | 그 → 구 | KCI vowel-system |
| `nasal_coda_confusion` | ㄴ/ㅇ merged toward moraic ん | 산 → 상 | 이화진 2018 |

---

## 4. Validating the Instrument I: Reproducibility and Correctness

Before asking whether the score agrees with humans, we ask two prior questions that the field usually skips: is the score *reproducible*, and is the G2P engine *correct within a declared scope*?

### 4.1 Experiment 1 — LLM-generated IPA scoring is not reproducible

**Question.** The pre-refactor version of this system asked an LLM to transcribe both sentences to IPA and scored the Levenshtein distance between those strings. Is that a valid measurement instrument?

**Method.** A single fixed input pair — target 감사합니다, ASR hypothesis 감사하무니다 (a canonical vowel-epenthesis error) — was scored *ten times* by each method. Version 1 (v1) replicates the original LLM prompt logic verbatim, including `temperature = 0.2`; version 2 (v2) is the deterministic G2P scorer. Script: `exp1_reproducibility.py`.

**Results (Table 3).**

**Table 3. Reproducibility on identical input, ten runs.**

| Method | Mean | SD | Range | Distinct IPA transcriptions of the *same* target |
|---|---|---|---|---|
| v1 — LLM-generated IPA | 91.9 | **1.45** | 89–93 | **4** (`[kamsahamnida]`, `kamsahamnida`, `kɐm.sɐ.ɦɐm.ni.dɐ`, `[kam.sa.ham.ni.da]`) |
| v2 — deterministic G2P | 92.0 | **0.0** | 92–92 | 1 |

**Interpretation.** The LLM's score fluctuates over a 5-point range on *byte-identical* input because the model freely varies bracketing (`[...]` vs. none), syllable delimiters (`.`), and vowel symbols (ɐ vs. a, ɦ vs. h) between calls. The Levenshtein distance then measures **transcription-style noise, not pronunciation**. Two observations matter. First, the *magnitude* (SD 1.45, range 4) is large enough to be pedagogically visible: a learner repeating one sentence sees the score move with no change in speech. Second, and more damning, the noise is in the **instrument's own encoding of the reference** — the target 감사합니다 never changes, yet it receives four different IPA strings. This is the empirical foundation of the entire architecture: **if the ruler's own zero mark wanders, no reading from it can be trusted.** The determinism of v2 is not a marginal engineering nicety; it is the precondition for the score being a measurement at all.

### 4.2 Experiment 3 — G2P correctness on held-out Standard Korean, with a declared scope

**Question.** The rule engine was developed against ~30 textbook examples (now unit tests). Does it generalize to unseen items — and does it know the boundary of its own competence?

**Method.** A 73-item evaluation set with gold labels drawn from the 표준국어대사전 pronunciation fields, partitioned into three buckets: 51 items exercising the context-free rule pipeline (disjoint from the development examples), 17 items exercising the morphology-conditioned layer, and 5 items that are **out of scope by design** — 사잇소리 tensification in native compounds, which requires semantic knowledge of whether a compound is of the relevant type. The core and morphology buckets run in continuous integration as a regression gate. Data: `experiments/data/g2p_heldout.tsv`.

**Results (Table 4).**

**Table 4. Held-out G2P accuracy.**

| Bucket | Accuracy | Interpretation |
|---|---|---|
| Core (context-free rules) | **51/51 = 100%** | Generalizes beyond dev set |
| Morphology-conditioned | **17/17 = 100%** | POS-disambiguated boundary rules hold |
| Out-of-scope (semantic 사잇소리) | 0/5 = 0% (expected) | 강가 → predicted [강가], gold [강까] |

**Interpretation.** Within its declared scope the engine is exact on held-out data, including the categories (ㄴ-insertion, in-context POS-disambiguated stem tensification, cluster resolution, liaison blocking, Sino-Korean ㄴ+ㄹ) that were *out-of-scope failures before the morphology layer existed*. The out-of-scope bucket is the methodological point: the five 사잇소리 items (강가→[강까], 밤길→[밤낄], 술잔→[술짠], …) fail because distinguishing a 사잇소리 compound from a non-compound requires semantics not present in POS tags. Reporting them as a *named, bounded* failure — rather than letting the engine guess and quietly err — is what makes the 100% figure meaningful. A scorer that is honest about where it stops is more trustworthy than one that pretends to cover everything.

**Note on library comparison.** A head-to-head with `g2pK` was attempted but the library could not be installed on Windows/Python 3.13 (its `python-mecab-ko` dependency fails to build). A Linux-CI comparison is future work.

---

## 5. Validating the Instrument II: Construct Validity on Controlled Perturbations

Having established that the instrument is reproducible (§4.1) and correct within scope (§4.2), we test whether it *responds to pronunciation error in the right direction and by the right amount* — before introducing the confounds of real recordings. Both experiments here use neural text-to-speech (TTS) to synthesize audio with controlled, known errors, which cleanly isolates scorer behavior from recording-quality and speaker-variability nuisances.

### 5.1 Experiment 2 — Discriminant validity

**Question.** Does the pipeline actually assign lower scores to speech containing typical Japanese-L1 errors?

**Method.** For 10 target sentences, two clips were synthesized with the *same* neural TTS voice: the correct sentence, and an error-injected version encoding attested L1 patterns (epenthesis, ㅓ→ㅗ, coda repair, laryngeal loss). Both ran through the full pipeline (ffmpeg → Wav2Vec2 → G2P → jamo alignment). Script: `exp2_error_discrimination.py`.

**Results.** Pairwise ranking accuracy (correct > error) was **8/10 (80%)**; mean correct-audio score 85.8, mean error-audio score 75.5, mean gap 10.3 points.

**Failure analysis — the informative part.** The two misrankings are not noise; they are the **ASR confound in miniature**, and they preview §6.3:

- **괜찮아요** (misranked): Wav2Vec2 transcribed even the *correct* native TTS clip as 근로아 (score 25). The acoustics channel cannot distinguish its own recognition error from a pronunciation error.
- **맛있어요 → 마시소요** (tie): the recognizer's implicit lexical bias "auto-corrected" the injected perturbation back to the correct sentence, so the error clip scored 100.

These are precisely the two ways an off-the-shelf, non-adapted acoustic model corrupts a production-oriented score: it hallucinates errors on clean audio, and it absorbs real errors near high-frequency words. Naming them here motivates measuring their aggregate size on real data later.

### 5.2 Experiment 4 — Graded-severity monotonicity

**Question.** Without human ratings, can we test whether the score tracks error *severity*, not just error presence?

**Method.** For 5 base sentences, four clips were synthesized at severity 0–3, where severity *k* applies the first *k* cumulative error injections (all attested Japanese-L1 patterns). The injected error count is a controlled ground-truth ordinal; a valid scorer must decrease monotonically. 20 clips, full pipeline. Script: `exp4_severity_monotonicity.py`.

**Results.** Spearman ρ (severity vs. score) = **−0.702**, 95% bootstrap CI **[−0.928, −0.325]**; mean score by severity 0→3 was 91.8 → 88.2 → 78.0 → 74.8 (strictly decreasing); 13/15 (87%) of steps were monotone; 4/5 ladders were perfectly monotone.

**Interpretation.** The confidence interval excludes zero: the score is a statistically significant **monotone function of controlled error severity**. The single non-monotone ladder (비빔밥을 먹었어요) again traces to ASR misrecognition of the clean severity-0 clip, and one ladder plateaued when the recognizer absorbed later perturbations — the same confound. Together, §5.1 and §5.2 establish **ordinal sensitivity**: the instrument moves in the right direction and, on average, by an amount that tracks severity. Absolute calibration against human judgment is the remaining question, taken up next on real speech.

---

## 6. Validating the Instrument III: Real Japanese-Accented Speech

Everything to this point used either fixed inputs or synthetic audio. The decisive question is whether the score carries signal on **genuine** L2 speech, and how much of that signal survives the ASR confound previewed in §5.

### 6.1 Corpus and human signals

We use **AI-Hub dataset 131** (외국인 한국어 발화 음성 데이터, *Foreign-speaker Korean Speech*), Japanese-L1 subset. Its scale is substantial (Table 5), and — critically — each read-aloud utterance carries **three human signals** we can validate against:

- `Reading` — the script the learner was asked to read (the target).
- `ReadingLabelText` — **what native transcribers actually heard**, including misreadings, repeats, and particle swaps. This is not a cleaned copy of the script: ~21% of read-aloud clips deviate from it. This gives us a per-utterance human record of *perceived* deviation, at corpus scale, without recruiting raters.
- `SentenceSpeechLV` — a per-utterance speech-level rating (상/중/하, "high/mid/low").

**Table 5. AI-Hub 131 Japanese-L1 subset (verified locally by CRC over every archive member).**

| Split | Utterances | Read-aloud (with script) | Speakers | Hours |
|---|---|---|---|---|
| Training | 173,280 | 131,445 | 255 | 607.2 |
| Validation | 21,663 | 16,435 | 252 | ~76 |

**Sample.** From the Validation split we drew a **stratified sample of 615 clips from 198 speakers**: all 215 하-rated read-aloud clips (the scarce stratum, taken whole) plus 200 each of 상 and 중, with a fixed random seed for reproducibility. Each clip was scored by the full acoustics pipeline (Wav2Vec2-CTC → G2P → jamo alignment). The corpus license forbids redistribution, so extracted audio and label text remain local; only aggregate results are committed (`experiments/results/exp6_l2_validation.json`).

We define two derived quantities per clip: **System score** (the pipeline's 0–100 score of the script against the acoustics transcription) and **Heard score** (the pipeline's score of the script against `ReadingLabelText` — i.e., how far, at the jamo level, the *human transcription* departed from the script). A clip is "deviating" when its Heard score < 100.

### 6.2 Experiment 6 — Correlation with human signals

**Two validation axes** are available. Axis A asks whether the System score agrees with the deviation native transcribers *heard*; axis B asks whether it tracks the *proficiency* rating. Results in Table 6.

**Table 6. System score vs. human signals (n = 615, 198 speakers).**

| Axis | Metric | Value |
|---|---|---|
| **A. Agreement with transcribers** | Spearman ρ, System vs. Heard deviation | **0.322** [95% CI 0.246, 0.391] |
| | AUC detecting transcriber-noted deviations | **0.717** [0.665, 0.764] |
| **B. Proficiency association** | Spearman ρ, System vs. speech-level rating | **0.473** [0.409, 0.535] |
| | AUC 상 vs. 하 / 상 vs. 중 / 중 vs. 하 | **0.818** / 0.637 / 0.720 |
| | Speaker-level ρ vs. TOPIK grade (n=55 speakers) | 0.317 |
| | Mean System score by level 상/중/하 | 83.1 / 79.2 / 71.8 (monotone) |

**Interpretation.** On genuine accented speech the score separates the 상 and 하 speech-level ratings at AUC 0.818 and decreases monotonically across levels — the **first evidence on real L2 audio**, not synthetic perturbations. Axis-A agreement (ρ 0.322; deviation-detection AUC 0.717) is respectable given that Heard scores are ceiling-bound (median 100 at every level), which mechanically attenuates the correlation.

**These are conservative lower bounds, for two structural reasons.** (i) `SentenceSpeechLV` rates *overall speech level* — fluency included — not pronunciation alone, so it is a noisy proxy for what the score measures. (ii) The speaker pool skews proficient (TOPIK 5–6 dominant), so **range restriction** deflates every correlation relative to a balanced L2 population. Both effects push the observed numbers *down*; the true pronunciation-only, full-range association is plausibly higher.

### 6.3 The ASR noise floor — measuring the confound

The confound previewed in §5.1–5.2 can now be *quantified* rather than merely acknowledged. Restrict attention to the **483 clips that transcribers heard as faithful** (Heard score = 100). For these, the reading was correct by human judgment; any score below 100 is the acoustics channel disagreeing with a faithful reading — i.e., the ASR noise floor.

**Result: mean System-vs-heard score 79.6, median 81, 10th percentile 68.**

This is the honest headline of the study. It means: **with the current off-the-shelf acoustic model, an absolute learner-facing score ("your pronunciation is 80/100") is not defensible**, because a faithful reading already scores ~80. What *is* defensible is **ranking** and **within-learner progress deltas**, where the floor is a roughly common offset.

But the floor is **not merely noise** — and this is the study's most interesting finding. The 하-rated speakers also read faithfully (mean Heard score 98.8, vs. 99.7 for 상): the transcription channel is ceiling-limited and does *not* separate the proficiency levels. Yet the System score *does* separate them (§6.2). The only way both facts hold is if the acoustics channel is responding **systematically to accent strength** even when the reading is orthographically faithful. In other words, part of the "noise floor" is exactly the signal a pronunciation scorer needs — sub-lexical accent that a human orthographic transcription cannot capture. The confound and the signal are entangled in the same quantity, and adapting the acoustic model to the accent is the way to disentangle them.

### 6.4 Experiment 6b — Three-way comparison against a GOP baseline

The field's classic scorer is GOP. Does our decode-then-align approach actually *beat* it, or merely differ?

**Method.** A CTC variant of GOP was computed from the *same* Wav2Vec2 model on the *same* 615 clips:

$$\text{GOP} = \frac{\ell_{\text{forced}} - \ell_{\text{free}}}{n_{\text{frames}}}$$

where $\ell_{\text{forced}}$ is the CTC log-likelihood of the target transcript (summed over all valid alignments, via the CTC loss) and $\ell_{\text{free}}$ is the unconstrained greedy-path log-likelihood — the standard denominator that cancels audio-quality and duration effects. The CTC target is the **orthographic** script, not the G2P surface form, because the acoustic model was trained on orthographic transcripts and its 1,204-syllable vocabulary covers orthography far better (0.27% out-of-vocabulary tokens vs. 2.4% for surface spellings); out-of-vocabulary target tokens are dropped, touching 77/615 clips. Script: `exp6b_gop_baseline.py`.

**Results (Table 7).**

**Table 7. Alignment-based score vs. CTC-GOP, same model and sample.**

| Metric | System (jamo alignment) | GOP (CTC likelihood ratio) |
|---|---|---|
| Spearman ρ vs. transcriber-heard deviation | **0.322** [0.246, 0.391] | 0.197 [0.120, 0.266] |
| Deviation-detection AUC | **0.717** [0.665, 0.764] | 0.635 [0.582, 0.683] |
| Spearman ρ vs. speech-level rating | **0.473** [0.409, 0.535] | 0.153 [0.073, 0.228] |
| AUC 상 vs. 하 | **0.818** | 0.606 |
| Speaker-level ρ vs. TOPIK | **0.317** | 0.144 |

Scorer agreement: ρ(System, GOP) = **0.666** — correlated but far from redundant.

**Interpretation.** The alignment-based score **dominates GOP on every axis**, and by the largest margin exactly where it matters for a CAPT product — 상/하 discrimination (0.818 vs. 0.606). A plausible mechanism: the likelihood ratio absorbs *everything* that makes the audio unlike the model's training distribution — channel, rate, global accent — whereas decoding-then-aligning **quantizes** frame-level uncertainty into a discrete hypothesis before comparison, discarding nuisance variance while preserving segmental errors. This is independent, real-data support for the project's core architectural bet (score text-level similarity over ASR output rather than raw model confidences), consistent with the morpheme-similarity findings of the Korean pronunciation-assessment literature (§2.1).

**Honest scoping of the baseline.** This is an *utterance-level* GOP over a *non-adapted* model — a fair **same-budget** baseline, not the strongest possible GOP. A phone-level GOP over a fine-tuned, phoneme-output model would be stronger, and is exactly what the fine-tuning roadmap (§7.3) enables; the re-match is planned, not claimed. Both scorers also share one acoustic model, so their errors are not independent, and the comparison is *relative* under a shared confound rather than absolute.

---

## 7. Discussion

### 7.1 What has actually been shown

The claim of this work is deliberately modest and, we argue, unusually well-supported: **the score is a reproducible, scope-bounded, ordinally-valid measurement of pronunciation for Japanese-L1 Korean speech, whose dominant failure mode has been measured rather than hidden.** Each clause is backed by a specific experiment — reproducible (§4.1), scope-bounded (§4.2), ordinally valid on synthetic (§5) and real (§6.2, §6.4) speech, failure measured (§6.3). What is *not* claimed is absolute calibration against pronunciation-specific human ratings; that is the one square still open, and §7.3 states precisely how to close it.

### 7.2 The noise floor reframed: a limitation that is a research program

The instinct in a systems paper is to bury a number like "a faithful reading scores 79.6." We foreground it, because it is the most scientifically productive object the study produced. It converts a vague roadmap item ("the ASR confound") into a **falsifiable target with a number attached**, and it comes with a design implication that is immediately actionable: *ship ranking and progress, not absolute scores, until the floor is raised.* For a graduate research agenda, a well-measured limitation is worth more than an unmeasured success — it is a thesis-shaped problem that the author has already instrumented.

### 7.3 The path to closing the gap

Three concrete steps, in dependency order:

1. **A pronunciation-specific human-rater study.** A pre-registered protocol exists (anchored 1–5 intelligibility rubric, ≥3 native raters, blinding, pre-registered threshold ρ ≥ 0.6) with a self-tested analysis harness. Experiment 6 covered the *scale* axis with corpus labels that conflate fluency; this study covers the *precision* axis with pronunciation-only ratings. The two are complementary, not redundant.
2. **Accent-aware acoustic-model adaptation.** Fine-tune Wav2Vec2 on the AI-Hub Japanese-L1 **Training** split (131k utterances, 255 speakers, 607 h, verified intact). The success criterion is now **concrete and pre-committed, thanks to §6.3**: raise the faithful-reading noise floor (79.6) toward the high 90s while maintaining or improving the 상/하 AUC (0.818).
3. **A stronger GOP rematch.** After adaptation, re-run the §6.4 three-way comparison including *phone-level* GOP over the fine-tuned phoneme-output model — the fair strong baseline.

This is a natural three-paper arc: (I) the system + reproducibility + G2P benchmark + the real-data validation of this draft; (II) the human-correlation study with the GOP comparison; (III) accent-robust acoustic modeling and formal MDD, whose *problem statement is the noise floor this work measured*.

### 7.4 Threats to validity (consolidated)

- **Proxy labels.** `SentenceSpeechLV` conflates fluency with pronunciation; `ReadingLabelText` is orthographic and ceiling-bound, so it validates *deviation detection*, not fine phonetic scoring. Both attenuate §6 correlations downward.
- **Range restriction.** TOPIK 5–6 dominance deflates all real-speech correlations relative to a balanced population.
- **Shared acoustic model.** The System/GOP comparison (§6.4) is relative under a common confound; neither is absolutely calibrated.
- **Single language pair, read speech.** Findings are for Japanese-L1, read-aloud Korean; spontaneous speech and other L1s are out of scope by design (the specificity is intentional, but it bounds generalization).
- **TTS prosody (§5).** Synthetic clips carry native prosody, so §5 is a controlled sensitivity study, not an L2 evaluation — which is why §6 exists.

### 7.5 Why the Japanese↔Korean specificity is a feature, not a limitation

A reviewer might ask why we do not target a generic L2-Korean scorer. The answer is that **specificity is where both the science and the value are.** The error taxonomy (§3.5) is only writable because the contrastive phonology of *this* pair is documented; a generic system would have to rediscover it from data. The validation is only sharp because the expected errors are known in advance. And the practical need is real and under-served: Japanese-L1 learners are the second-largest population in Korean-education research, yet self-study audio materials are documented as a persistent gap. The same methodology would transfer to another pair, but the *content* — the rules, the tags, the interpretation — is inherently pair-specific, and owning that pair is the contribution.

---

## 8. Conclusion

We argued that a pronunciation score is a **measurement instrument**, and that the systems literature too often trusts the instrument without validating it — a habit the LLM turn has made both more tempting and more dangerous. We built a pronunciation-assessment pipeline for Japanese-L1 Korean that separates a deterministic, unit-tested measurement from an LLM interpretation layer, and we validated the instrument along a ladder from fixed inputs to real accented speech: reproducible where an LLM scorer is not (SD 0.0 vs. 1.45), correct within a declared scope (100% held-out, with out-of-scope cases named), ordinally valid on controlled perturbations (ρ = −0.702 with severity), and — on 615 clips of genuine Japanese-accented Korean — correlated with two independent human signals (speech-level AUC 0.818) while dominating a same-budget GOP baseline on every axis. Above all, we **measured the system's own dominant failure mode**: a faithful reading scores 79.6, so absolute scores await accent adaptation while rankings are already usable. The contribution is a defensible procedure for earning trust in a score, instantiated for a specific and under-served language pair, with the limitation quantified precisely enough to become the next experiment.

---

## References

*(Draft bibliography; full details in `docs/REFERENCES.md`. Korean-venue entries are cited by title and venue where author/year could be verified from source metadata; the repository's reference file is the authoritative list.)*

**Pronunciation assessment methodology**

- Witt, S. M., & Young, S. J. (2000). Phone-level pronunciation scoring and assessment for interactive language learning. *Speech Communication*, 30(2–3), 95–108. [Goodness of Pronunciation.]
- 형태소 분석 기반 외국인 발화 한국어 발음평가 개선 방법. DBpia. [Morpheme-level text-similarity scoring outperforms off-the-shelf APIs on L2 Korean — support for the text-similarity lineage.]
- 김선정 (2022). 언어 유형을 활용한 한국어 종성 발음 교육 방안. *언어와 문화*.

**Japanese-L1 Korean contrastive phonology (grounds the error taxonomy)**

- 장향실 (2016). 중국어와 일본어 모어 화자의 한국어 음절 종성 산출 차이 연구. *우리어문연구*. [Grounds `vowel_epenthesis`.]
- 하호빈·이화진 (2019). 음절 연쇄에서 나타나는 일본인 학습자의 한국어 종성 발음 유형. *언어사실과 관점*. [Grounds `vowel_epenthesis`, `coda_deletion`.]
- 이화진 (2021). 일본인 학습자의 한국어 발음 오류에 대한 종적 연구 — 자연 발화 데이터 분석을 중심으로. KCI. [Longitudinal persistence of epenthesis/coda errors.]
- 이화진 (2018). 한국어 비음화의 오류 유형과 원인 분석 — 중국인 학습자와 일본인 학습자를 중심으로. *언어사실과 관점*. [Grounds `nasal_coda_confusion`.]
- Native language interference in producing the Korean rhythmic structure: Focusing on Japanese. *Phonetics and Speech Sciences* (말소리와 음성과학) 10(4). [Mora-timed rhythm transfer; %V, VarcoV, VarcoS.]
- 모음 체계와 자질에 의한 일본인 학습자의 한국어 모음 발음 분석. KCI. [Grounds `vowel_ʌ_o_confusion`, `vowel_ɯ_u_confusion`.]
- 森香奈 (2008). 일본인의 영어와 한국어 발음의 오류 분석. 한국언어학회 학술대회지. [Cross-target isolation of L1-driven error sources.]

**Korean-language education for Japanese learners (context)**

- 일본인 학습자 대상 한국어교육 관련 연구 최근 동향 분석 (2008–2014). *한국어교육* 26(1), 2015. [Japanese learners = 2nd-largest population; 258-study survey.]
- 일본에서의 한국어 듣기 교재 분석 연구. KCI. [Documents the self-study audio-materials gap.]

**Corpora and resources (fine-tuning roadmap)**

- AI-Hub. 인공지능 학습용 외국인 한국어 발화 음성 데이터 (Foreign-speaker Korean Speech; Japanese-L1 subset). [Primary source for accent-aware fine-tuning; §6 evaluation corpus.]
- 한일 병렬 코퍼스 구축의 실제와 문제점. KCI.

**Standard references (to be completed in a submission version)**

- Radford, A., et al. (2023). Robust speech recognition via large-scale weak supervision (Whisper). *ICML*.
- Baevski, A., et al. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. *NeurIPS*.
- Graves, A., et al. (2006). Connectionist temporal classification. *ICML*.
- 국립국어원. 표준발음법 (Standard Korean Pronunciation), 한국어 어문 규범.

---

## Appendix A. Reproducibility summary

| Exp | Script | Result file | Headline |
|---|---|---|---|
| 1 | `exp1_reproducibility.py` | `exp1_reproducibility.json` | LLM IPA SD 1.45 (4 IPA variants) vs. deterministic SD 0.0 |
| 2 | `exp2_error_discrimination.py` | `exp2_error_discrimination.json` | 80% pairwise ranking; gap 10.3 |
| 3 | `exp3_g2p_heldout.py` | `exp3_g2p_heldout.json` | 100% (68/68) in scope; 0/5 out of scope (named) |
| 4 | `exp4_severity_monotonicity.py` | `exp4_severity_monotonicity.json` | Spearman ρ = −0.702 [−0.928, −0.325] |
| 6 | `exp6_l2_validation.py` | `exp6_l2_validation.json` | 상/하 AUC 0.818; ρ 0.473; noise floor 79.6 |
| 6b | `exp6b_gop_baseline.py` | `exp6b_gop_threeway.json` | System > GOP on every axis (0.818 vs. 0.606) |

Statistical utilities (Pearson r, Spearman ρ with tie-averaged ranks, percentile bootstrap CI, Mann-Whitney AUC) are implemented dependency-free in `experiments/statsutil.py` and unit-tested, so the analysis reproduces without SciPy. Figures: `experiments/plot_exp6.py` → `docs/assets/exp6_score_by_level.png`, `docs/assets/exp6_roc.png`. Corpus integrity is verified by `tools/verify_aihub131.py` (CRC over every archive member). Experiment 5 (pronunciation-specific human ratings) is designed and harness-ready but not yet executed; it is the one result reported as *planned*.
