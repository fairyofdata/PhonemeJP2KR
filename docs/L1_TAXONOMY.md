# L1 Error Taxonomy — Coverage Audit

The error classifier in [`src/scoring.py`](../src/scoring.py) tags known
Japanese-L1 interference patterns. This document maps it against the most
complete pedagogical inventory of those patterns for Japanese learners of
Korean, and states — per category — whether a jamo-alignment scorer over
ASR output **can** detect it, and why not when it cannot.

## Reference inventory

**Lee, Yuna (이유나), 2022.** *일본인 학습자를 위한 한국어 발음 학습용 모바일
애플리케이션 설계 연구* [A study on designing a mobile application for Korean
pronunciation learning for Japanese learners]. M.A. thesis, Kyonggi
University Graduate School. [DBpia T16143963](https://www.dbpia.co.kr/journal/detail?nodeId=T16143963)

From a review of the contrastive literature, Lee derives eleven frequent
error categories and orders a curriculum around them (monophthongs →
diphthongs → onsets → syllable structure → codas → phonological rules →
intonation). The thesis then designs — but, by its own account, does not
implement or evaluate — an ASR-based mobile app around that curriculum.

Independent corpus evidence comes from **Yeo et al. (ICPhS 2023)**
([arXiv:2306.10821](https://arxiv.org/abs/2306.10821)), who aligned
canonical phone sequences with a fine-tuned Wav2Vec2 XLS-R *phone*
recognizer on the same AI-Hub L2 corpus used in Experiment 6. Their
cross-L1 common errors include plain-for-aspirated/tense substitution,
coda deletion, and diphthong monophthongization; /ɯ/ insertion is among
the patterns they found specific to Japanese L1.

## Coverage matrix

| # | Category (Lee 2022) | Tag | Status | Notes |
|---|---|---|---|---|
| 1 | ㅓ/ㅗ confusion | `vowel_ʌ_o_confusion` | ✅ detected | |
| 2 | ㅜ/ㅡ confusion | `vowel_ɯ_u_confusion` | ✅ detected | |
| 3 | ㅕ/ㅛ confusion | `vowel_jʌ_jo_confusion` | ✅ detected | ʌ/o merger carried over to the j-onglide |
| 4 | ㅢ confusion | `diphthong_ɰi_monophthongization` | ✅ detected (word-initial) | Needed a G2P fix first: 표준발음법 §5 (희망 → [히망], non-initial 의 → [이]) was missing, so native-like [히망] used to be penalized. Only word-initial 의, where [으]/[이] is a real error, is tagged. |
| 5 | Lenis/aspirated/tense confusion | `laryngeal_confusion` | ✅ detected | Yeo et al.: most common cross-L1 error |
| 6 | Voiced/voiceless confusion | — | ❌ structurally invisible | Allophonic (VOT, intervocalic voicing): never changes the Hangul transcript. Needs an acoustic measure. |
| 7 | Open syllabification (CV repair) | `vowel_epenthesis`, `coda_deletion` | ✅ detected | Yeo et al.: /ɯ/ insertion specific to Japanese L1 |
| 8 | Nasal coda confusion (ㄴ/ㅁ/ㅇ) | `nasal_coda_confusion` | ✅ detected | Extended from ㄴ/ㅇ to include ㅁ; now position-aware (onset ㄴ/ㅁ swaps are not tagged) |
| 9 | Stop coda confusion (ㄱ/ㄷ/ㅂ) | `stop_coda_confusion` | ✅ detected | Unreleased codas; position-aware |
| 10 | Rule application (aspiration, tensification, nasalization) | — | ❌ structurally invisible *with the current ASR* | See below |
| 11 | Yes/no vs wh-question intonation | — | ❌ structurally invisible | Both ASR channels discard F0. Needs a pitch-track channel. |

**8 of 11 categories are detected; the three that are not share one root
cause:** the acoustic channel is a character-level CTC model trained on
*orthographic* transcripts (see `exp6b_gop_baseline.py`). A learner who
says [합니다] without nasalization is still transcribed as `합니다`, which
the G2P then maps to [함니다] — the error disappears before scoring. The
same holds for any system that scores a spelling-level transcript.

## Implications for the roadmap

- **Rule-application errors (#10)** become visible only with a recognizer
  whose output tracks pronunciation, not spelling — a phone recognizer as
  in Yeo et al., or a Wav2Vec2 fine-tuned on *pronunciation-faithful*
  labels. Fine-tuning on AI-Hub `ReadingLabelText` alone will not do it:
  those labels are orthographic too.
- **Intonation (#11)** is a separate, cheap channel: F0 contour of the
  final syllable (rise for yes/no questions, fall for wh-questions) from
  `librosa.pyin`, independent of ASR.
- **Voicing (#6)** is the lowest priority: VOT measurement on read speech
  is feasible, but its pedagogical payoff for intelligibility is small
  relative to the laryngeal triad.

## Which categories matter empirically?

Lee's inventory, like this classifier, comes from the literature. Whether
each category actually separates proficiency levels in real speech — as
opposed to firing uniformly because of ASR noise — is measured by
[Experiment 7](../experiments/exp7_error_profile.py) on the Experiment 6
sample: per-tag rate by 상/중/하 stratum, a 하-vs-상 AUC, and a noise
baseline from faithful readings.
