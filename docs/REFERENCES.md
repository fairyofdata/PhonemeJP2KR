# Academic References & Bibliography

Studies grounding Phomene's dual-ASR architecture and the rule-based L1
interference classifier. Bibliographic details are listed only where they
could be verified from the source metadata; entries without a verifiable
author/year are cited by title and venue link alone. Grouped by the role
they play in the project.

---

## 1. L1 interference phonology — grounds the error taxonomy (`src/scoring.py`)

### [중국어와 일본어 모어 화자의 한국어 음절 종성 산출 차이 연구](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10761462)
장향실 · 우리어문연구 · 2016 — grounds `vowel_epenthesis`
> Japanese learners resyllabify Korean CVC into CV.CV by unconsciously mapping the coda onto Japanese 촉음 /Q/ or 발음(撥音) /N/ (CV-Q.CV / CV-N.CV), whereas Chinese learners tend to delete the coda outright. The error source is the learner's L1 syllable structure, not the target language's.

### [음절 연쇄에서 나타나는 일본인 학습자의 한국어 종성 발음 유형](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09235049)
하호빈·이화진 · 언어사실과 관점 · 2019 — grounds `vowel_epenthesis`, `coda_deletion`
> Elementary-level Japanese learners' coda errors fall into three types: ill-formed place assimilation, open syllabification, and sonority changes of obstruents — driven by the underlying /Q/ and /N/ templates and the feature asymmetry between Japanese and Korean syllable-final nasals.

### [일본인 학습자의 한국어 발음 오류에 대한 종적 연구 ─ 자연 발화 데이터 분석을 중심으로](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002730504)
이화진 · 2021 — longitudinal evidence that epenthesis/coda errors persist
> Longitudinal spontaneous-speech study; keywords include open syllabification, final-consonant deletion, and velar insertion — the persistent error classes the classifier targets.

### [한국어 비음화의 오류 유형과 원인 분석 - 중국인 학습자와 일본인 학습자를 중심으로](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE08838909)
이화진 · 언어사실과 관점 · 2018 — grounds `nasal_coda_confusion`
> Where nasalization is required, Japanese learners redundantly copy the place of articulation from the following consonant (redundant place assimilation), blocking correct application of Korean nasalization in manner terms.

### [Native language interference in producing the Korean rhythmic structure: Focusing on Japanese](https://www.eksss.org/archive/view_article?pid=pss-10-4-45)
말소리와 음성과학 (Phonetics and Speech Sciences) 10(4) — grounds the mora-timing framing
> Acoustic comparison (%V, VarcoV, VarcoS) of 4 native Koreans and 10 advanced Japanese learners: learners' rhythm differs significantly in %V and VarcoV, with the largest VarcoS deviation in VC/CVC syllables whose coda is a nasal — direct acoustic evidence of mora-timed L1 rhythm transfer.

### [모음 체계와 자질에 의한 일본인 학습자의 한국어 모음 발음 분석](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002158469)
KCI — grounds `vowel_ʌ_o_confusion`, `vowel_ɯ_u_confusion`
> Analyzes Japanese learners' Korean vowel errors through vowel-system and feature differences (roundness opposition, glide distribution restrictions, vowel coalescence).

### [일본인 한국어 학습자의 유음 습득에 있어서의 음향음성학적 특징에 관한 연구](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002606636)
KCI — liquid (ㄹ) acquisition; candidate future error tag
> Acoustic study of liquid acquisition by Japanese learners under the contrastive analysis hypothesis (preclusive/intrusive interference).

### [일본인의 영어와 한국어 발음의 오류 분석](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE02269367)
森香奈 · 한국언어학회 학술대회지 · 2008
> Cross-target-language view: errors Japanese speakers share when pronouncing English and Korean, isolating L1-driven (rather than target-specific) error sources.

## 2. Pronunciation assessment methodology

### [형태소 분석기반 외국인 발화 한국어 발음평가 개선 방법](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11438586)
DBpia — independent support for text-similarity scoring over ASR output
> Off-the-shelf pronunciation-scoring APIs poorly reflect L2 Korean speech. Scoring by morpheme-level similarity between the reference sentence and the ASR transcript (with distance metrics over a morpheme lexicon) tracks what native Korean listeners actually understood better than existing scoring models — the same design decision Phomene makes at the jamo level.

### [언어 유형을 활용한 한국어 종성 발음 교육 방안](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11280986)
김선정 · 언어와 문화 · 2022
> Typology-informed pedagogy for teaching Korean coda pronunciation — informs how feedback should be phrased per L1 type.

## 3. Korean-language education for Japanese learners (market & pedagogy context)

### [일본인 학습자 대상 한국어교육 관련 연구 최근 동향 분석 (2008–2014)](https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART001970757)
한국어교육 26(1) · 2015
> Survey of 258 studies: Japanese learners are the second-largest population in Korean-language education research; identifies under-served research areas.

### [일본에서의 한국어 듣기 교재 분석 연구](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001250398)
KCI
> Analysis of listening textbooks published in Japan (discourse types, listening activities) — documents the gap that self-study materials leave in audio training.

### [일본인을 위한 한국어 교재 개발과 교수 방법](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE02349434)
간노 · 교육한글 · 1991

### [일본에서의 한국어 교재 개발의 문제점 및 해결 방안](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11037540)
후지이시 다카요 · 국어교육연구 · 2000

### [한일 대조언어학적 관점에서 본 한국어 문법학습의 과제와 코퍼스의 활용 가능성](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001733081)
KCI

## 4. Corpora & data resources (fine-tuning roadmap)

### [AI-Hub 인공지능 학습용 외국인 한국어 발화 음성 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=505)
Includes Japanese-L1 speakers — the primary candidate for fine-tuning Wav2Vec2 on Japanese-accented Korean (see README roadmap).

### [한일 병렬 코퍼스 구축의 실제와 문제점](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART000999318)
KCI — parallel-corpus construction practice for the JP↔KR pair.

> **Note (direction matters):** NINJAL's C-JAS corpus is *Japanese-language* speech produced by Chinese/Korean L1 learners — the reverse direction of this project. It is intentionally **not** listed as a fine-tuning source here; it would only be relevant to a future KR→JP sibling service.
