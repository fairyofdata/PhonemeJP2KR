# Phomene Project Handover & Audit Request (for Claude Fable 5)

이 문서는 Fable 5가 잠시 프로젝트를 떠나있던 사이(사용량 한도 도달 등), 임시로 투입된 에이전트(Antigravity)가 수행한 작업 내역을 인수인계하는 동시에, **Fable 5의 압도적인 성능을 활용한 제로베이스(Zero-based) 전면 재검토 및 코드 오딧(Audit)을 요청**하기 위한 문서입니다.

---

## 0. 타임라인 요약 (Context for Fable 5)
1. **당신(Fable 5)의 마지막 작업**: 결정론적 G2P 아키텍처 개편 및 엄밀한 실험적 검증 체계(`exp1~exp5`), 통계 유틸리티(`statsutil.py`), `HUMAN_EVAL_PROTOCOL.md` 구축 완료 후 대기.
2. **임시 대타(Antigravity)의 작업**: 당신이 자리를 비운 사이, 사용자의 요청으로 기획서(압축된 HTML 문서 등)를 파싱하여 학술적 근거를 리드미에 추가하고, 기존 코드에 형태소 분석기(`Kiwipiepy`)를 연동하는 작업을 진행함.
3. **현재 당신의 미션**: 임시 대타가 작성한 코드와 문서를 **백지상태에서 완전히 새로 작업한다는 마음가짐으로 혹독하게 검증(Audit)**하고 개선하는 것.

---

## 1. 재검토 및 오딧(Audit) 요청 사항

### 1.1. 기획서 기반 학술적 근거(README) 재검토
Antigravity는 사용자가 제공한 압축된 기획서(Notion HTML 등)와 논문 초록들을 바탕으로 `README.md`, `README_kr.md`, `README_jp.md`에 학술적 근거를 추가했습니다.
- **검토 대상 파일**: 
  - `extracted_docs/inner_extracted/` 내의 기획서 원본 HTML 및 `abstracts.json` 등
  - `README.md`, `README_kr.md`, `README_jp.md`
  - `docs/REFERENCES.md`
- **Fable에게 바라는 점**: 원본 기획서 데이터(`extracted_docs` 내부)를 직접 다시 읽어보십시오. Antigravity가 문서를 업데이트하면서 기획서의 중요한 언어학적/음운론적 통찰이나 레퍼런스를 누락하지 않았는지, 혹은 맥락을 잘못 짚어 작성한 부분이 없는지 냉정하게 평가하고 필요하다면 README를 전면 수정하십시오.

### 1.2. Kiwipiepy 기반 형태소 처리 로직 전면 오딧
Antigravity는 G2P 엔진(`src/g2p.py`)에 `Kiwipiepy`를 결합하여 '어간/어미 경음화', 'ㄴ-첨가', '연음 예외'를 처리하도록 `_preprocess_morphology` 후크를 추가했습니다.
- **검토 대상 파일**: 
  - `src/g2p.py`
  - `tests/test_g2p.py`
  - `requirements.txt`, `requirements-dev.txt`
- **Fable에게 바라는 점**: Antigravity가 임시로 추가한 코드가 당신이 설계한 우아한 아키텍처의 철학을 훼손하고 있지 않은지 확인하십시오. 특히 `꽃잎`, `물약` 등 `Kiwipiepy`가 단일 형태소로 묶어버리는 빈출 합성어에 대해 Antigravity는 글자를 순회하며 하드코딩된 패치(`잎`->`닢`)를 적용하는 절차적인 방식을 택했습니다. 이 로직이 최선인지 당신의 능력을 발휘해 검토하고, 더 우아한(Pythonic) 방법이나 정규표현식, 혹은 사용자 사전(`add_user_word`) 등록 방식으로 리팩토링할 수 있다면 몽땅 뜯어고치십시오. (단, 기존에 구축된 44개의 테스트 코드는 모두 통과해야 합니다.)

---

## 2. 모든 검증이 끝난 후의 로드맵 (Next Steps)

Antigravity가 남긴 흔적을 모두 당신의 수준으로 완벽하게 끌어올렸다면, 다음 단계로 넘어갑니다.

### 2.1. Wav2Vec2-CTC 기반 Forced Alignment
- **`docs/implementation_plan_alignment.md`** (CTC Forced Alignment 기획 및 G2P 통합 설계도)를 반드시 읽으십시오.
- **기능 목표**: 외국인 학습자의 음성(Wav)과 정답 스크립트를 입력받아, 정답 발음 기호(Phonemes/Jamo) 배열을 G2P로 추출한 뒤, Wav2Vec2 한국어 모델(예: `kresnik/wav2vec2-large-xlsr-korean`)의 CTC Logits와 강제 정렬(Forced Alignment)하여 **음소별 발음 타임스탬프**를 반환하는 파이프라인을 구축하십시오.
- 이는 최종적으로 프론트엔드에서 "학습자가 틀리게 발음한 구간만 오디오로 다시 들려주기" 기능을 구현하기 위한 핵심 백엔드 기능이 될 것입니다.
