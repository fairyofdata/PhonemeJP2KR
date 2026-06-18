[🇺🇸 English](Readme.md) | [🇯🇵 日本語 (Japanese)](README_jp.md)

# **🗣️ 일본인 학습자를 위한 LLM 기반 한국어 발음 교정 서비스** 🧑‍🏫📝

[![영상 제목](https://img.youtube.com/vi/4SwwmzEcpZQ/0.jpg)](https://youtu.be/4SwwmzEcpZQ)<br>

> 일본인 한국어 학습자를 위한 LLM 기반 음소 단위(Phoneme-level) 발음 교정 웹 애플리케이션입니다. 
학습자의 음성을 분석하여 **'4각 언어학적 대조 분석(목표 문장, 명료도, 조음 정확도, 모국어 간섭)'**을 수행하고, 구글 Gemini 2.5 Flash를 활용하여 일본인 특유의 발음 습관을 고려한 맞춤형 교정 피드백을 일본어로 제공합니다.

---

## ✨ 주요 특징
- **4각 언어학적 대조 분석 (4-Way Analysis)**: 
  - 🎯 **목표 문장 (Target)**: 의도한 기준점
  - 👥 **원어민 체감 (Intelligibility - Whisper)**: 원어민의 귀에 문맥상 어떻게 들리는지 시뮬레이션
  - 🗣️ **물리적 소리 (Acoustics - Wav2Vec2)**: 혀와 입술이 만들어낸 순수한 발음 기호
  - 🇯🇵 **모국어 간섭 (Katakana L1 Interference)**: 불필요한 모음 추가 등 일본어 모라(Mora) 시스템으로 인한 착각 시각화
- **IPA 병기**: 모든 한국어 문장에 정확한 국제음성기호(IPA)를 병기하여 직관적인 비교를 지원합니다.
- **최상급 언어학 피드백**: **Gemini 2.5 Flash** 모델이 조음적 오류를 포착하여 대학 교수급의 코칭을 제공합니다.
- **UI 전면 현지화**: 최종 사용자를 위해 인터페이스 및 피드백이 100% 일본어로 제공됩니다.

## 🛠 기술 스택
- **Frontend**: Streamlit
- **Acoustic Model**: `kresnik/wav2vec2-large-xlsr-korean` (실제 소리 인식)
- **Native Listener Model**: `openai/whisper-small` (원어민 명료도 시뮬레이션)
- **Linguistic AI**: Google Gemini 2.5 Flash (`google-generativeai`)
- **Audio Processing**: Librosa, FFmpeg

---

## 🚀 설치 및 실행 방법

1. **저장소 클론**:
   ```bash
   git clone https://github.com/fairyofdata/PhonemeJP2KR
   cd PhonemeJP2KR
   ```

2. **패키지 설치**:
   ```bash
   pip install -r requirements.txt
   ```
   *(오디오 처리를 위해 OS에 `ffmpeg`가 설치되어 있어야 합니다.)*

3. **Gemini API 키 설정**:
   - 프로젝트 최상단 루트에 `vertex_key_new.md` 파일을 생성합니다.
   - [Google AI Studio](https://aistudio.google.com/)에서 무료로 발급받은 API 키를 해당 파일에 붙여넣습니다.

4. **애플리케이션 실행**:
   ```bash
   cd Main
   streamlit run app.py
   ```

---

## 💡 실행 화면

학습자가 발음한 오디오를 바탕으로 점수, 파형, 4각 대조 결과, 그리고 상세한 교정 피드백이 제공됩니다.

![실험 결과](screencapture-localhost-8501-2026-06-18-22_08_56.png)

## 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다.