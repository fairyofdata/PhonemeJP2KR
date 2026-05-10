[🇺🇸 English](Readme.md) | [🇯🇵 日本語 (Japanese)](README_jp.md)

⚠️ *주의: 이 프로젝트는 현재 기획 및 프로토타입 단계입니다.*

[프로젝트 기획 노션 링크](https://fairydata.notion.site/49b72350b1074fb094e0ec792cff7d59?pvs=4)

# **🗣️ 일본인 학습자를 위한 LLM 기반 한국어 발음 교정 서비스** 🧑‍🏫📝
[![영상 제목](https://img.youtube.com/vi/4SwwmzEcpZQ/0.jpg)](https://youtu.be/4SwwmzEcpZQ)<br>

> 일본인 한국어 학습자를 위한 LLM 기반 음소 단위(Phoneme-level) 발음 교정 웹 애플리케이션입니다. 
학습자의 음성을 분석하여 '의도한 발음'과 '실제 인식된 발음'을 대조하고, 일본어 모어 화자 특유의 발음 습관을 고려하여 맞춤형 교정 피드백을 제공합니다.

---

## ✨ 주요 기능
- **음성 파일 업로드 및 시각화**: `wav`, `mp3`, `flac` 형태의 오디오 파일을 업로드하고 파형(Waveform)을 시각적으로 확인할 수 있습니다.
- **의도한 발음 인식 (Whisper)**: OpenAI의 Whisper 모델을 활용하여 학습자가 의도한 한국어 문장을 텍스트로 추출합니다.
- **실제 발음 분석 (Wav2Vec2)**: 한국어에 특화된 Wav2Vec2 모델을 통해 실제 소리나는 대로의 발음을 음소 단위로 인식합니다.
- **전문가 AI 피드백 (OpenAI GPT-4o-mini)**: 두 발음의 차이를 분석하여, 입모양이나 혀의 위치 등 구체적이고 친절한 교정 방법을 제시합니다.

## 🛠 기술 스택
- **Frontend**: Streamlit
- **AI / ML**: 
  - `openai/whisper-small` (의도된 발음 인식)
  - `jonatasgrosman/wav2vec2-large-xlsr-53-korean` (실제 발음 인식)
  - `OpenAI API` (GPT-4o-mini 기반 피드백 생성)
- **Audio Processing**: Librosa, Matplotlib

## 🚀 설치 및 실행 방법

1. **저장소 클론**:
   ```bash
   git clone https://github.com/fairyofdata/JP2KR_PhonemeTutor
   cd JP2KR_PhonemeTutor
   ```

2. **패키지 설치**:
   ```bash
   pip install streamlit torch transformers librosa langdetect openai matplotlib
   ```
   *(GPU 환경인 경우, 사용 중인 CUDA 버전에 맞는 `torch`를 설치해 주세요.)*

3. **애플리케이션 실행**:
   ```bash
   cd Main
   streamlit run app.py
   ```

4. **사용 방법**:
   - 앱이 실행되면 좌측 사이드바에 **OpenAI API Key**(`sk-...`)를 입력합니다.
   - 일본인 학습자가 한국어로 녹음한 음성 파일을 업로드합니다.
   - **"Analyze Pronunciation"** 버튼을 클릭하여 파형, 발음 분석 결과 및 피드백을 확인합니다!

## 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다.