# 🗣️ 한국어 발음 교정 서비스 (일본인 학습자 대상)

일본인 한국어 학습자를 위한 LLM 기반 음소 단위(Phoneme-level) 발음 교정 웹 애플리케이션입니다. 
학습자의 음성을 분석하여 '의도한 발음'과 '실제 인식된 발음'을 대조하고, 일본어 모어 화자 특유의 발음 습관을 분석하여 맞춤형 교정 피드백을 제공합니다.

## ✨ 주요 기능 (Features)

- **음성 파일 업로드 및 시각화**: `wav`, `mp3`, `flac` 형태의 오디오 파일을 업로드하고 파형(Waveform)을 시각적으로 확인할 수 있습니다.
- **의도한 발음 인식 (Whisper)**: OpenAI의 Whisper 모델을 활용하여 학습자가 의도한 한국어 문장을 텍스트로 추출합니다.
- **실제 발음 분석 (Wav2Vec2)**: 한국어에 특화된 Wav2Vec2 모델을 통해 실제 소리나는 대로의 발음을 음소 단위로 인식합니다.
- **전문가 AI 피드백 (OpenAI GPT-4o-mini)**: 두 발음의 차이를 분석하여, 입모양이나 혀의 위치 등 구체적이고 친절한 교정 방법을 제시합니다.

## 🛠 기술 스택 (Tech Stack)

- **Frontend**: Streamlit
- **AI / ML**: 
  - `openai/whisper-small` (의도된 발음 인식)
  - `jonatasgrosman/wav2vec2-large-xlsr-53-korean` (실제 발음 인식)
  - `OpenAI API` (GPT-4o-mini 기반 피드백 생성)
- **Audio Processing**: Librosa, Matplotlib

## 🚀 설치 및 실행 방법 (How to Run)

### 1. 사전 요구 사항 (Prerequisites)
- Python 3.8 이상
- (선택) `ffmpeg` 설치 (오디오 파일 처리를 위해 필요할 수 있습니다)

### 2. 패키지 설치
프로젝트 실행에 필요한 파이썬 패키지를 설치합니다.
```bash
pip install streamlit torch transformers librosa langdetect openai matplotlib
```
*(GPU 환경인 경우, 사용 중인 CUDA 버전에 맞는 `torch`를 설치해 주세요.)*

### 3. 애플리케이션 실행
```bash
streamlit run app.py
```

### 4. 사용 방법
1. 앱이 실행되면 브라우저에서 `http://localhost:8501`로 접속됩니다.
2. 좌측 사이드바에 **OpenAI API Key**(`sk-...`)를 입력합니다.
3. 한국어로 녹음된 음성 파일을 업로드합니다.
4. **"Analyze Pronunciation"** 버튼을 클릭하여 결과를 확인합니다!

## 📂 프로젝트 구조 (Project Structure)

```text
Nantokanaru/Main/
 ├── app.py             # Streamlit 메인 애플리케이션
 ├── src/
 │    └── models.py     # AI 모델 로드, 음성 인식 및 LLM 피드백 생성 로직
 └── README.md          # 프로젝트 설명서
```