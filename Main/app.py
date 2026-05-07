import streamlit as st
from src.models import load_whisper_model, load_wav2vec_model, transcribe_audio, analyze_pronunciation, generate_feedback
import tempfile
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display

st.set_page_config(page_title="Korean Pronunciation Correction", layout="wide")

# Load models on app start
@st.cache_resource(show_spinner="AI 모델을 불러오는 중입니다... (최초 1회 시간이 소요됩니다)")
def load_models():
    whisper_processor, whisper_model = load_whisper_model()
    wav2vec_processor, wav2vec_model = load_wav2vec_model()
    return whisper_processor, whisper_model, wav2vec_processor, wav2vec_model

whisper_proc, whisper_mod, wav2vec_proc, wav2vec_mod = load_models()

st.title("LLM-based Phoneme-level Korean Pronunciation Correction for Japanese Native Speakers")

st.markdown("""
Upload an audio file to analyze pronunciation. The system will detect intended vs. actual pronunciation and provide feedback.
""")

# Sidebar for API Key
openai_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password", help="여기에 OpenAI API 키를 입력하세요 (sk-...)")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Analyze Pronunciation"):
        with st.spinner("Analyzing..."):
            # 분석을 시작할 때만 Temp 파일에 기록하여 자원 절약
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name

            try:
                intended = transcribe_audio(audio_path, whisper_proc, whisper_mod)
                actual = analyze_pronunciation(audio_path, wav2vec_proc, wav2vec_mod)

                st.subheader("Results")
                
                # 1. 오디오 파형 시각화
                st.write("**Audio Waveform**")
                fig, ax = plt.subplots(figsize=(10, 2))
                y, sr = librosa.load(audio_path, sr=16000)
                librosa.display.waveshow(y, sr=sr, ax=ax, color="#1f77b4")
                ax.set_axis_off()
                st.pyplot(fig)

                # 2. 발음 대조 시각화
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"🎯 **의도한 발음 (Intended)**\n### {intended}")
                with col2:
                    st.warning(f"🗣️ **실제 들린 발음 (Actual)**\n### {actual}")

                # 3. LLM 교정 피드백
                st.write("---")
                st.subheader("💡 Expert Feedback")
                feedback = generate_feedback(intended, actual, api_key=openai_api_key)
                st.success(feedback)
            finally:
                # 작업 완료 또는 에러 발생 시 파일 삭제를 확실히 보장
                if os.path.exists(audio_path):
                    os.unlink(audio_path)

st.markdown("---")
st.markdown("**Note:** This is a basic prototype. Models are loaded locally for efficiency.")