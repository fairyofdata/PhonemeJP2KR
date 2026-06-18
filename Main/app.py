import streamlit as st
from src.models import load_whisper_model, load_wav2vec_model, transcribe_audio, analyze_pronunciation, generate_feedback
from src.models import calculate_phoneme_score
from src.database import init_db, save_record, get_all_records
import tempfile
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
from streamlit_mic_recorder import mic_recorder
import imageio_ffmpeg
import subprocess

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

st.set_page_config(page_title="Korean Pronunciation Correction", layout="wide")

# Initialize DB
init_db()

# Load models on app start
@st.cache_resource(show_spinner="AIモデルを読み込んでいます... (初回のみ時間がかかります)")
def load_models():
    whisper_processor, whisper_model = load_whisper_model()
    wav2vec_processor, wav2vec_model = load_wav2vec_model()
    return whisper_processor, whisper_model, wav2vec_processor, wav2vec_model

whisper_proc, whisper_mod, wav2vec_proc, wav2vec_mod = load_models()

st.title("LLMベース 音素レベル 韓国語 発音矯正 (Japanese Native Speakers用)")

# Sidebar
st.sidebar.markdown("🔒 **認証情報**\n\nGoogle Gemini API 認証を使用してフィードバックを生成します。")

# Tabs
tab_analysis, tab_history = st.tabs(["🎙️ 発音矯正", "📚 学習記録"])

with tab_analysis:
    st.markdown("""
    練習する文章を入力した後、ブラウザのマイクで直接録音するか、音声ファイルをアップロードして発音を分析してみてください。
    """)
    
    target = st.text_input("🎯 練習する文章を正確に入力してください (目標文章)", value="감사합니다")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. 直接録音する")
        audio_recorder_data = mic_recorder(start_prompt="🔴 録音開始", stop_prompt="⏹️ 録音停止", key='mic')
    with col2:
        st.subheader("2. ファイルをアップロードする")
        uploaded_file = st.file_uploader("音声ファイルを選択してください", type=["wav", "mp3", "flac"])

    audio_data = None
    if audio_recorder_data and audio_recorder_data.get('bytes'):
        audio_data = audio_recorder_data['bytes']
        st.audio(audio_data, format="audio/wav")
    elif uploaded_file is not None:
        audio_data = uploaded_file.getvalue()
        st.audio(audio_data, format="audio/wav")

    if audio_data is not None:
        if st.button("発音を分析する", type="primary"):
            with st.spinner("Analyzing..."):
                in_path = None
                audio_path = None
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_in:
                    tmp_in.write(audio_data)
                    in_path = tmp_in.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
                    audio_path = tmp_out.name

                try:
                    subprocess.run(
                        [ffmpeg_exe, "-y", "-i", in_path, "-ar", "16000", "-ac", "1", audio_path],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )

                    whisper_text = transcribe_audio(audio_path, whisper_proc, whisper_mod)
                    wav2vec_text = analyze_pronunciation(audio_path, wav2vec_proc, wav2vec_mod)
                    
                    st.subheader("分析結果 (Results)")
                    st.markdown("---")

                    with st.spinner("AIが言語学的な分析を実行しています..."):
                        analysis_result = generate_feedback(target, whisper_text, wav2vec_text)
                    
                    target_ipa = analysis_result.get("target_ipa", "N/A")
                    whisper_ipa = analysis_result.get("whisper_ipa", "N/A")
                    wav2vec_ipa = analysis_result.get("wav2vec_ipa", "N/A")
                    katakana = analysis_result.get("katakana", "N/A")
                    feedback_jp = analysis_result.get("feedback_jp", "N/A")
                    
                    # Score is based on Target vs Wav2Vec2 (physical sound)
                    score = calculate_phoneme_score(target_ipa, wav2vec_ipa)

                    st.markdown(f"### 🏆 発音の正確さ (Phoneme Score): **{score} / 100**")
                    st.progress(score / 100.0)
                    
                    st.write("**Audio Waveform**")
                    fig, ax = plt.subplots(figsize=(10, 2))
                    y, sr = librosa.load(audio_path, sr=16000)
                    librosa.display.waveshow(y, sr=sr, ax=ax, color="#1f77b4")
                    ax.set_axis_off()
                    st.pyplot(fig)

                    st.markdown("### 🔍 4角 言語学的対照分析")
                    col3, col4 = st.columns(2)
                    with col3:
                        st.info(f"🎯 **目標文章 (Target)**\n\n### {target}\n**IPA**: `/{target_ipa}/`")
                    with col4:
                        st.warning(f"👥 **ネイティブの体感 (Intelligibility)**\n\n### {whisper_text}\n**IPA**: `/{whisper_ipa}/`\n*(Whisperの認識結果)*")
                        
                    col5, col6 = st.columns(2)
                    with col5:
                        st.error(f"🗣️ **物理的な音声 (Acoustics)**\n\n### {wav2vec_text}\n**IPA**: `/{wav2vec_ipa}/`\n*(Wav2Vec2の認識結果)*")
                    with col6:
                        st.error(f"🇯🇵 **L1 干渉 (Katakana)**\n\n### {katakana}\n*(カタカナマッピング)*")

                    st.write("---")
                    st.subheader("💡 Expert Feedback")
                    if "API_KEY_SERVICE_BLOCKED" in feedback_jp:
                        st.error("⚠️ **API Key Error: API_KEY_SERVICE_BLOCKED**\n\nご提供いただいたAPIキーで Gemini API (Generative Language API) にアクセスできません。\n\n**解決方法:**\nGoogle Cloud Console にアクセスし、このAPIキーが存在するプロジェクトで **'Generative Language API'** を有効にするか、Google AI Studio (aistudio.google.com) で新しく無料のAPIキーを発行して `vertex_key_new.md` に貼り付けてください。")
                    else:
                        st.success(feedback_jp)
                    
                    combined_feedback = f"**[Katakana Mapping]**: {katakana}\n\n{feedback_jp}"
                    save_record(target, wav2vec_text, score, combined_feedback)
                finally:
                    if in_path and os.path.exists(in_path):
                        os.unlink(in_path)
                    if audio_path and os.path.exists(audio_path):
                        os.unlink(audio_path)

with tab_history:
    st.subheader("지난 발음 교정 기록")
    records = get_all_records()
    if not records:
        st.write("아직 저장된 학습 기록이 없습니다. 발음 분석을 진행해 보세요!")
    else:
        for r in records:
            with st.expander(f"[{r['timestamp']}] {r['intended']} (점수: {r['score']}점)"):
                st.write(f"**의도한 발음:** {r['intended']}")
                st.write(f"**실제 들린 발음:** {r['actual']}")
                st.markdown("**피드백:**")
                st.write(r['feedback'])

st.markdown("---")
st.markdown("**Note:** This is a basic prototype. Models are loaded locally for efficiency.")