"""Streamlit UI — phoneme-level Korean pronunciation coaching for JP speakers.

Analysis flow:
    audio → ffmpeg 16kHz mono → [Whisper, Wav2Vec2] →
    deterministic G2P/IPA + jamo alignment score →
    Gemini interprets the structured evidence (katakana + coaching).

The deterministic layer always renders, even if the LLM call fails.
"""

import hashlib
import os
import subprocess
import tempfile

import imageio_ffmpeg
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_mic_recorder import mic_recorder

from src.asr import (
    load_whisper_model,
    load_wav2vec_model,
    transcribe_acoustics,
    transcribe_intelligibility,
)
from src.database import delete_record, get_all_records, init_db, save_record
from src.g2p import to_ipa, to_surface
from src.llm import GeminiUnavailableError, generate_feedback, translate_jp_to_kr
from src.scoring import render_diff_markdown, score_pronunciation
from src.tts import VOICES, generate_tts_audio

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

st.set_page_config(page_title="Korean Pronunciation Coach", layout="wide")

init_db()


@st.cache_resource(show_spinner="AIモデルを読み込んでいます... (初回のみ時間がかかります)")
def load_models():
    whisper_processor, whisper_model = load_whisper_model()
    wav2vec_processor, wav2vec_model = load_wav2vec_model()
    return whisper_processor, whisper_model, wav2vec_processor, wav2vec_model


whisper_proc, whisper_mod, wav2vec_proc, wav2vec_mod = load_models()


def convert_to_wav16k(audio_bytes: bytes) -> str:
    """Normalize arbitrary input audio to 16kHz mono WAV. Returns the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_in:
        tmp_in.write(audio_bytes)
        in_path = tmp_in.name
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        subprocess.run(
            [FFMPEG_EXE, "-y", "-i", in_path, "-ar", "16000", "-ac", "1", out_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    finally:
        os.unlink(in_path)
    return out_path


def run_analysis(target: str, audio_bytes: bytes) -> dict:
    """Full pipeline. Returns a result dict stored in session_state."""
    wav_path = convert_to_wav16k(audio_bytes)
    try:
        whisper_text = transcribe_intelligibility(wav_path, whisper_proc, whisper_mod)
        wav2vec_text = transcribe_acoustics(wav_path, wav2vec_proc, wav2vec_mod)
        waveform, _ = librosa.load(wav_path, sr=16000)
    finally:
        os.unlink(wav_path)

    report = score_pronunciation(target, wav2vec_text)
    result = {
        "target": target,
        "target_surface": to_surface(target),
        "target_ipa": to_ipa(target),
        "whisper_text": whisper_text,
        "whisper_ipa": to_ipa(whisper_text),
        "wav2vec_text": wav2vec_text,
        "actual_ipa": to_ipa(wav2vec_text),
        "score": report.score,
        "diff_markdown": render_diff_markdown(report.pairs),
        "error_tags": report.error_tags,
        "waveform": waveform,
        "llm": None,
        "llm_error": None,
    }
    try:
        result["llm"] = generate_feedback(
            target=target,
            target_surface=result["target_surface"],
            target_ipa=result["target_ipa"],
            whisper_text=whisper_text,
            wav2vec_text=wav2vec_text,
            actual_ipa=result["actual_ipa"],
            score=report.score,
            error_tags=report.error_tags,
        )
    except GeminiUnavailableError as e:
        result["llm_error"] = str(e)

    katakana = (result["llm"] or {}).get("katakana", "N/A")
    feedback = (result["llm"] or {}).get("feedback_jp", result["llm_error"] or "")
    save_record(target, wav2vec_text, report.score,
                f"**[Katakana Mapping]**: {katakana}\n\n{feedback}")
    return result


def render_result(res: dict):
    st.markdown(f"### 🏆 音素スコア (Phoneme Score): **{res['score']} / 100**")
    st.caption("スコアはG2P音韻規則エンジンによるjamoアライメント（1 − 音素誤り率）で決定論的に算出されます。")
    st.progress(res["score"] / 100.0)

    st.write("**Audio Waveform**")
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(res["waveform"], sr=16000, ax=ax, color="#1f77b4")
    ax.set_axis_off()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### 🔍 4チャンネル言語学的対照分析")
    col1, col2 = st.columns(2)
    with col1:
        st.info(
            f"🎯 **目標文章 (Target)**\n\n### {res['target']}\n"
            f"表面発音: **{res['target_surface']}**\n\n"
            f"IPA: `/{res['target_ipa']}/`"
        )
    with col2:
        st.warning(
            f"👂 **ネイティブの聞こえ方 (Intelligibility / Whisper)**\n\n"
            f"### {res['whisper_text']}\n"
            f"IPA: `/{res['whisper_ipa']}/`"
        )
    col3, col4 = st.columns(2)
    with col3:
        st.error(
            f"🗣️ **物理的な音 (Acoustics / Wav2Vec2)**\n\n"
            f"### {res['wav2vec_text']}\n"
            f"IPA: `/{res['actual_ipa']}/`"
        )
    with col4:
        katakana = (res["llm"] or {}).get("katakana", "（LLM未実行）")
        st.error(
            f"🇯🇵 **L1干渉の可視化 (Katakana)**\n\n### {katakana}\n"
            f"*(あなたの発音をカタカナで表記)*"
        )

    st.markdown("### 🧬 音素レベル差分 (Target vs 実際の発音)")
    st.caption("上段: 目標のjamo列 / 下段: 実際に発音されたjamo列（**太字** = 不一致、· = 欠落/挿入）")
    st.markdown(res["diff_markdown"])

    st.write("---")
    st.subheader("💡 専門家フィードバック (Gemini)")
    if res["llm"]:
        if res["llm"].get("error_summary"):
            st.markdown(f"**要約:** {res['llm']['error_summary']}")
        st.success(res["llm"].get("feedback_jp", ""))
    else:
        st.warning(
            f"LLMフィードバックは利用できませんでした（決定論的な分析結果のみ表示中）。\n\n{res['llm_error']}"
        )


st.title("音素レベル韓国語発音コーチ 🇰🇷 (日本語母語話者向け)")
st.caption(
    "デュアルASR (Whisper × Wav2Vec2) + 決定論的G2P音韻規則エンジン + LLM解釈による発音矯正"
)

tab_analysis, tab_history = st.tabs(["🎙️ 発音分析", "📚 学習記録"])

with tab_analysis:
    if "target_sentence" not in st.session_state:
        st.session_state.target_sentence = "감사합니다"

    st.markdown("### Step 1. 言いたいことを入力 (日本語 → 韓国語)")
    col_jp1, col_jp2 = st.columns([3, 1])
    with col_jp1:
        jp_input = st.text_input(
            "🗣️ 言いたい日本語を入力してください",
            placeholder="例: こんにちは、お会いできて嬉しいです",
        )
    with col_jp2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("韓国語に翻訳", use_container_width=True):
            if jp_input.strip():
                with st.spinner("翻訳中..."):
                    try:
                        st.session_state.target_sentence = translate_jp_to_kr(jp_input)
                    except GeminiUnavailableError as e:
                        st.error(str(e))
            else:
                st.warning("日本語を入力してください。")

    st.markdown("### Step 2. 目標文章の確認と録音")
    target = st.text_input("🎯 練習する韓国語の文章", key="target_sentence")

    if target.strip():
        st.markdown(
            f"標準表面発音: **{to_surface(target)}**　｜　IPA: `/{to_ipa(target)}/`"
        )
        st.markdown("#### 🎧 ネイティブの発音を聞く (TTS)")
        voice_choice = st.radio(
            "音声を選択:",
            ["👩 SunHi (落ち着いた女性)", "👨 InJoon (信頼感のある男性)", "👦 Hyunsu (柔らかい男性)"],
            horizontal=True,
        )
        if st.button("🔊 お手本を再生する"):
            voice_key = next((v for v in VOICES if v in voice_choice), "SunHi")
            text_hash = hashlib.md5(target.encode()).hexdigest()
            out_path = os.path.join(
                tempfile.gettempdir(), f"tts_{text_hash}_{voice_key}.mp3"
            )
            with st.spinner("音声を生成しています..."):
                ok = os.path.exists(out_path) or generate_tts_audio(
                    target, voice_key, out_path
                )
            if ok:
                st.audio(out_path, format="audio/mp3")
            else:
                st.error("音声の生成に失敗しました。")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("A. マイクで録音する")
        mic_data = mic_recorder(
            start_prompt="🔴 録音開始", stop_prompt="⏹️ 録音停止", key="mic"
        )
    with col2:
        st.subheader("B. ファイルをアップロードする")
        uploaded_file = st.file_uploader(
            "音声ファイルを選択してください", type=["wav", "mp3", "flac"]
        )

    audio_bytes = None
    if mic_data and mic_data.get("bytes"):
        audio_bytes = mic_data["bytes"]
    elif uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

    if audio_bytes and st.button("発音を分析する", type="primary"):
        if not target.strip():
            st.warning("先に目標文章を入力してください。")
        else:
            with st.spinner("音声を分析しています..."):
                st.session_state.last_result = run_analysis(target, audio_bytes)

    if st.session_state.get("last_result"):
        st.markdown("---")
        render_result(st.session_state.last_result)

with tab_history:
    st.subheader("これまでの練習記録")
    records = get_all_records()
    if not records:
        st.write("まだ記録がありません。発音分析を試してみてください！")
    for r in records:
        with st.expander(f"[{r['timestamp']}] {r['intended']} (スコア: {r['score']}点)"):
            st.write(f"**目標文章:** {r['intended']}")
            st.write(f"**実際に聞こえた音:** {r['actual']}")
            st.markdown("**フィードバック:**")
            st.write(r["feedback"])
            if st.button("🗑️ この記録を削除", key=f"del_{r['id']}", type="secondary"):
                delete_record(r["id"])
                st.rerun()
