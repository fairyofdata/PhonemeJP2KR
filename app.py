"""Streamlit UI — phoneme-level Korean pronunciation coaching for JP speakers.

Analysis flow:
    audio → ffmpeg 16kHz mono → [Whisper, Wav2Vec2] →
    deterministic G2P/IPA + jamo alignment score →
    Gemini interprets the structured evidence (katakana + coaching).

The deterministic layer always renders, even if the LLM call fails.
Markup lives in src/ui.py; this file holds the page flow.
"""

import hashlib
import os
import subprocess
import tempfile

import imageio_ffmpeg
import librosa
import streamlit as st
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder

from src import ui
from src.asr import (
    load_whisper_model,
    load_wav2vec_model,
    transcribe_acoustics,
    transcribe_intelligibility,
)
from src.database import (
    delete_record, get_all_records, get_previous_score, get_weak_points,
    init_db, save_record,
)
from src.drills import DRILL_BY_TAG, DRILLS
from src.g2p import to_ipa, to_surface
from src.llm import GeminiUnavailableError, generate_feedback, translate_jp_to_kr
from src.reference import load_reference, score_band
from src.scoring import score_pronunciation
from src.tts import VOICES, generate_tts_audio

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 16000

st.set_page_config(page_title="韓国語 発音コーチ", page_icon=":material/graphic_eq:",
                   layout="wide", initial_sidebar_state="auto")
st.markdown(ui.GLOBAL_CSS, unsafe_allow_html=True)

init_db()
SCORE_REFERENCE = load_reference()

# how a score reads against faithful readings (see src/reference.py)
_BAND_TEXT = {
    "typical_faithful": ("正確な音読と同じ水準",
                         "正確に読んだ発話の約半数がこの範囲に入ります。"),
    "indeterminate": ("判定保留ゾーン",
                      "正確に読んでもよく出る範囲です。誤りかASRの揺れかは区別できないので、"
                      "もう一度録音して前回と比べてみましょう。"),
    "rare_for_faithful": ("正確な音読では稀なスコア",
                          "正確に読んだ発話でこの範囲に入るのは約10%のみです。"
                          "下の「音素の比較」で誤りの位置を確認しましょう。"),
}
_VOICE_LABELS = {"SunHi": "SunHi · 女性", "InJoon": "InJoon · 男性", "Hyunsu": "Hyunsu · 男性（柔らかめ）"}


@st.cache_resource(show_spinner="音声認識モデルを読み込んでいます…（初回のみ時間がかかります）")
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
            [FFMPEG_EXE, "-y", "-i", in_path, "-ar", str(SAMPLE_RATE), "-ac", "1", out_path],
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
        wav2vec_text, char_timestamps = transcribe_acoustics(wav_path, wav2vec_proc, wav2vec_mod)
        waveform, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    finally:
        os.unlink(wav_path)

    report = score_pronunciation(target, wav2vec_text, char_timestamps)
    result = {
        "target": target,
        "target_surface": to_surface(target),
        "target_ipa": to_ipa(target),
        "whisper_text": whisper_text,
        "whisper_ipa": to_ipa(whisper_text),
        "wav2vec_text": wav2vec_text,
        "actual_ipa": to_ipa(wav2vec_text),
        "score": report.score,
        "previous_score": get_previous_score(target),  # read before saving this one
        "diff_html": ui.diff_html(report.pairs),
        "error_tags": report.error_tags,
        "peaks": ui.envelope(waveform),
        "duration": len(waveform) / SAMPLE_RATE,
        "audio_bytes": audio_bytes,
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
                f"**[Katakana Mapping]**: {katakana}\n\n{feedback}",
                report.error_tags)
    return result


# --- callbacks (run before the next script pass, so they may set widget state) --

def _set_target(sentence: str):
    st.session_state.target_sentence = sentence
    st.session_state.pop("last_result", None)


def _translate():
    jp = st.session_state.get("jp_input", "").strip()
    if not jp:
        st.session_state.translate_error = "日本語を入力してください。"
        return
    try:
        _set_target(translate_jp_to_kr(jp))
        st.session_state.translate_error = None
    except GeminiUnavailableError as e:
        st.session_state.translate_error = str(e)


# --- page sections -------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("#### 弱点ドリル")
        st.caption("日本語母語話者に多い誤りを、単母音 → 二重母音 → 初声 → 音節構造 → 終声の順に練習します。")
        drill_idx = st.selectbox(
            "練習項目", range(len(DRILLS)), key="drill_idx",
            format_func=lambda i: f"{DRILLS[i]['stage']}　{DRILLS[i]['label']}",
            label_visibility="collapsed",
        )
        for sentence in DRILLS[drill_idx]["sentences"]:
            st.button(sentence, key=f"drill_{drill_idx}_{sentence}", width="stretch",
                      on_click=_set_target, args=(sentence,))

        st.divider()
        st.markdown("#### お手本の声")
        st.radio("お手本の声", list(VOICES), key="voice", label_visibility="collapsed",
                 format_func=lambda v: _VOICE_LABELS.get(v, v))

        st.divider()
        st.markdown("#### よく出る誤り")
        st.markdown(ui.weak_points_html(get_weak_points()[:4], DRILL_BY_TAG),
                    unsafe_allow_html=True)
        st.caption("直近30回の分析から集計しています。")


def render_practice():
    if "target_sentence" not in st.session_state:
        st.session_state.target_sentence = "감사합니다"

    left, right = st.columns([1.15, 1], gap="medium")
    with left, st.container(border=True):
        target = st.text_input("練習する韓国語の文", key="target_sentence",
                               placeholder="例: 서울에서 친구를 만났어요")
        with st.popover("日本語から作る", icon=":material/translate:"):
            st.text_input("言いたいことを日本語で", key="jp_input",
                          placeholder="例: お会いできて嬉しいです")
            st.button("韓国語に翻訳", on_click=_translate, type="primary", width="stretch")
            if st.session_state.get("translate_error"):
                st.error(st.session_state.translate_error)

        if target.strip():
            st.markdown(ui.target_html(target, to_surface(target), to_ipa(target)),
                        unsafe_allow_html=True)
            if st.button("お手本を聞く", icon=":material/volume_up:"):
                voice = st.session_state.get("voice", "SunHi")
                text_hash = hashlib.md5(target.encode()).hexdigest()
                out_path = os.path.join(tempfile.gettempdir(), f"tts_{text_hash}_{voice}.mp3")
                with st.spinner("音声を生成しています…"):
                    ok = os.path.exists(out_path) or generate_tts_audio(target, voice, out_path)
                if ok:
                    st.audio(out_path, format="audio/mp3")
                else:
                    st.error("音声の生成に失敗しました。ネットワーク接続を確認してください。")

    with right, st.container(border=True):
        st.markdown('<div class="pc-eyebrow">あなたの発音</div>', unsafe_allow_html=True)
        source = st.segmented_control("入力方法", ["マイクで録音", "ファイル"], key="source",
                                      default="マイクで録音", label_visibility="collapsed")
        audio_bytes = None
        if source == "ファイル":
            uploaded = st.file_uploader("音声ファイル（wav / mp3 / flac）",
                                        type=["wav", "mp3", "flac"])
            if uploaded is not None:
                audio_bytes = uploaded.getvalue()
        else:
            st.caption("ボタンを押して文を読み、終わったらもう一度押してください。")
            mic_data = mic_recorder(start_prompt="● 録音を開始", stop_prompt="■ 録音を停止",
                                    key="mic")
            if mic_data and mic_data.get("bytes"):
                audio_bytes = mic_data["bytes"]

        if audio_bytes:
            st.audio(audio_bytes, format=ui.audio_mime(audio_bytes))
        clicked = st.button("発音を分析する", type="primary", width="stretch",
                            icon=":material/analytics:", disabled=not audio_bytes)
        if not audio_bytes:
            st.caption("録音またはファイルを用意すると分析できます。")

    if clicked:
        if not target.strip():
            st.warning("先に練習する文を入力してください。")
        else:
            with st.spinner("音声を分析しています…"):
                try:
                    st.session_state.last_result = run_analysis(target, audio_bytes)
                except subprocess.CalledProcessError:
                    st.error("音声ファイルを読み込めませんでした。別の形式で試してください。")

    if st.session_state.get("last_result"):
        render_result(st.session_state.last_result)


def render_result(res: dict):
    st.markdown("### 分析結果")
    band = score_band(res["score"], SCORE_REFERENCE)
    label, text = _BAND_TEXT[band["band"]]
    st.markdown(ui.score_hero_html(res["score"], band, label, text, SCORE_REFERENCE,
                                   res.get("previous_score")), unsafe_allow_html=True)
    with st.expander("この点数の読み方"):
        st.markdown(
            f"- 点数は決定論的です（同じ音声なら常に同じ点数）。ただし**絶対値は未較正**で、"
            f"日本語母語話者が正確に読んだ発話 {SCORE_REFERENCE['n_faithful']} 件でも"
            f"中央値は {SCORE_REFERENCE['median']} 点、下位10%の境界は {SCORE_REFERENCE['p10']} 点でした"
            f"（AI-Hub コーパス、実験6）。\n"
            "- 上のバーの色分けはこの分布に基づく参考範囲です。上級者中心のコーパス音読文から"
            "得た基準のため、特に短い文（1字の誤りで点数が大きく動く）では目安としてご覧ください。\n"
            "- 最も信頼できるのは、**同じ文での前回との比較**です。"
        )

    if res.get("peaks"):
        markers = [e for e in res["error_tags"] if "timestamp" in e]
        components.html(
            ui.waveform_player_html(res["audio_bytes"], res["peaks"], res["duration"],
                                    res["error_tags"]),
            height=ui.player_height(len(markers)),
        )

    tab_diff, tab_channels, tab_coach = st.tabs(["音素の比較", "聞こえ方", "コーチング"])
    with tab_diff:
        st.caption("上段が実際の発音、置き換えの場合は下段に本来の音を表示します。")
        st.markdown(res["diff_html"], unsafe_allow_html=True)
        st.markdown("##### 検出された誤り")
        st.markdown(ui.error_list_html(res["error_tags"]), unsafe_allow_html=True)
        drills = {e["tag"] for e in res["error_tags"]} & set(DRILL_BY_TAG)
        if drills:
            st.caption("サイドバーの「弱点ドリル」で、"
                       + "・".join(f"「{DRILL_BY_TAG[t]['label']}」" for t in sorted(drills))
                       + " を練習できます。")
    with tab_channels:
        st.markdown(ui.channel_cards_html(res), unsafe_allow_html=True)
    with tab_coach:
        llm = res.get("llm")
        if llm:
            if llm.get("error_summary"):
                st.markdown(f"**{llm['error_summary']}**")
            with st.container(border=True):
                st.markdown(llm.get("feedback_jp", ""))
            st.caption("コーチングはLLM（Gemini）が上の分析結果を根拠に作成したものです。点数には影響しません。")
        else:
            st.info("コーチング文は利用できませんでした。点数と音素の比較は通常どおり表示しています。",
                    icon=":material/info:")
            if res.get("llm_error"):
                st.caption(res["llm_error"])


def render_history():
    records = get_all_records()
    if not records:
        st.markdown('<div class="pc-empty">まだ記録がありません。発音を分析すると、ここに履歴が残ります。</div>',
                    unsafe_allow_html=True)
        return

    col_chart, col_weak = st.columns([1.2, 1], gap="medium")
    with col_chart, st.container(border=True):
        st.markdown('<div class="pc-eyebrow">スコアの推移（直近30回）</div>', unsafe_allow_html=True)
        recent = list(reversed(records[:30]))
        st.line_chart({"スコア": [r["score"] for r in recent]}, height=210)
    with col_weak, st.container(border=True):
        st.markdown('<div class="pc-eyebrow">よく出る誤り（直近30回）</div>', unsafe_allow_html=True)
        st.markdown(ui.weak_points_html(get_weak_points(), DRILL_BY_TAG), unsafe_allow_html=True)

    st.markdown("##### これまでの練習")
    for r in records[:50]:
        with st.expander(f"{r['timestamp'][:16]}　{r['intended']}　·　{r['score']} 点"):
            st.markdown(f"**認識された発音:** {r['actual']}")
            st.markdown(r["feedback"])
            st.button("この記録を削除", key=f"del_{r['id']}", icon=":material/delete:",
                      on_click=delete_record, args=(r["id"],))


st.markdown(ui.header_html(), unsafe_allow_html=True)
render_sidebar()
tab_practice, tab_history = st.tabs(["練習", "学習記録"])
with tab_practice:
    render_practice()
with tab_history:
    render_history()
