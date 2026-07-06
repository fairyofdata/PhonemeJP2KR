"""Central configuration: model IDs, paths, API-key resolution."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "history.db")

WHISPER_MODEL_ID = "openai/whisper-small"
WAV2VEC_MODEL_ID = "kresnik/wav2vec2-large-xlsr-korean"
GEMINI_MODEL_ID = "gemini-2.5-flash"

AUDIO_SAMPLE_RATE = 16000

# legacy key file kept for backwards compatibility with earlier setups
_LEGACY_KEY_FILE = os.path.join(PROJECT_ROOT, "vertex_key_new.md")


def get_gemini_api_key() -> str | None:
    """Resolve the Gemini API key: env var → Streamlit secrets → legacy file."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    if os.path.exists(_LEGACY_KEY_FILE):
        with open(_LEGACY_KEY_FILE, "r", encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key
    return None
