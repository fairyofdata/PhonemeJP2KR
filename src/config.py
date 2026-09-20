"""Central configuration: model IDs, paths, API-key resolution."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "history.db")
CLIPS_DIR = os.path.join(DATA_DIR, "clips")   # recordings kept for replay

WHISPER_MODEL_ID = "openai/whisper-small"
WAV2VEC_MODEL_ID = "kresnik/wav2vec2-large-xlsr-korean"
GEMINI_MODEL_ID = "gemini-2.5-flash"

AUDIO_SAMPLE_RATE = 16000

ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

# legacy key file kept for backwards compatibility with earlier setups
_LEGACY_KEY_FILE = os.path.join(PROJECT_ROOT, "vertex_key_new.md")


def read_env_file(path: str = None) -> dict:
    """Parse a .env file into a dict (default: ENV_FILE). Missing file → {}.

    Deliberately small: KEY=VALUE per line, '#' comments, optional
    surrounding quotes, an optional 'export ' prefix. Nothing is written
    into os.environ, so a real environment variable always wins.
    """
    values = {}
    try:
        with open(path or ENV_FILE, encoding="utf-8-sig") as f:
            lines = f.readlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.removeprefix("export ").strip()
        value = value.split(" #", 1)[0].strip().strip("\"'")
        if name and value:
            values[name] = value
    return values


def get_gemini_api_key() -> str | None:
    """Resolve the Gemini API key: env var → .env → Streamlit secrets → legacy file."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    key = read_env_file().get("GEMINI_API_KEY")
    if key:
        return key
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
