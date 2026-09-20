"""API-key resolution, including the .env file."""

import os

from src import config


def _write(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_read_env_file_parses_common_forms(tmp_path):
    path = _write(tmp_path, "\n".join([
        "# comment",
        "",
        "GEMINI_API_KEY=abc123",
        'QUOTED="q-value"',
        "SINGLE='s-value'",
        "export EXPORTED=e-value",
        "SPACED = spaced-value ",
        "NO_VALUE=",
        "malformed line",
    ]))
    values = config.read_env_file(path)
    assert values == {
        "GEMINI_API_KEY": "abc123",
        "QUOTED": "q-value",
        "SINGLE": "s-value",
        "EXPORTED": "e-value",
        "SPACED": "spaced-value",
    }


def test_read_env_file_missing_returns_empty(tmp_path):
    assert config.read_env_file(str(tmp_path / "absent")) == {}


def test_env_var_wins_over_env_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENV_FILE", _write(tmp_path, "GEMINI_API_KEY=from-file"))
    monkeypatch.setenv("GEMINI_API_KEY", "from-environment")
    assert config.get_gemini_api_key() == "from-environment"


def test_env_file_used_when_no_env_var(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENV_FILE", _write(tmp_path, "GEMINI_API_KEY=from-file"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert config.get_gemini_api_key() == "from-file"


def test_reading_env_file_does_not_leak_into_process_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENV_FILE", _write(tmp_path, "SOME_OTHER_KEY=value"))
    monkeypatch.delenv("SOME_OTHER_KEY", raising=False)
    config.get_gemini_api_key()
    assert "SOME_OTHER_KEY" not in os.environ
