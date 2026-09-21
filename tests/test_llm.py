"""Coaching-model fallback (src/llm.py). Needs google-genai; skipped in CI."""

import pytest

pytest.importorskip("google.genai")

from src import llm  # noqa: E402


class _Err(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class _Client:
    """Fails with the given code per model; answers 'ok' otherwise."""

    def __init__(self, failures):
        self.failures, self.tried = failures, []
        self.models = self

    def generate_content(self, model, contents, config):
        self.tried.append(model)
        if model in self.failures:
            raise _Err(self.failures[model])
        return "ok"


def test_falls_through_overloaded_and_unavailable_models(monkeypatch):
    monkeypatch.setattr(llm, "GEMINI_MODELS", ("a", "b", "c"))
    client = _Client({"a": 503, "b": 404})
    assert llm._generate(client, "x", None) == ("ok", "c")
    assert client.tried == ["a", "b", "c"]


def test_other_errors_are_not_retried_on_another_model(monkeypatch):
    monkeypatch.setattr(llm, "GEMINI_MODELS", ("a", "b"))
    client = _Client({"a": 400})
    with pytest.raises(_Err):
        llm._generate(client, "x", None)
    assert client.tried == ["a"]


def test_raises_the_last_error_when_every_model_fails(monkeypatch):
    monkeypatch.setattr(llm, "GEMINI_MODELS", ("a", "b"))
    with pytest.raises(_Err) as e:
        llm._generate(_Client({"a": 503, "b": 429}), "x", None)
    assert e.value.code == 429
