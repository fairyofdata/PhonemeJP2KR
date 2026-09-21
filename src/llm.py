"""Gemini-based pedagogical feedback and JP→KR translation.

Design principle: the LLM never computes measurements. IPA transcriptions,
the phoneme score, and the error alignment all come from the deterministic
G2P/scoring pipeline and are passed *into* the prompt as evidence. The LLM
is used only for what it is good at — interpreting structured evidence and
writing natural pedagogical feedback in Japanese. (The katakana line is
not the LLM's: src/kana.py derives it from the alignment by rule.)
"""

import json

from google import genai
from google.genai import types

from .config import GEMINI_MODELS, get_gemini_api_key
from .labels import tag_label

_FEEDBACK_PROMPT = """あなたは日本語母語話者の母語干渉（L1 Interference）を深く理解している韓国語発音矯正の専門家です。

以下の分析データは、決定論的な音韻規則エンジン（G2P）と2つのASRモデルによって機械的に算出されたものです。データの再計算はせず、解釈と指導に専念してください。

[分析データ]
- 学習者が意図した文章 (Target): {target}
- Targetの標準表面発音 (Surface form, G2Pによる): {target_surface}
- TargetのIPA (G2Pによる決定論的転写): /{target_ipa}/
- ネイティブの聞こえ方 (Whisper認識結果): {whisper_text}
- 物理的に発音された音 (Wav2Vec2認識結果): {wav2vec_text}
- 実際の発音のIPA (G2Pによる): /{actual_ipa}/
- 音素レベルスコア: {score}/100
- 自動検出された誤り (jamoアライメントに基づく。nameは学習者に見せる日本語名): {error_tags}

[日本語母語話者の典型的エラーパターン (参考)]
1. 母音挿入 (Epenthesis): モーラ拍リズムの影響でパッチムの後に /ɯ/ や /u/ を挿入する。
2. 三項対立の混同: 平音・激音・濃音を有声/無声の二項対立で代替する。
3. 母音の歪み: /ʌ/(ㅓ) を /o/ で、/ɯ/(ㅡ) を /u/ で代替する。
4. 終声の弁別失敗: /n/, /ŋ/, /m/ のパッチムを日本語の「ん」に統合する。
5. 閉鎖音終声の混同: /k̚/, /t̚/, /p̚/ のパッチムを促音「っ」のように後続子音に同化させ、調音位置を失う。
6. 二重母音の問題: /jʌ/(ㅕ) を /jo/(ㅛ) で代替し、語頭の /ɰi/(ㅢ) を /i/ や /ɯ/ に単母音化する。

[タスク]
上記の検出結果と2つのASR結果の差分を根拠として、以下のJSONのみを出力してください。

[執筆ルール]
- 学習者に見せる文章です。`coda_deletion` のような内部タグIDは絶対に書かないでください。誤りを指す場合は上記の name（日本語名）を使ってください。
- マークダウンの見出し記法(#)は使わず、各誤りは「**日本語名**」で始まる短い段落にしてください。
- 重要度の高い誤りを最大3件まで取り上げ、それ以外は最後に1文でまとめてください。

{{
  "error_summary": "検出された誤りの要点を1〜2文の日本語で。誤りがなければその旨を書く。",
  "feedback_jp": "エラータグごとに、口・舌・喉の使い方まで踏み込んだ具体的な矯正アドバイス（日本語、マークダウン使用可、3〜6文程度）。エビデンス（どの音がどう変わったか）を必ず引用すること。"
}}"""

_TRANSLATE_PROMPT = (
    "次の日本語の文を、発音練習に適した自然な話し言葉の韓国語に翻訳してください。"
    "訳文の韓国語1文のみを出力し、引用符・説明・マークダウンは一切含めないでください。\n\n"
    "日本語: {jp_text}"
)


def _named(error_tags: list) -> list:
    """Attach the learner-facing Japanese name to each tag for the prompt."""
    return [{**t, "name": tag_label(t.get("tag", ""))} for t in (error_tags or [])]


class GeminiUnavailableError(RuntimeError):
    """Raised when no API key is configured or the API call fails."""


def _get_client() -> genai.Client:
    api_key = get_gemini_api_key()
    if not api_key:
        raise GeminiUnavailableError(
            "Gemini APIキーが見つかりません。環境変数 GEMINI_API_KEY を設定するか、"
            ".streamlit/secrets.toml に GEMINI_API_KEY を追加してください。"
        )
    return genai.Client(api_key=api_key)


# "try another model": overloaded (503), quota or free-tier limit (429),
# not available to this key (403/404)
_FALLBACK_CODES = (403, 404, 429, 503)


def _generate(client, contents, config):
    """generate_content on the first model in GEMINI_MODELS that answers.

    Returns (response, model_id). Any other error is raised at once.
    """
    last = None
    for model in GEMINI_MODELS:
        try:
            return client.models.generate_content(model=model, contents=contents,
                                                  config=config), model
        except Exception as e:
            if getattr(e, "code", None) not in _FALLBACK_CODES:
                raise
            last = e
    raise last


def generate_feedback(target: str, target_surface: str, target_ipa: str,
                      whisper_text: str, wav2vec_text: str, actual_ipa: str,
                      score: int, error_tags: list) -> dict:
    """Interpret the deterministic analysis and return the coaching text."""
    prompt = _FEEDBACK_PROMPT.format(
        target=target,
        target_surface=target_surface,
        target_ipa=target_ipa,
        whisper_text=whisper_text,
        wav2vec_text=wav2vec_text,
        actual_ipa=actual_ipa,
        score=score,
        error_tags=json.dumps(_named(error_tags), ensure_ascii=False),
    )
    client = _get_client()
    try:
        response, model = _generate(client, prompt, types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ))
        return {**json.loads(response.text), "model": model}
    except Exception as e:
        raise GeminiUnavailableError(f"Gemini API呼び出しに失敗しました: {e}") from e


def translate_jp_to_kr(jp_text: str) -> str:
    client = _get_client()
    try:
        response, _ = _generate(client, _TRANSLATE_PROMPT.format(jp_text=jp_text),
                                types.GenerateContentConfig(temperature=0.2))
        return response.text.strip()
    except Exception as e:
        raise GeminiUnavailableError(f"翻訳に失敗しました: {e}") from e
