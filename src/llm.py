"""Gemini-based pedagogical feedback and JP→KR translation.

Design principle: the LLM never computes measurements. IPA transcriptions,
the phoneme score, and the error alignment all come from the deterministic
G2P/scoring pipeline and are passed *into* the prompt as evidence. The LLM
is used only for what it is good at — interpreting structured evidence,
rendering the learner's output as katakana to visualize L1 interference,
and writing natural pedagogical feedback in Japanese.
"""

import json

from google import genai
from google.genai import types

from .config import GEMINI_MODEL_ID, get_gemini_api_key

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
- 自動検出されたエラータグ (jamoアライメントに基づく): {error_tags}

[日本語母語話者の典型的エラーパターン (参考)]
1. 母音挿入 (Epenthesis): モーラ拍リズムの影響でパッチムの後に /ɯ/ や /u/ を挿入する。
2. 三項対立の混同: 平音・激音・濃音を有声/無声の二項対立で代替する。
3. 母音の歪み: /ʌ/(ㅓ) を /o/ で、/ɯ/(ㅡ) を /u/ で代替する。
4. 終声の弁別失敗: /n/, /ŋ/, /m/ のパッチムを日本語の「ん」に統合する。

[タスク]
上記のエラータグと2つのASR結果の差分を根拠として、以下のJSONのみを出力してください。

{{
  "katakana": "Wav2Vec2が認識した音（実際の発音）を、日本人がカタカナで発音したかのように表記した文字列。L1干渉の可視化用。",
  "error_summary": "検出された誤りの要点を1〜2文の日本語で。誤りがなければその旨を書く。",
  "feedback_jp": "エラータグごとに、口・舌・喉の使い方まで踏み込んだ具体的な矯正アドバイス（日本語、マークダウン使用可、3〜6文程度）。エビデンス（どの音がどう変わったか）を必ず引用すること。"
}}"""

_TRANSLATE_PROMPT = (
    "次の日本語の文を、発音練習に適した自然な話し言葉の韓国語に翻訳してください。"
    "訳文の韓国語1文のみを出力し、引用符・説明・マークダウンは一切含めないでください。\n\n"
    "日本語: {jp_text}"
)


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


def generate_feedback(target: str, target_surface: str, target_ipa: str,
                      whisper_text: str, wav2vec_text: str, actual_ipa: str,
                      score: int, error_tags: list) -> dict:
    """Interpret the deterministic analysis and return katakana + coaching text."""
    prompt = _FEEDBACK_PROMPT.format(
        target=target,
        target_surface=target_surface,
        target_ipa=target_ipa,
        whisper_text=whisper_text,
        wav2vec_text=wav2vec_text,
        actual_ipa=actual_ipa,
        score=score,
        error_tags=json.dumps(error_tags, ensure_ascii=False),
    )
    client = _get_client()
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        raise GeminiUnavailableError(f"Gemini API呼び出しに失敗しました: {e}") from e


def translate_jp_to_kr(jp_text: str) -> str:
    client = _get_client()
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=_TRANSLATE_PROMPT.format(jp_text=jp_text),
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return response.text.strip()
    except Exception as e:
        raise GeminiUnavailableError(f"翻訳に失敗しました: {e}") from e
