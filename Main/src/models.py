import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
from langdetect import detect
import google.generativeai as genai
import os
import asyncio
import edge_tts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_whisper_model():
    model_id = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
    return processor, model

def load_wav2vec_model():
    # 일반 한국어 Wav2Vec2 모델 로드 (일본인 타겟팅은 LLM 프롬프트가 담당)
    model_id = "kresnik/wav2vec2-large-xlsr-korean"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(DEVICE)
    return processor, model

def transcribe_audio(audio_path, processor, model):
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, return_tensors="pt", sampling_rate=sr)
    input_features = inputs.input_features.to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(input_features, language="korean", task="transcribe")
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

def analyze_pronunciation(audio_path, processor, model):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    # Process and get transcribed text (한글로 인식된 실제 발음 소리)
    inputs = processor(audio, return_tensors="pt", sampling_rate=sr)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Language detection
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Pure Python Levenshtein (Windows 빌드 에러 우회)
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_phoneme_score(intended_ipa, actual_ipa):
    if not intended_ipa or not actual_ipa:
        return 0
    distance = levenshtein_distance(intended_ipa, actual_ipa)
    max_len = max(len(intended_ipa), len(actual_ipa))
    if max_len == 0:
        return 100
    score = max(0, 100 - (distance / max_len * 100))
    return round(score)

import json

# LLM 기반 피드백 및 IPA, Katakana 추출
def generate_feedback(target, whisper_text, wav2vec_text):
    prompt = f"""あなたは親切で専門的な韓国語の発音矯正の先生であり、日本人学習者（Japanese Native Speakers）の母語干渉（L1 Interference）を深く理解している言語学の専門家です。

[分析データ]
- 学習者が意図した文章 (Target): {target}
- Whisperが認識した文章 (Intelligibility - 韓国人の体感): {whisper_text}
- Wav2Vec2が認識した音声 (Acoustics - 物理的な音声): {wav2vec_text}

[日本人話者の最適化ガイドライン (Linguistic Rules)]
日本語はモーラ（Mora）拍言語であり、終声（パッチム）が制限されているため、韓国語の発音時に典型的なエラーが発生します。
1. 母音挿入 (Epenthesis): パッチムの後に不要な母音 /u/ や /o/ を追加する。
2. 破裂音/摩擦音の混同: 平音、激音、硬音を区別できず、有声音/無声音としてのみ区別する。
3. 母音の歪み: /ʌ/（オ）を /o/ や /a/ で発音する。
4. 鼻音化エラー: 終声 /ŋ/ の発音を明確に切れない。

これらのガイドラインと上記の3つのデータを基に、以下の項目を分析し、**必ず有効なJSONフォーマットのみ**で応答してください。他の説明は一切加えないでください。

{{
    "target_ipa": "目標文章の国際音声記号(IPA)",
    "whisper_ipa": "Whisperが認識した文章の国際音声記号(IPA)",
    "wav2vec_ipa": "Wav2Vec2が認識した音声の国際音声記号(IPA)",
    "katakana": "Wav2Vec2の音声を日本人が発音したかのようにカタカナで表記（L1干渉を視覚化するため）",
    "feedback_jp": "上記の指標とL1干渉ルールに基づき、日本語で作成された詳細な発音矯正フィードバック（マークダウン使用可）"
}}
"""
    
    try:
        key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'vertex_key_new.md')
        with open(key_path, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
            
        genai.configure(api_key=api_key)
        
        # 最新で安定している 2.5-flash モデルを使用
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2}
        )
        
        # JSON 파싱
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:-3].strip()
            
        result = json.loads(response_text)
        return result
    except Exception as e:
        return {
            "target_ipa": "N/A",
            "whisper_ipa": "N/A",
            "wav2vec_ipa": "N/A",
            "katakana": "N/A",
            "feedback_jp": f"API 呼び出し中にエラーが発生しました: {str(e)}\n\nAPIキーを確認してください。"
        }

def translate_jp_to_kr(jp_text):
    prompt = f"다음 일본어 문장을 자연스러운 회화체 한국어로 번역하세요. 결과는 번역된 한국어 문장 단 하나만 출력해야 하며, 따옴표나 다른 부가 설명, 마크다운 형식 등은 일절 포함하지 마세요.\n\n일본어: {jp_text}"
    try:
        key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'vertex_key_new.md')
        with open(key_path, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2}
        )
        
        return response.text.strip()
    except Exception as e:
        return f"번역 오류: {str(e)}"

async def _generate_tts(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def generate_tts_audio(text, voice_name, output_path):
    voice_map = {
        "SunHi": "ko-KR-SunHiNeural",
        "InJoon": "ko-KR-InJoonNeural",
        "JiMin": "ko-KR-JiMinNeural"
    }
    voice_id = voice_map.get(voice_name, "ko-KR-SunHiNeural")
    try:
        asyncio.run(_generate_tts(text, voice_id, output_path))
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False