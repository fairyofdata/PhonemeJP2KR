import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
from langdetect import detect
from openai import OpenAI

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load models (placeholder for now)
def load_whisper_model():
    # Load Whisper model for intended pronunciation
    # large-v3는 로컬에서 너무 무거우므로 프로토타이핑용으로 small을 권장합니다.
    model_id = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
    return processor, model

def load_wav2vec_model():
    # 일반 한국어 Wav2Vec2 모델 로드 (일본인 타겟팅은 LLM 프롬프트가 담당)
    model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-korean"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(DEVICE)
    return processor, model

# Function to transcribe audio with Whisper
def transcribe_audio(audio_path, processor, model):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    # Process and generate transcription
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

# LLM 기반 피드백 생성을 위한 뼈대
def generate_feedback(intended, actual, api_key=None):
    if not api_key:
        return "⚠️ OpenAI API 키가 제공되지 않았습니다. 사이드바에 API 키를 입력해주세요.\n\n**[Mock Response]**\n의도하신 발음이 한국어 인식기에는 다르게 들렸습니다. API 키를 연결하면 상세한 교정 피드백이 제공됩니다."
        
    prompt = f"""당신은 일본인 학습자를 위한 한국어 발음 교정 전문가입니다.
    학습자가 의도한 발음: {intended}
    일반 한국어 음성인식기가 실제 인식한 발음: {actual}
    
    한국어 음성인식기가 원래 의도와 다르게 '{actual}'이라고 인식했다면, 이는 일본어 모어 화자 특유의 발음 습관(예: 자음 끝에 불필요한 모음 'ㅜ/ㅗ' 추가, 받침 발음 누락, 평음/격음/경음 혼동 등) 때문일 확률이 높습니다.
    위 두 발음을 비교하여:
    1. 일본인 학습자가 어떤 부분에서 특유의 발음 실수를 했는지 분석하세요.
    2. 어떻게 입모양이나 혀의 위치를 바꿔야 정확한 한국어 발음이 되는지 구체적이고 친절한 교정 피드백을 제공하세요.
    """
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 비용 효율적이고 빠른 최신 모델
            messages=[
                {"role": "system", "content": "당신은 친절하고 전문적인 한국어 발음 교정 선생님입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API 호출 중 오류가 발생했습니다: {str(e)}"