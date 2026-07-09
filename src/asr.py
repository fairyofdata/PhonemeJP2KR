"""Dual-ASR pipeline: Whisper (intelligibility) + Wav2Vec2-CTC (acoustics).

The two models are deliberately asymmetric:

  - Whisper carries a strong internal language model, so its output
    approximates what a native listener *understands* after their brain
    auto-corrects the signal with context (intelligibility channel).
  - Wav2Vec2 with CTC decoding has no language model, so its output stays
    close to the raw phone sequence actually produced (acoustics channel).

The gap between the two channels quantifies the perception/production
mismatch that L2 learners cannot hear themselves.
"""

import torch
import librosa
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

from .config import WHISPER_MODEL_ID, WAV2VEC_MODEL_ID, AUDIO_SAMPLE_RATE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_whisper_model():
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID).to(DEVICE)
    model.eval()
    return processor, model


def load_wav2vec_model():
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC_MODEL_ID).to(DEVICE)
    model.eval()
    return processor, model


def _load_audio(audio_path: str):
    audio, _ = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
    return audio


def transcribe_intelligibility(audio_path: str, processor, model) -> str:
    """Whisper transcription — what a native listener would perceive."""
    audio = _load_audio(audio_path)
    inputs = processor(audio, return_tensors="pt", sampling_rate=AUDIO_SAMPLE_RATE)
    input_features = inputs.input_features.to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(
            input_features, language="korean", task="transcribe"
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def transcribe_acoustics(audio_path: str, processor, model) -> tuple[str, list[tuple[str, float, float]]]:
    """Wav2Vec2-CTC greedy decoding — returns (text, [(char, start_time, end_time), ...])."""
    audio = _load_audio(audio_path)
    inputs = processor(audio, return_tensors="pt", sampling_rate=AUDIO_SAMPLE_RATE)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    outputs = processor.batch_decode(predicted_ids, output_char_offsets=True)
    text = outputs.text[0].strip()
    
    frame_duration = 0.02  # 320 / 16000 for standard wav2vec2
    char_timestamps = []
    if hasattr(outputs, "char_offsets") and outputs.char_offsets and outputs.char_offsets[0]:
        for char_info in outputs.char_offsets[0]:
            char = char_info["char"]
            start_time = float(char_info["start_offset"]) * frame_duration
            end_time = float(char_info["end_offset"]) * frame_duration
            char_timestamps.append((char, start_time, end_time))
            
    return text, char_timestamps
