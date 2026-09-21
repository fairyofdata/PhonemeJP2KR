"""Native-speaker reference audio via Microsoft Edge neural TTS."""

import asyncio
import os

import edge_tts

VOICES = {
    "SunHi": "ko-KR-SunHiNeural",
    "InJoon": "ko-KR-InJoonNeural",
    "Hyunsu": "ko-KR-HyunsuMultilingualNeural",
}
DEFAULT_VOICE = "SunHi"


async def _generate(text: str, voice: str, output_path: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def generate_tts_audio(text: str, voice_name: str, output_path: str) -> bool:
    voice_id = VOICES.get(voice_name, VOICES[DEFAULT_VOICE])
    try:
        asyncio.run(_generate(text, voice_id, output_path))
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False


# admin text input (app.py): Japanese voices read a kana script so the
# check audio carries a Japanese speaker's mora timing and voicing
ADMIN_VOICES = {
    "Nanami (ja)": "ja-JP-NanamiNeural",
    "Keita (ja)": "ja-JP-KeitaNeural",
    **{f"{k} (ko)": v for k, v in VOICES.items()},
}


def synthesize(text: str, voice_id: str, output_path: str) -> bool:
    """Like generate_tts_audio, but with an explicit Edge voice id.

    A failed request (e.g. a Japanese voice given Hangul) leaves an empty
    file behind; it is removed so a cache check never picks it up.
    """
    try:
        asyncio.run(_generate(text, voice_id, output_path))
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
