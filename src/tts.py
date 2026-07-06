"""Native-speaker reference audio via Microsoft Edge neural TTS."""

import asyncio

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
