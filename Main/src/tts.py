import os
import asyncio
import edge_tts

async def _generate_tts(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def generate_tts_audio(text, voice_name, output_path):
    voice_map = {
        "SunHi": "ko-KR-SunHiNeural",
        "InJoon": "ko-KR-InJoonNeural",
        "Hyunsu": "ko-KR-HyunsuMultilingualNeural"
    }
    voice_id = voice_map.get(voice_name, "ko-KR-SunHiNeural")
    try:
        asyncio.run(_generate_tts(text, voice_id, output_path))
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False
