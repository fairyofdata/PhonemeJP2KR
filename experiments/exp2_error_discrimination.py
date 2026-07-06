"""Experiment 2 — Discriminant validity via synthetic error injection.

For each target sentence we synthesize two audio clips with the same
neural TTS voice: (a) the correct sentence and (b) an error-injected
version encoding typical Japanese-L1 mispronunciations (vowel
epenthesis, ㅓ→ㅗ substitution, coda repair, …). Both clips run through
the full pipeline (ffmpeg → Wav2Vec2 → G2P → jamo alignment), and a
valid scorer must (1) score correct audio near 100 and (2) rank every
correct clip above its error-injected counterpart.

This is a controlled perturbation study, not an L2-speech evaluation:
TTS renders segmental errors cleanly, without genuine accent prosody.
It validates pipeline sensitivity, complementing — not replacing — a
human-rater correlation study (see docs/EVALUATION.md, future work).

Usage:  python experiments/exp2_error_discrimination.py
Output: experiments/results/exp2_error_discrimination.json + console table
"""

import json
import os
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import imageio_ffmpeg  # noqa: E402

from src.asr import load_wav2vec_model, transcribe_acoustics  # noqa: E402
from src.scoring import score_pronunciation  # noqa: E402
from src.tts import generate_tts_audio  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "exp2_error_discrimination.json")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# (target, error-injected version, dominant simulated error pattern)
PAIRS = [
    ("감사합니다", "감사하무니다", "vowel_epenthesis"),
    ("안녕하세요", "안뇽하세요", "vowel_substitution"),
    ("서울에 갑니다", "소우루에 가무니다", "ʌ→o + epenthesis"),
    ("한국어를 공부합니다", "한구고루 고부하무니다", "multiple"),
    ("만나서 반갑습니다", "만나소 반가푸스무니다", "severe multiple"),
    ("물 주세요", "무루 주세요", "coda repair"),
    ("괜찮아요", "겐차나요", "vowel simplification"),
    ("어디에 있어요", "오디에 이소요", "ʌ→o + tense loss"),
    ("맛있어요", "마시소요", "tense loss + ʌ→o"),
    ("저는 학생입니다", "조누는 학세이무니다", "multiple"),
]


def synthesize_wav(text: str, workdir: str, tag: str) -> str:
    mp3 = os.path.join(workdir, f"{tag}.mp3")
    wav = os.path.join(workdir, f"{tag}.wav")
    if not generate_tts_audio(text, "SunHi", mp3):
        raise RuntimeError(f"TTS failed for: {text}")
    subprocess.run([FFMPEG, "-y", "-i", mp3, "-ar", "16000", "-ac", "1", wav],
                   check=True, capture_output=True)
    return wav


def main():
    processor, model = load_wav2vec_model()
    rows = []
    with tempfile.TemporaryDirectory() as workdir:
        for i, (target, injected, pattern) in enumerate(PAIRS):
            wav_ok = synthesize_wav(target, workdir, f"{i}_ok")
            wav_err = synthesize_wav(injected, workdir, f"{i}_err")
            hyp_ok = transcribe_acoustics(wav_ok, processor, model)
            hyp_err = transcribe_acoustics(wav_err, processor, model)
            score_ok = score_pronunciation(target, hyp_ok).score
            score_err = score_pronunciation(target, hyp_err).score
            rows.append({
                "target": target, "injected": injected, "pattern": pattern,
                "asr_ok": hyp_ok, "asr_err": hyp_err,
                "score_ok": score_ok, "score_err": score_err,
                "correctly_ranked": score_ok > score_err,
            })
            print(f"{target}: correct={score_ok} error={score_err}"
                  f" ({'OK' if score_ok > score_err else 'MISRANKED'})"
                  f" | asr_err={hyp_err}")

    ok_scores = [r["score_ok"] for r in rows]
    err_scores = [r["score_err"] for r in rows]
    summary = {
        "n_pairs": len(rows),
        "pairwise_ranking_accuracy": sum(r["correctly_ranked"] for r in rows) / len(rows),
        "correct_audio": {"mean": round(statistics.mean(ok_scores), 1),
                          "stdev": round(statistics.stdev(ok_scores), 1),
                          "min": min(ok_scores), "max": max(ok_scores)},
        "error_audio": {"mean": round(statistics.mean(err_scores), 1),
                        "stdev": round(statistics.stdev(err_scores), 1),
                        "min": min(err_scores), "max": max(err_scores)},
        "mean_gap": round(statistics.mean(ok_scores) - statistics.mean(err_scores), 1),
        "pairs": rows,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(f"pairwise ranking accuracy: {summary['pairwise_ranking_accuracy']:.0%}")
    print(f"correct audio: {summary['correct_audio']['mean']} ± {summary['correct_audio']['stdev']}")
    print(f"error audio  : {summary['error_audio']['mean']} ± {summary['error_audio']['stdev']}")
    print(f"mean gap     : {summary['mean_gap']}")


if __name__ == "__main__":
    main()
