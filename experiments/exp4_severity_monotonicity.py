"""Experiment 4 — Graded severity monotonicity (human-rating surrogate).

A human-rater correlation study needs L2 recordings we do not yet have
(see docs/HUMAN_EVAL_PROTOCOL.md for that pipeline). This experiment is
the strongest validation available without them: the *number of injected
segmental errors* serves as a controlled ground-truth severity ordinal.

For each base sentence we synthesize four TTS clips at severity 0-3,
where severity k applies the first k cumulative error injections (all
drawn from attested Japanese-L1 patterns). A valid scorer must decrease
monotonically with severity. We report Spearman rho between severity and
system score (with bootstrap CI) and the per-sentence monotonicity rate.

Caveat: like Experiment 2 this is a perturbation study on TTS audio, not
genuine L2 speech; it validates ordinal sensitivity, not absolute
calibration against human judgment.

Usage:  python experiments/exp4_severity_monotonicity.py
Output: experiments/results/exp4_severity_monotonicity.json + console table
"""

import json
import os
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import imageio_ffmpeg  # noqa: E402

from experiments.statsutil import bootstrap_ci, spearman_rho  # noqa: E402
from src.asr import load_wav2vec_model, transcribe_acoustics  # noqa: E402
from src.scoring import score_pronunciation  # noqa: E402
from src.tts import generate_tts_audio  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "exp4_severity_monotonicity.json")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# severity 0 = target; severity k = k cumulative injected L1 errors
LADDERS = [
    {
        "target": "감사합니다",
        "steps": ["감사하무니다",      # +epenthesis (ㅁ→무)
                  "캄사하무니다",      # +laryngeal (ㄱ→ㅋ)
                  "캄사하무니다스"],   # +final epenthesis (다→다스)
    },
    {
        "target": "서울에서 만나요",
        "steps": ["소울에서 만나요",    # +ʌ→o
                  "소우루에서 만나요",  # +epenthesis (ㄹ→루)
                  "소우루에소 만나요"],  # +ʌ→o
    },
    {
        "target": "도서관에 갑니다",
        "steps": ["도서관에 가무니다",  # +epenthesis
                  "도소관에 가무니다",  # +ʌ→o
                  "도소콴에 가무니다"],  # +laryngeal (ㄱ→ㅋ)
    },
    {
        "target": "비빔밥을 먹었어요",
        "steps": ["비빔바부 먹었어요",  # +coda repair (밥을→바부)
                  "비빔바부 머거서요",  # +tense loss (ㅆ→ㅅ)
                  "피빔바부 머거서요"],  # +laryngeal (ㅂ→ㅍ)
    },
    {
        "target": "저는 학생입니다",
        "steps": ["저는 학생이무니다",  # +epenthesis
                  "조는 학생이무니다",  # +ʌ→o
                  "조는 학세이이무니다"],  # +coda ŋ loss (생→세이)
    },
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
    severities, scores, rows = [], [], []

    with tempfile.TemporaryDirectory() as workdir:
        for li, ladder in enumerate(LADDERS):
            texts = [ladder["target"]] + ladder["steps"]
            ladder_scores = []
            for sev, text in enumerate(texts):
                wav = synthesize_wav(text, workdir, f"{li}_{sev}")
                hyp = transcribe_acoustics(wav, processor, model)[0]
                score = score_pronunciation(ladder["target"], hyp).score
                ladder_scores.append(score)
                severities.append(sev)
                scores.append(score)
            violations = sum(
                1 for a, b in zip(ladder_scores, ladder_scores[1:]) if b > a
            )
            rows.append({"target": ladder["target"], "scores": ladder_scores,
                         "violations": violations})
            print(f"{ladder['target']}: {ladder_scores}"
                  f" ({violations} monotonicity violation(s))")

    rho = spearman_rho(severities, scores)
    ci_lo, ci_hi = bootstrap_ci(severities, scores, spearman_rho)
    by_severity = {
        sev: round(statistics.mean(
            s for v, s in zip(severities, scores) if v == sev), 1)
        for sev in sorted(set(severities))
    }
    total_steps = sum(len(r["scores"]) - 1 for r in rows)
    total_violations = sum(r["violations"] for r in rows)

    summary = {
        "n_clips": len(scores),
        "spearman_rho": round(rho, 3),
        "bootstrap_95ci": [round(ci_lo, 3), round(ci_hi, 3)],
        "mean_score_by_severity": by_severity,
        "monotonic_step_rate": round(1 - total_violations / total_steps, 3),
        "ladders": rows,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(f"Spearman rho (severity vs score): {rho:.3f}"
          f" [95% CI {ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"mean score by severity: {by_severity}")
    print(f"monotonic step rate: {summary['monotonic_step_rate']:.0%}")


if __name__ == "__main__":
    main()
