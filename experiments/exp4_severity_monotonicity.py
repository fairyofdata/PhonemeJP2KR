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

Usage:
  python experiments/exp4_severity_monotonicity.py           # TTS + ASR (local)
  python experiments/exp4_severity_monotonicity.py --check   # scoring only (CI)

The first form synthesizes and recognizes the 20 clips, saves the ASR
hypotheses to experiments/data/exp4_asr_fixture.json, and prints the
result next to the one recorded in experiments/results/ — it never
overwrites the recorded result. --check re-scores the saved hypotheses
with the current scorer (no TTS, no ASR, no torch) and fails if ρ or the
monotonic step rate fall outside the tolerance around the recorded value.
"""

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.statsutil import bootstrap_ci, spearman_rho  # noqa: E402
from src.scoring import score_pronunciation  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "exp4_severity_monotonicity.json")
FIXTURE = os.path.join(os.path.dirname(__file__), "data", "exp4_asr_fixture.json")

# --check tolerance around the fixture's own baseline (see DECISIONS 16).
# The fixture stores the numbers its hypotheses gave when it was made; a
# scorer change may move them by at most this much before CI fails.
RHO_SLACK = 0.05        # fail if rho weakens by more than 0.05
MONOTONE_SLACK = 1 / 15  # fail if more than one extra step (of 15) breaks

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
    import imageio_ffmpeg
    from src.tts import generate_tts_audio

    mp3 = os.path.join(workdir, f"{tag}.mp3")
    wav = os.path.join(workdir, f"{tag}.wav")
    if not generate_tts_audio(text, "SunHi", mp3):
        raise RuntimeError(f"TTS failed for: {text}")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", mp3, "-ar", "16000",
                    "-ac", "1", wav], check=True, capture_output=True)
    return wav


def summarize(ladders) -> dict:
    """Score saved hypotheses → the statistics of the experiment.

    ``ladders``: [{"target": …, "hyps": [severity 0..3 ASR text]}].
    """
    severities, scores, rows = [], [], []
    for ladder in ladders:
        ladder_scores = [score_pronunciation(ladder["target"], h).score for h in ladder["hyps"]]
        severities += list(range(len(ladder_scores)))
        scores += ladder_scores
        violations = sum(1 for a, b in zip(ladder_scores, ladder_scores[1:]) if b > a)
        rows.append({"target": ladder["target"], "scores": ladder_scores,
                     "violations": violations})
    rho = spearman_rho(severities, scores)
    ci_lo, ci_hi = bootstrap_ci(severities, scores, spearman_rho)
    total_steps = sum(len(r["scores"]) - 1 for r in rows)
    return {
        "n_clips": len(scores),
        "spearman_rho": round(rho, 3),
        "bootstrap_95ci": [round(ci_lo, 3), round(ci_hi, 3)],
        "mean_score_by_severity": {
            str(sev): round(statistics.mean(s for v, s in zip(severities, scores) if v == sev), 1)
            for sev in sorted(set(severities))},
        "monotonic_step_rate": round(1 - sum(r["violations"] for r in rows) / total_steps, 3),
        "ladders": rows,
    }


def _recorded() -> dict:
    with open(OUT, encoding="utf-8") as f:
        return json.load(f)


def record():
    """TTS + ASR for every clip → save the hypotheses; compare, don't overwrite."""
    from src.asr import load_wav2vec_model, transcribe_acoustics

    processor, model = load_wav2vec_model()
    ladders = []
    with tempfile.TemporaryDirectory() as workdir:
        for li, ladder in enumerate(LADDERS):
            texts = [ladder["target"]] + ladder["steps"]
            hyps = [transcribe_acoustics(synthesize_wav(t, workdir, f"{li}_{k}"),
                                         processor, model)[0]
                    for k, t in enumerate(texts)]
            ladders.append({"target": ladder["target"], "texts": texts, "hyps": hyps})
            print(f"{ladder['target']}: {hyps}")
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"voice": "SunHi", "ladders": ladders}, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"hypotheses saved: {FIXTURE}")
    _report(summarize(ladders), _recorded())


def _report(now: dict, recorded: dict):
    print("\n=== this run vs the recorded result (not overwritten) ===")
    for key in ("spearman_rho", "bootstrap_95ci", "mean_score_by_severity", "monotonic_step_rate"):
        mark = "" if now[key] == recorded[key] else "   ← differs"
        print(f"{key}: now {now[key]}   recorded {recorded[key]}{mark}")
    for n, r in zip(now["ladders"], recorded["ladders"]):
        mark = "" if n["scores"] == r["scores"] else "   ← differs"
        print(f"  {n['target']}: now {n['scores']}   recorded {r['scores']}{mark}")


def check() -> int:
    """Re-score the saved hypotheses; 0 if within tolerance of the fixture's baseline."""
    if not os.path.exists(FIXTURE):
        print(f"missing {FIXTURE} — create it once, locally, with:")
        print("  python experiments/exp4_severity_monotonicity.py")
        return 1
    with open(FIXTURE, encoding="utf-8") as f:
        fixture = json.load(f)
    now = summarize(fixture["ladders"])
    _report(now, _recorded())
    base = fixture["baseline"]
    rho_max = round(base["spearman_rho"] + RHO_SLACK, 3)
    mono_min = round(base["monotonic_step_rate"] - MONOTONE_SLACK, 3)
    ok = now["spearman_rho"] <= rho_max and now["monotonic_step_rate"] >= mono_min
    print(f"\ncheck (fixture of {fixture.get('created', '?')}, baseline rho "
          f"{base['spearman_rho']}, monotone {base['monotonic_step_rate']}): "
          f"rho {now['spearman_rho']} (≤ {rho_max}), monotone "
          f"{now['monotonic_step_rate']} (≥ {mono_min}) → {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv[1:] else record())
