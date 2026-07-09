"""Experiment 5 — Human-rater correlation study (analysis harness).

This is the decisive validity test: do system scores agree with native
Korean listeners' judgments of real L2 speech? Recordings are not yet
collected — docs/HUMAN_EVAL_PROTOCOL.md specifies the full collection
protocol (materials, participants, rating rubric, file naming). This
script is the ready-to-run analysis half: once `manifest.csv` and
`ratings.csv` exist, a single command produces the correlation report.

Inputs (default under experiments/data/human_eval/):
  manifest.csv  audio_file,speaker_id,target_text
  ratings.csv   audio_file,rater_id,rating        (rating: 1-5 integer)

Outputs:
  experiments/results/exp5_human_correlation.json
  - Spearman/Pearson between system score and mean human rating,
    with bootstrap 95% CIs
  - inter-rater reliability (mean pairwise Spearman across raters)
  - per-speaker breakdown

Usage:
  python experiments/exp5_human_correlation.py                # real run
  python experiments/exp5_human_correlation.py --selftest     # harness check
                                                (synthesizes TTS clips and
                                                 dummy ratings, no real data)
"""

import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import imageio_ffmpeg  # noqa: E402

from experiments.statsutil import bootstrap_ci, pearson_r, spearman_rho  # noqa: E402
from src.asr import load_wav2vec_model, transcribe_acoustics  # noqa: E402
from src.scoring import score_pronunciation  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "human_eval")
OUT = os.path.join(os.path.dirname(__file__), "results", "exp5_human_correlation.json")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [row for row in csv.DictReader(f)]


def to_wav16k(audio_path: str, workdir: str, tag: str) -> str:
    wav = os.path.join(workdir, f"{tag}.wav")
    subprocess.run([FFMPEG, "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav],
                   check=True, capture_output=True)
    return wav


def inter_rater_reliability(ratings_rows):
    """Mean pairwise Spearman across raters, over commonly rated clips."""
    by_rater = defaultdict(dict)
    for row in ratings_rows:
        by_rater[row["rater_id"]][row["audio_file"]] = float(row["rating"])
    raters = sorted(by_rater)
    rhos = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            common = sorted(set(by_rater[raters[i]]) & set(by_rater[raters[j]]))
            if len(common) >= 3:
                rhos.append(spearman_rho(
                    [by_rater[raters[i]][c] for c in common],
                    [by_rater[raters[j]][c] for c in common],
                ))
    return round(statistics.mean(rhos), 3) if rhos else None


def analyze(manifest, ratings_rows, audio_dir):
    mean_rating = defaultdict(list)
    for row in ratings_rows:
        mean_rating[row["audio_file"]].append(float(row["rating"]))
    mean_rating = {k: statistics.mean(v) for k, v in mean_rating.items()}

    processor, model = load_wav2vec_model()
    clips = []
    with tempfile.TemporaryDirectory() as workdir:
        for i, row in enumerate(manifest):
            fname = row["audio_file"]
            if fname not in mean_rating:
                print(f"  warning: no ratings for {fname}, skipping")
                continue
            wav = to_wav16k(os.path.join(audio_dir, fname), workdir, str(i))
            hyp = transcribe_acoustics(wav, processor, model)[0]
            score = score_pronunciation(row["target_text"], hyp).score
            clips.append({
                "audio_file": fname, "speaker_id": row["speaker_id"],
                "target": row["target_text"], "asr": hyp,
                "system_score": score,
                "mean_human_rating": round(mean_rating[fname], 2),
            })
            print(f"{fname}: system={score} human={mean_rating[fname]:.2f}")

    if len(clips) < 5:
        raise SystemExit("Need at least 5 rated clips for a correlation.")

    xs = [c["system_score"] for c in clips]
    ys = [c["mean_human_rating"] for c in clips]
    rho = spearman_rho(xs, ys)
    rho_ci = bootstrap_ci(xs, ys, spearman_rho)
    r = pearson_r(xs, ys)
    r_ci = bootstrap_ci(xs, ys, pearson_r)

    per_speaker = {}
    for spk in sorted({c["speaker_id"] for c in clips}):
        sub = [c for c in clips if c["speaker_id"] == spk]
        per_speaker[spk] = {
            "n": len(sub),
            "mean_system_score": round(statistics.mean(
                c["system_score"] for c in sub), 1),
            "mean_human_rating": round(statistics.mean(
                c["mean_human_rating"] for c in sub), 2),
        }

    return {
        "n_clips": len(clips),
        "spearman_rho": round(rho, 3),
        "spearman_95ci": [round(v, 3) for v in rho_ci],
        "pearson_r": round(r, 3),
        "pearson_95ci": [round(v, 3) for v in r_ci],
        "inter_rater_reliability": inter_rater_reliability(ratings_rows),
        "per_speaker": per_speaker,
        "clips": clips,
    }


def build_selftest_data(workdir):
    """Synthesize a tiny fake study (TTS clips + dummy ratings) to verify
    the harness end-to-end. Results are meaningless by construction."""
    from src.tts import generate_tts_audio
    items = [  # (filename, spoken text, target, dummy ratings)
        ("spk1_s1.mp3", "감사합니다", "감사합니다", [5, 5, 4]),
        ("spk1_s2.mp3", "감사하무니다", "감사합니다", [3, 2, 3]),
        ("spk2_s1.mp3", "안녕하세요", "안녕하세요", [5, 4, 4]),
        ("spk2_s2.mp3", "소울에서 만나요", "서울에서 만나요", [3, 3, 2]),
        ("spk3_s1.mp3", "캄사하무니다스", "감사합니다", [1, 2, 1]),
    ]
    manifest, ratings = [], []
    for fname, spoken, target, rates in items:
        assert generate_tts_audio(spoken, "SunHi", os.path.join(workdir, fname))
        manifest.append({"audio_file": fname,
                         "speaker_id": fname.split("_")[0],
                         "target_text": target})
        for ri, rating in enumerate(rates, 1):
            ratings.append({"audio_file": fname, "rater_id": f"rater{ri}",
                            "rating": str(rating)})
    return manifest, ratings


def main():
    if "--selftest" in sys.argv:
        print("Running harness self-test with synthetic data...")
        with tempfile.TemporaryDirectory() as workdir:
            manifest, ratings_rows = build_selftest_data(workdir)
            summary = analyze(manifest, ratings_rows, workdir)
        summary["selftest"] = True
    else:
        manifest = read_csv(os.path.join(DATA_DIR, "manifest.csv"))
        ratings_rows = read_csv(os.path.join(DATA_DIR, "ratings.csv"))
        summary = analyze(manifest, ratings_rows, os.path.join(DATA_DIR, "audio"))
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(f"n = {summary['n_clips']}")
    print(f"Spearman rho: {summary['spearman_rho']}"
          f" [95% CI {summary['spearman_95ci'][0]}, {summary['spearman_95ci'][1]}]")
    print(f"Pearson r  : {summary['pearson_r']}"
          f" [95% CI {summary['pearson_95ci'][0]}, {summary['pearson_95ci'][1]}]")
    print(f"inter-rater reliability (mean pairwise rho): "
          f"{summary['inter_rater_reliability']}")


if __name__ == "__main__":
    main()
