# -*- coding: utf-8 -*-
"""Experiment 6 — Large-scale validation on real Japanese-L1 Korean speech.

Data: AI-Hub dataset 131 ("외국인 한국어 발화 음성 데이터", Japanese-L1
subset), Validation split. Each read-aloud utterance carries three human
signals we never had before:

  Reading            the script the learner was asked to read (target)
  ReadingLabelText   what native transcribers actually heard — including
                     misreadings, repeats, particle swaps (~20% deviate)
  SentenceSpeechLV   per-utterance speech-level rating 상/중/하

This gives two validation axes that exp5 (recruited raters, still pending)
was designed to provide, at three orders of magnitude more data:

  A. Deviation agreement — the system's acoustics-channel score against
     the jamo-level deviation that human transcribers heard
     (score of script vs ReadingLabelText). Primary: Spearman rho and
     the AUC of detecting transcriber-noted deviations from system score.
  B. Proficiency association — system score vs SentenceSpeechLV (ordinal)
     and, speaker-aggregated, vs TOPIK grade.

It also yields the number exp2/exp4 flagged as unknown: the ASR noise
floor — how far the acoustics channel strays from what humans heard even
when the reading was faithful (score of ReadingLabelText vs ASR output on
the faithful subset).

Caveats (stated up front, they matter for interpretation):
  - ReadingLabelText is orthographic; transcribers normalise phone-level
    accent into standard spelling, so axis A validates error *detection*,
    not fine phonetic scoring.
  - SentenceSpeechLV rates overall speech level (fluency included), not
    pronunciation alone; expect attenuated correlation.
  - Speakers skew proficient (TOPIK 5-6), so range restriction lowers all
    correlations relative to a balanced L2 population.

AI-Hub license forbids redistribution: extracted audio and the manifest
(which contains label text) live in experiments/data/l2_aihub/ (gitignored);
only aggregate results are committed.

Usage:
  python experiments/exp6_l2_validation.py build   # sample + extract corpus
  python experiments/exp6_l2_validation.py run     # ASR + scoring (resumable)
  python experiments/exp6_l2_validation.py analyze # stats -> results JSON
Options:
  --root PATH     AI-Hub dataset root (or set AIHUB131_ROOT)
  --per-level N   sample size per 상/중 stratum (default 200; 하 is taken whole)
  --limit N       process at most N clips in `run` (pilot runs)
"""

import argparse
import csv
import json
import os
import random
import re
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.statsutil import (  # noqa: E402
    bootstrap_ci, pearson_r, rank_auc, spearman_rho,
)

DEFAULT_ROOT = (
    r"C:\Users\Baek\Phomene"
    "\\131.인공지능 학습을 위한 외국인 한국어 발화 음성 데이터 (일본어 모어 화자)"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "l2_aihub")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
MANIFEST = os.path.join(DATA_DIR, "manifest.csv")
RUN_LOG = os.path.join(DATA_DIR, "run_results.jsonl")
OUT = os.path.join(os.path.dirname(__file__), "results", "exp6_l2_validation.json")

SEED = 131
LV_ORD = {"하": 1, "중": 2, "상": 3}


def zip_paths(root):
    base = os.path.join(root, "01.데이터_new_20220719", "2.Validation")
    return (
        os.path.join(base, "라벨링데이터", "3. 일본어.zip"),
        os.path.join(base, "원천데이터", "VS_3. 일본어.zip"),
    )


def fix_name(name):
    """AI-Hub zips store CP949 filenames that zipfile mis-decodes as CP437."""
    try:
        return name.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def clean_text(s):
    """Normalise a script/label: drop '(원표기)' glosses and punctuation."""
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^가-힣0-9a-zA-Z\s]", " ", s)
    return " ".join(s.split())


def cmd_build(root, per_level):
    label_zip, audio_zip = zip_paths(root)
    rng = random.Random(SEED)

    print("scanning labels...")
    records = []
    skipped = Counter()
    with zipfile.ZipFile(label_zip) as z:
        for info in z.infolist():
            if not info.filename.endswith(".json"):
                continue
            d = json.loads(z.read(info).decode("utf-8-sig"))
            t = d.get("transcription", {})
            script = clean_text((t.get("Reading") or "").strip())
            if not script:
                skipped["free_speech"] += 1
                continue
            if re.search(r"[0-9a-zA-Z]", script):
                skipped["non_hangul_target"] += 1
                continue
            lv = t.get("SentenceSpeechLV", "")
            if lv not in LV_ORD:
                skipped["no_level"] += 1
                continue
            label = clean_text((t.get("ReadingLabelText") or "").strip())
            if not label:
                skipped["no_label_text"] += 1
                continue
            records.append({
                "audio_file": d["fileName"],
                "speaker_id": d.get("SpeakerID", ""),
                "sentence_id": d.get("file_info", {}).get("sentenceID", ""),
                "level": lv,
                "topik": d.get("skill_info", {}).get("topikGrade", ""),
                "record_time": d.get("file_info", {}).get("recordTime", ""),
                "target_text": script,
                "heard_text": label,
                "deviates": int(script != label),
            })
    print(f"usable read-aloud records: {len(records)}  skipped: {dict(skipped)}")

    by_level = defaultdict(list)
    for r in records:
        by_level[r["level"]].append(r)
    sample = list(by_level["하"])
    for lv in ("중", "상"):
        pool = sorted(by_level[lv], key=lambda r: r["audio_file"])
        rng.shuffle(pool)
        sample.extend(pool[:per_level])
    print("sampled:", Counter(r["level"] for r in sample))

    print("extracting audio...")
    os.makedirs(AUDIO_DIR, exist_ok=True)
    wanted = {r["audio_file"] for r in sample}
    found = set()
    with zipfile.ZipFile(audio_zip) as z:
        members = {}
        for info in z.infolist():
            base = fix_name(info.filename).rsplit("/", 1)[-1]
            if base in wanted:
                members[base] = info
        for base, info in members.items():
            dest = os.path.join(AUDIO_DIR, base)
            if not (os.path.exists(dest) and os.path.getsize(dest) == info.file_size):
                with z.open(info) as src, open(dest, "wb") as out:
                    out.write(src.read())
            found.add(base)
    missing = wanted - found
    if missing:
        print(f"warning: {len(missing)} sampled clips missing from audio zip, dropped")
        sample = [r for r in sample if r["audio_file"] in found]

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(sample[0].keys()))
        w.writeheader()
        w.writerows(sample)
    print(f"manifest written: {MANIFEST}  n={len(sample)}")


def cmd_run(limit=None):
    from src.asr import load_wav2vec_model, transcribe_acoustics
    from src.scoring import score_pronunciation

    with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
        manifest = list(csv.DictReader(f))

    done = set()
    if os.path.exists(RUN_LOG):
        with open(RUN_LOG, encoding="utf-8") as f:
            done = {json.loads(line)["audio_file"] for line in f if line.strip()}
    todo = [r for r in manifest if r["audio_file"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(done)} done, {len(todo)} to run")
    if not todo:
        return

    processor, model = load_wav2vec_model()
    t0 = time.time()
    with open(RUN_LOG, "a", encoding="utf-8") as out:
        for k, row in enumerate(todo, 1):
            wav = os.path.join(AUDIO_DIR, row["audio_file"])
            try:
                hyp, _ = transcribe_acoustics(wav, processor, model)
            except Exception as e:  # keep the batch alive; log the failure
                print(f"  ERROR {row['audio_file']}: {e}")
                continue
            res = {
                "audio_file": row["audio_file"],
                "asr": hyp,
                "system_score": score_pronunciation(row["target_text"], hyp).score,
                "heard_score": score_pronunciation(row["target_text"],
                                                   row["heard_text"]).score,
                "asr_vs_heard": score_pronunciation(row["heard_text"], hyp).score,
            }
            out.write(json.dumps(res, ensure_ascii=False) + "\n")
            out.flush()
            if k % 10 == 0 or k == len(todo):
                rate = (time.time() - t0) / k
                print(f"  {k}/{len(todo)}  {rate:.1f}s/clip  "
                      f"eta {rate * (len(todo) - k) / 60:.0f}min")


def cmd_analyze():
    with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
        meta = {r["audio_file"]: r for r in csv.DictReader(f)}
    rows = []
    with open(RUN_LOG, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["audio_file"] in meta:
                rows.append({**meta[r["audio_file"]], **r})
    print(f"analyzing {len(rows)} scored clips")

    sys_scores = [r["system_score"] for r in rows]
    heard = [r["heard_score"] for r in rows]
    lv = [LV_ORD[r["level"]] for r in rows]
    # deviation = transcribers heard something jamo-level different from the
    # script (heard_score < 100); the manifest's string-compare flag also
    # counts spacing-only differences, so it is not used here
    deviates = [int(r["heard_score"] < 100) for r in rows]

    # A. agreement with what human transcribers heard
    rho_heard = spearman_rho(sys_scores, heard)
    rho_heard_ci = bootstrap_ci(sys_scores, heard, spearman_rho)
    # deviating utterances should get LOWER system scores -> flip sign for AUC
    neg_scores = [-s for s in sys_scores]
    auc_dev = rank_auc(neg_scores, deviates)
    auc_dev_ci = bootstrap_ci(neg_scores, deviates, rank_auc)

    # B. proficiency association
    rho_lv = spearman_rho(sys_scores, lv)
    rho_lv_ci = bootstrap_ci(sys_scores, lv, spearman_rho)
    pair_auc = {}
    for hi, lo in (("상", "하"), ("상", "중"), ("중", "하")):
        sub = [r for r in rows if r["level"] in (hi, lo)]
        pair_auc[f"{hi}>{lo}"] = round(rank_auc(
            [r["system_score"] for r in sub],
            [r["level"] == hi for r in sub]), 3)

    # speaker-level vs TOPIK (numeric grades only)
    spk = defaultdict(list)
    for r in rows:
        spk[(r["speaker_id"], r["topik"])].append(r["system_score"])
    spk_scores, spk_topik = [], []
    for (sid, grade), scores in spk.items():
        if grade.isdigit() and len(scores) >= 3:
            spk_scores.append(statistics.mean(scores))
            spk_topik.append(int(grade))
    rho_topik = (spearman_rho(spk_scores, spk_topik)
                 if len(spk_scores) >= 5 else None)

    # ASR noise floor: faithful readings only (jamo-level faithful)
    faithful = [r for r in rows if r["heard_score"] == 100]
    floor = [r["asr_vs_heard"] for r in faithful]

    def level_stats(key):
        out = {}
        for name in ("상", "중", "하"):
            vals = [r[key] for r in rows if r["level"] == name]
            if vals:
                out[name] = {"n": len(vals),
                             "mean": round(statistics.mean(vals), 1),
                             "median": statistics.median(vals)}
        return out

    summary = {
        "n_clips": len(rows),
        "n_speakers": len({r["speaker_id"] for r in rows}),
        "strata": dict(Counter(r["level"] for r in rows)),
        "deviation_rate": round(statistics.mean(deviates), 3),
        "A_deviation_agreement": {
            "spearman_system_vs_heard": round(rho_heard, 3),
            "spearman_95ci": [round(v, 3) for v in rho_heard_ci],
            "deviation_detection_auc": round(auc_dev, 3),
            "auc_95ci": [round(v, 3) for v in auc_dev_ci],
        },
        "B_proficiency_association": {
            "spearman_system_vs_speechLV": round(rho_lv, 3),
            "spearman_95ci": [round(v, 3) for v in rho_lv_ci],
            "pairwise_auc": pair_auc,
            "speaker_level_spearman_vs_topik": (
                round(rho_topik, 3) if rho_topik is not None else None),
            "n_speakers_with_topik": len(spk_scores),
        },
        "asr_noise_floor": {
            "n_faithful": len(faithful),
            "mean_asr_vs_heard_score": round(statistics.mean(floor), 1),
            "median": statistics.median(floor),
            "p10": sorted(floor)[len(floor) // 10],
        },
        "system_score_by_level": level_stats("system_score"),
        "heard_score_by_level": level_stats("heard_score"),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwritten: {OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["build", "run", "analyze"])
    ap.add_argument("--root", default=os.environ.get("AIHUB131_ROOT", DEFAULT_ROOT))
    ap.add_argument("--per-level", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.command == "build":
        cmd_build(args.root, args.per_level)
    elif args.command == "run":
        cmd_run(args.limit)
    else:
        cmd_analyze()


if __name__ == "__main__":
    main()
