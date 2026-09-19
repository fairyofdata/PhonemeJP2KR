# -*- coding: utf-8 -*-
"""Experiment 7 — Which L1 error categories actually separate proficiency?

The error taxonomy in src/scoring.py is grounded in the contrastive
literature (docs/L1_TAXONOMY.md), and Lee (2022) turns that same
literature into a curriculum of eleven "frequent error" categories for
Japanese learners. Neither step asks the empirical question this one
does: on real Japanese-L1 read speech, which of those categories fire at
a rate that *tracks proficiency* — and which fire at the same rate for
everyone, i.e. are dominated by ASR noise rather than learner error?

Data: the exp6 sample (615 clips, AI-Hub 131 Validation, 상/중/하) and its
cached ASR output (run_results.jsonl). No ASR is re-run: every tag is
recomputed from (script, ASR text) with the *current* classifier, so the
analysis also reflects later G2P/classifier changes.

Per tag we report:
  rate_by_level   occurrences per 100 target jamo, pooled per stratum
  auc_lo_vs_hi    P(per-clip rate of a 하 clip > that of a 상 clip), ties ½ —
                  0.5 = no proficiency signal, > 0.5 = fires more for 하
  noise_rate      rate on jamo-faithful readings of 상 speakers (transcribers
                  heard exactly the script) — an upper bound on how much of
                  the tag is ASR noise rather than learner error

Caveats: tags are inferred from orthographic ASR output, so categories
whose realization never changes the spelling-level transcript (rule
application, voicing, intonation) are structurally invisible here — see
the coverage audit in docs/L1_TAXONOMY.md. Rates are per target jamo, not
per opportunity (e.g. per coda), so compare a tag across levels, not
tags against each other.

Usage (on the machine holding the exp6 data):
  python experiments/exp7_error_profile.py
  python experiments/exp7_error_profile.py --data-dir PATH   # custom location
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.statsutil import bootstrap_ci, rank_auc  # noqa: E402
from src.scoring import score_pronunciation  # noqa: E402

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "l2_aihub")
OUT = os.path.join(os.path.dirname(__file__), "results", "exp7_error_profile.json")
LEVELS = ("상", "중", "하")
# residual tags carry no linguistic hypothesis; reported only as a total
GENERIC = {"substitution", "insertion", "deletion", "consonant_deletion"}


def load_rows(data_dir):
    with open(os.path.join(data_dir, "manifest.csv"), newline="",
              encoding="utf-8-sig") as f:
        meta = {r["audio_file"]: r for r in csv.DictReader(f)}
    rows = []
    with open(os.path.join(data_dir, "run_results.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["audio_file"] in meta:
                    rows.append({**meta[r["audio_file"]], **r})
    return rows


def profile(rows):
    per_clip = []
    for r in rows:
        report = score_pronunciation(r["target_text"], r["asr"])
        per_clip.append({
            "level": r["level"],
            "faithful": r["heard_score"] == 100,
            "n_jamo": max(report.ref_len, 1),
            "tags": Counter(t["tag"] for t in report.error_tags),
        })
    return per_clip


def rate(clips, tag):
    n = sum(c["n_jamo"] for c in clips)
    return round(100 * sum(c["tags"][tag] for c in clips) / n, 3) if n else None


def analyze(per_clip):
    tags = sorted({t for c in per_clip for t in c["tags"]} - GENERIC)
    by_level = {lv: [c for c in per_clip if c["level"] == lv] for lv in LEVELS}
    hi_lo = by_level["상"] + by_level["하"]
    labels = [c["level"] == "하" for c in hi_lo]
    noise_pool = [c for c in by_level["상"] if c["faithful"]]

    result = {}
    for tag in tags:
        clip_rates = [c["tags"][tag] / c["n_jamo"] for c in hi_lo]
        auc = rank_auc(clip_rates, labels)
        lo, hi = bootstrap_ci(clip_rates, labels, rank_auc)
        result[tag] = {
            "rate_by_level": {lv: rate(by_level[lv], tag) for lv in LEVELS},
            "clips_with_tag": sum(1 for c in per_clip if c["tags"][tag]),
            "auc_lo_vs_hi": round(auc, 3),
            "auc_95ci": [round(lo, 3), round(hi, 3)],
            "noise_rate": rate(noise_pool, tag),
        }
    generic_total = defaultdict(int)
    for c in per_clip:
        for t in GENERIC:
            generic_total[c["level"]] += c["tags"][t]
    return {
        "n_clips": len(per_clip),
        "strata": {lv: len(by_level[lv]) for lv in LEVELS},
        "n_noise_pool": len(noise_pool),
        "tags": dict(sorted(result.items(), key=lambda kv: -kv[1]["auc_lo_vs_hi"])),
        "untyped_errors_per_100_jamo": {
            lv: round(100 * generic_total[lv]
                      / max(sum(c["n_jamo"] for c in by_level[lv]), 1), 3)
            for lv in LEVELS
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows = load_rows(args.data_dir)
    print(f"profiling {len(rows)} clips")
    summary = analyze(profile(rows))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"{'tag':34s} {'상':>7s} {'중':>7s} {'하':>7s} {'noise':>7s}  AUC(하>상) [95% CI]")
    for tag, s in summary["tags"].items():
        r = s["rate_by_level"]
        print(f"{tag:34s} {r['상']:7.3f} {r['중']:7.3f} {r['하']:7.3f} "
              f"{s['noise_rate'] or 0:7.3f}  {s['auc_lo_vs_hi']:.3f} {s['auc_95ci']}")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
