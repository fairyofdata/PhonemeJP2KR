# -*- coding: utf-8 -*-
"""Experiment 6b — GOP baseline and three-way comparison on the Exp 6 sample.

Implements a CTC variant of Goodness of Pronunciation (Witt & Young, 2000)
from the same Wav2Vec2 acoustic model the pipeline already uses, and
compares three scorers of the same 615 real-L2 clips:

  System  jamo-alignment score (1 − PER over G2P-normalised ASR output) —
          this project's method, from exp6 run_results.jsonl
  GOP     likelihood-ratio pronunciation score computed here:
              GOP = (ll_forced − ll_free) / n_frames
          where ll_forced = CTC log-likelihood of the target transcript
          (forced alignment, summed over all valid paths) and ll_free is
          the unconstrained greedy-path log-likelihood. 0 is perfect;
          more negative = the audio strays further from the target.
  Heard   jamo deviation that native transcribers heard (from exp6),
          plus the corpus's speech-level rating (상/중/하) and TOPIK.

Design choices (documented because they matter):
  - The CTC target is the *orthographic* script, not the G2P surface form:
    the acoustic model was trained on orthographic transcripts, and its
    1204-syllable vocabulary covers orthography far better (0.27% UNK
    tokens vs 2.4% for surface spellings). Tokens missing from the vocab
    are dropped from the target (77/615 clips are affected by ≥1 drop).
  - ll_free uses the greedy (argmax) path including blanks — the standard
    denominator approximation for CTC-GOP; it cancels audio-quality and
    duration effects that the raw forced likelihood would absorb.

Usage:
  python experiments/exp6b_gop_baseline.py run      # compute GOP (resumable)
  python experiments/exp6b_gop_baseline.py analyze  # three-way stats -> JSON
Options:
  --limit N   pilot runs
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.statsutil import bootstrap_ci, rank_auc, spearman_rho  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "l2_aihub")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
MANIFEST = os.path.join(DATA_DIR, "manifest.csv")
EXP6_LOG = os.path.join(DATA_DIR, "run_results.jsonl")
GOP_LOG = os.path.join(DATA_DIR, "gop_results.jsonl")
OUT = os.path.join(os.path.dirname(__file__), "results", "exp6b_gop_threeway.json")

LV_ORD = {"하": 1, "중": 2, "상": 3}


def read_manifest():
    with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    rows = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["audio_file"]] = r
    return rows


def gop_for_clip(wav_path, target_text, processor, model, torch):
    import librosa
    from src.config import AUDIO_SAMPLE_RATE

    audio, _ = librosa.load(wav_path, sr=AUDIO_SAMPLE_RATE)
    inputs = processor(audio, return_tensors="pt", sampling_rate=AUDIO_SAMPLE_RATE)
    with torch.no_grad():
        logits = model(inputs.input_values).logits[0]        # (T, V)
    logp = torch.log_softmax(logits, dim=-1)

    ll_free = logp.max(dim=-1).values.sum().item()           # greedy path

    tok = processor.tokenizer
    ids = [i for i in tok(target_text).input_ids if i != tok.unk_token_id]
    n_dropped = len(tok(target_text).input_ids) - len(ids)
    targets = torch.tensor([ids], dtype=torch.long)
    n_frames = logp.shape[0]
    loss = torch.nn.functional.ctc_loss(
        logp.unsqueeze(1),                                   # (T, 1, V)
        targets,
        input_lengths=torch.tensor([n_frames]),
        target_lengths=torch.tensor([len(ids)]),
        blank=tok.pad_token_id,
        reduction="sum",
        zero_infinity=True,
    )
    ll_forced = -loss.item()

    return {
        "gop": (ll_forced - ll_free) / n_frames,
        "ll_forced_per_token": ll_forced / max(len(ids), 1),
        "n_frames": n_frames,
        "n_target_tokens": len(ids),
        "n_unk_dropped": n_dropped,
    }


def cmd_run(limit=None):
    import torch
    from src.asr import load_wav2vec_model

    manifest = read_manifest()
    done = set(read_jsonl(GOP_LOG))
    todo = [r for r in manifest if r["audio_file"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(done)} done, {len(todo)} to run")
    if not todo:
        return

    processor, model = load_wav2vec_model()
    t0 = time.time()
    with open(GOP_LOG, "a", encoding="utf-8") as out:
        for k, row in enumerate(todo, 1):
            wav = os.path.join(AUDIO_DIR, row["audio_file"])
            try:
                res = gop_for_clip(wav, row["target_text"], processor, model, torch)
            except Exception as e:
                print(f"  ERROR {row['audio_file']}: {e}")
                continue
            res["audio_file"] = row["audio_file"]
            out.write(json.dumps(res, ensure_ascii=False) + "\n")
            out.flush()
            if k % 10 == 0 or k == len(todo):
                rate = (time.time() - t0) / k
                print(f"  {k}/{len(todo)}  {rate:.1f}s/clip  "
                      f"eta {rate * (len(todo) - k) / 60:.0f}min")


def scorer_vs_humans(rows, score_key):
    """One scorer's full metric block against the human signals."""
    scores = [r[score_key] for r in rows]
    heard = [r["heard_score"] for r in rows]
    lv = [LV_ORD[r["level"]] for r in rows]
    deviates = [int(r["heard_score"] < 100) for r in rows]
    neg = [-s for s in scores]

    pair_auc = {}
    for hi, lo in (("상", "하"), ("상", "중"), ("중", "하")):
        sub = [r for r in rows if r["level"] in (hi, lo)]
        pair_auc[f"{hi}>{lo}"] = round(rank_auc(
            [r[score_key] for r in sub], [r["level"] == hi for r in sub]), 3)

    spk = defaultdict(list)
    for r in rows:
        spk[(r["speaker_id"], r["topik"])].append(r[score_key])
    spk_scores, spk_topik = [], []
    for (sid, grade), vals in spk.items():
        if grade.isdigit() and len(vals) >= 3:
            spk_scores.append(statistics.mean(vals))
            spk_topik.append(int(grade))

    rho_h = spearman_rho(scores, heard)
    rho_lv = spearman_rho(scores, lv)
    auc_dev = rank_auc(neg, deviates)
    return {
        "spearman_vs_heard": round(rho_h, 3),
        "heard_95ci": [round(v, 3) for v in bootstrap_ci(scores, heard, spearman_rho)],
        "deviation_detection_auc": round(auc_dev, 3),
        "dev_auc_95ci": [round(v, 3) for v in bootstrap_ci(neg, deviates, rank_auc)],
        "spearman_vs_speechLV": round(rho_lv, 3),
        "speechLV_95ci": [round(v, 3) for v in bootstrap_ci(scores, lv, spearman_rho)],
        "pairwise_auc": pair_auc,
        "speaker_spearman_vs_topik": round(spearman_rho(spk_scores, spk_topik), 3),
        "mean_by_level": {
            name: round(statistics.mean(
                [r[score_key] for r in rows if r["level"] == name]), 3)
            for name in ("상", "중", "하")
        },
    }


def cmd_analyze():
    meta = {r["audio_file"]: r for r in read_manifest()}
    sys_rows = read_jsonl(EXP6_LOG)
    gop_rows = read_jsonl(GOP_LOG)
    rows = []
    for fname, m in meta.items():
        if fname in sys_rows and fname in gop_rows:
            rows.append({**m, **sys_rows[fname], **gop_rows[fname]})
    print(f"analyzing {len(rows)} clips with all three scores")

    summary = {
        "n_clips": len(rows),
        "unk_dropped_clips": sum(1 for r in rows if r["n_unk_dropped"] > 0),
        "scorers": {
            "system_jamo_alignment": scorer_vs_humans(rows, "system_score"),
            "gop_ctc": scorer_vs_humans(rows, "gop"),
        },
        "scorer_agreement": {
            "spearman_system_vs_gop": round(spearman_rho(
                [r["system_score"] for r in rows], [r["gop"] for r in rows]), 3),
        },
        "gop_distribution": {
            "mean": round(statistics.mean(r["gop"] for r in rows), 3),
            "median": round(statistics.median(r["gop"] for r in rows), 3),
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwritten: {OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["run", "analyze"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.command == "run":
        cmd_run(args.limit)
    else:
        cmd_analyze()


if __name__ == "__main__":
    main()
