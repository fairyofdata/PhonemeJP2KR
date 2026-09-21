"""Export one stored analysis as the demo-take JSON the portfolio site draws from.

    python tools/export_demo_take.py --record 6 --out docs/assets/demo_take.json

No model or API is called. The stored record already holds the ASR
outputs; everything else (surface form, IPA, word alignment, tags,
katakana, explanations) is recomputed deterministically and checked
against what was stored, so the file and the screenshots taken from the
same record describe the same take. Schema: docs/DEMO_TAKE.md.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import GEMINI_MODEL_ID, WAV2VEC_MODEL_ID, WHISPER_MODEL_ID  # noqa: E402
from src.database import get_record  # noqa: E402
from src.g2p import to_ipa, to_surface  # noqa: E402
from src.kana import to_kana  # noqa: E402
from src.labels import explain_error, tag_label  # noqa: E402
from src.scoring import score_pronunciation  # noqa: E402
from src.words import word_view  # noqa: E402

SCHEMA = "phonemejp2kr.demo_take/1"


def _pkg(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _error(e: dict) -> dict:
    out = {"tag": e["tag"], "label_ja": tag_label(e["tag"]), "ref": e["ref"], "hyp": e["hyp"],
           "explanation_ja": explain_error(e)}
    if "timestamp" in e:
        out["timestamp_s"] = e["timestamp"]
    return out


def build(record: dict) -> dict:
    a = record["analysis"]
    target, whisper, wav2vec = a["target"], a["whisper_text"], a["wav2vec_text"]

    # the deterministic layer must reproduce the stored take exactly
    report = score_pronunciation(target, wav2vec)
    checks = {
        "score": (report.score, a["score"]),
        "target_surface": (to_surface(target), a["target_surface"]),
        "target_ipa": (to_ipa(target), a["target_ipa"]),
        "whisper_ipa": (to_ipa(whisper), a["whisper_ipa"]),
        "actual_ipa": (to_ipa(wav2vec), a["actual_ipa"]),
        "error_tags": ([(t["tag"], t["ref"], t["hyp"]) for t in report.error_tags],
                       [(t["tag"], t["ref"], t["hyp"]) for t in a["error_tags"]]),
    }
    drift = [k for k, (now, stored) in checks.items() if now != stored]
    if drift:
        sys.exit(f"record {record['id']}: recomputation differs from the stored take: {drift}")

    words = word_view(target, whisper, wav2vec)
    stored_tags = iter(a["error_tags"])       # same order, verified above; carries timestamps
    for w in words:
        w["acoustic_errors"] = [_error(next(stored_tags)) for _ in w["acoustic_errors"]]
        w["heard_errors"] = [_error(e) for e in w["heard_errors"]]

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"record_id": record["id"], "recorded_at": record["timestamp"],
                   "screenshots": ["docs/assets/demo_phoneme_diff.png",
                                   "docs/assets/demo_channels.png",
                                   "docs/assets/demo_coaching.png"]},
        "versions": {
            "app": _pkg("phoneme-jp2kr") or "0.1.0",
            "git_commit": _git_commit(),
            "g2p_morphology": f"kiwipiepy {_pkg('kiwipiepy')}",
            "whisper": WHISPER_MODEL_ID,
            "wav2vec2": WAV2VEC_MODEL_ID,
            "coaching_llm": GEMINI_MODEL_ID,
        },
        "target": {"text": target, "surface": a["target_surface"], "ipa": a["target_ipa"]},
        "channels": {
            "heard": {
                "model": WHISPER_MODEL_ID, "text": whisper, "ipa": a["whisper_ipa"],
                "scored": False,
                "role": "What a listener model understood; its language model corrects "
                        "toward plausible Korean. Display only.",
            },
            "acoustic": {
                "model": WAV2VEC_MODEL_ID, "text": wav2vec, "ipa": a["actual_ipa"],
                "scored": True,
                "role": "Acoustic recognition without a language model; the score and all "
                        "scored error tags come from this channel.",
                "stripped_edge_noise": a.get("stripped_noise"),
                "katakana": {
                    "text": to_kana(wav2vec),
                    "is_channel": False,
                    "method": "rule-based transliteration of the surface jamo (src/kana.py); "
                              "merges lenis/aspirated/tense, ㅓ/ㅗ, ㅡ/ㅜ, ㄴ/ㅇ codas",
                },
            },
        },
        "score": {"value": a["score"], "definition": "round(100 × (1 − jamo PER)), acoustic vs target",
                  "calibrated": False},
        "words": words,
        "limitations": [
            "Phonological-rule errors are not observable: the acoustic output passes the same "
            "G2P as the target, so e.g. missing tensification in 춥고도 cannot be detected.",
            "heard_errors run the same classifier on Whisper for display; they are not scored.",
            "Explanations are fixed templates keyed by tag (src/labels.py); no LLM is involved.",
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", type=int, required=True, help="history record id")
    ap.add_argument("--out", default="docs/assets/demo_take.json")
    args = ap.parse_args()
    record = get_record(args.record)
    if not record or not record.get("analysis"):
        sys.exit(f"record {args.record} has no stored analysis")
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(build(record), f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
