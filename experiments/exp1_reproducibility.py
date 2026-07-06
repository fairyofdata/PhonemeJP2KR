"""Experiment 1 — Reproducibility: LLM-generated IPA scoring (v1) vs
deterministic G2P scoring (v2).

Replicates the pre-refactor (v1) scoring path: Gemini is asked to
transcribe both the target and the ASR hypothesis to IPA, and the score
is Levenshtein over those IPA strings — exactly the v1 logic (same
temperature 0.2). The identical input is scored N times by each method;
a valid measurement instrument must return the same value every time.

Usage:  python experiments/exp1_reproducibility.py [n_runs]
Output: experiments/results/exp1_reproducibility.json + console summary
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from google.genai import types  # noqa: E402

from src.config import GEMINI_MODEL_ID  # noqa: E402
from src.llm import _get_client  # noqa: E402
from src.scoring import score_pronunciation  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "exp1_reproducibility.json")

# fixed input: a typical Japanese-L1 epenthesis error
TARGET = "감사합니다"
ACTUAL = "감사하무니다"  # what Wav2Vec2 heard (ウ-vowel inserted)

_V1_PROMPT = """다음 두 한국어 문장을 각각 국제음성기호(IPA)로 전사하세요.
반드시 아래 JSON 형식으로만 응답하세요.

- 문장1 (목표): {target}
- 문장2 (실제 발음): {actual}

{{
  "target_ipa": "문장1의 IPA",
  "actual_ipa": "문장2의 IPA"
}}"""


def levenshtein(s1, s2):
    """v1's scoring metric, verbatim logic."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            current_row.append(min(previous_row[j + 1] + 1,
                                   current_row[j] + 1,
                                   previous_row[j] + (c1 != c2)))
        previous_row = current_row
    return previous_row[-1]


def v1_score(target_ipa, actual_ipa):
    if not target_ipa or not actual_ipa:
        return 0
    distance = levenshtein(target_ipa, actual_ipa)
    max_len = max(len(target_ipa), len(actual_ipa))
    return round(max(0, 100 - (distance / max_len * 100))) if max_len else 100


def run_v1_once(client):
    response = client.models.generate_content(
        model=GEMINI_MODEL_ID,
        contents=_V1_PROMPT.format(target=TARGET, actual=ACTUAL),
        config=types.GenerateContentConfig(
            temperature=0.2,  # v1 used 0.2
            response_mime_type="application/json",
        ),
    )
    data = json.loads(response.text)
    return {
        "target_ipa": data.get("target_ipa", ""),
        "actual_ipa": data.get("actual_ipa", ""),
        "score": v1_score(data.get("target_ipa", ""), data.get("actual_ipa", "")),
    }


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    client = _get_client()

    v1_runs = []
    for i in range(n_runs):
        for attempt in range(3):
            try:
                v1_runs.append(run_v1_once(client))
                break
            except Exception as e:
                print(f"  run {i}: retry after error: {e}")
                time.sleep(10)
        print(f"v1 run {i + 1}/{n_runs}: score={v1_runs[-1]['score']}"
              f" ipa=/{v1_runs[-1]['target_ipa']}/ vs /{v1_runs[-1]['actual_ipa']}/")

    v2_scores = [score_pronunciation(TARGET, ACTUAL).score for _ in range(n_runs)]

    v1_scores = [r["score"] for r in v1_runs]
    summary = {
        "input": {"target": TARGET, "actual": ACTUAL},
        "n_runs": n_runs,
        "v1_llm_ipa": {
            "scores": v1_scores,
            "mean": round(statistics.mean(v1_scores), 2),
            "stdev": round(statistics.stdev(v1_scores), 2) if len(v1_scores) > 1 else 0.0,
            "min": min(v1_scores),
            "max": max(v1_scores),
            "range": max(v1_scores) - min(v1_scores),
            "unique_target_ipa": sorted({r["target_ipa"] for r in v1_runs}),
            "unique_actual_ipa": sorted({r["actual_ipa"] for r in v1_runs}),
            "runs": v1_runs,
        },
        "v2_deterministic": {
            "scores": v2_scores,
            "mean": statistics.mean(v2_scores),
            "stdev": round(statistics.stdev(v2_scores), 2) if len(v2_scores) > 1 else 0.0,
            "min": min(v2_scores),
            "max": max(v2_scores),
            "range": max(v2_scores) - min(v2_scores),
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(f"v1 (LLM IPA):     mean={summary['v1_llm_ipa']['mean']}"
          f" sd={summary['v1_llm_ipa']['stdev']}"
          f" range={summary['v1_llm_ipa']['min']}–{summary['v1_llm_ipa']['max']}"
          f" | {len(summary['v1_llm_ipa']['unique_target_ipa'])} distinct target IPAs")
    print(f"v2 (deterministic): mean={summary['v2_deterministic']['mean']}"
          f" sd={summary['v2_deterministic']['stdev']}"
          f" range={summary['v2_deterministic']['min']}–{summary['v2_deterministic']['max']}")


if __name__ == "__main__":
    main()
