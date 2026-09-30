"""Experiment 1, deterministic half, as a CI regression gate.

Only v2 (G2P + jamo alignment) is here. v1 — the LLM transcribing to IPA —
stays out of CI: its run-to-run variation is the quantity the experiment
measures, so as a pass/fail test it would be flaky by design and would
need an API key (DECISIONS 16).
"""

import json
import os
import statistics

from src.scoring import score_pronunciation

RESULT = os.path.join(os.path.dirname(__file__), "..", "experiments", "results",
                      "exp1_reproducibility.json")


def test_identical_input_scores_identically_ten_times():
    scores = [score_pronunciation("감사합니다", "감사하무니다").score for _ in range(10)]
    assert statistics.pstdev(scores) == 0


def test_the_recorded_deterministic_score_still_holds():
    with open(RESULT, encoding="utf-8") as f:
        recorded = json.load(f)["v2_deterministic"]["scores"]
    assert score_pronunciation("감사합니다", "감사하무니다").score == recorded[0]
