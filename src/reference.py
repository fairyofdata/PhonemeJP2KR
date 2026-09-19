"""Interpret a phoneme score against measured ASR noise, not as an absolute.

Experiment 6 showed the score is ordinally valid (it ranks proficiency and
tracks error severity) but not calibrated: even jamo-faithful readings by
Japanese-L1 speakers score only 79.6 on average, because the acoustics
channel mis-transcribes accented speech. So the UI states where a score
falls in the distribution of *faithful* readings instead of implying that
100 − score is the learner's error.

Bands (cut points read from the Exp 6 results file, so a re-run after
fine-tuning moves them automatically):

    score ≥ median    as high as a typical faithful reading
    p10 ≤ score < med also common for faithful readings — indeterminate
    score < p10       rare for faithful readings (bottom 10%)

The bottom band is deliberately *not* labelled "error": P(score < p10 |
faithful) = 10% does not give P(error | score < p10) without the base rate
and the score distribution of erroneous readings. Deviation detection AUC
in Exp 6 is 0.717 — informative, not diagnostic.

The comparison with the learner's previous attempt at the same sentence
is the most strongly supported reading of the score (Exp 1: deterministic;
Exp 4: monotone in error severity), so the UI shows it alongside the band.
"""

import json
import os

_RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "experiments", "results", "exp6_l2_validation.json")
# values of the committed Exp 6 run, used if the results file is absent
_FALLBACK = {"p10": 68, "median": 81, "n_faithful": 483}


def load_reference(path: str = _RESULTS) -> dict:
    """Noise-floor quantiles of faithful readings from the Exp 6 results."""
    try:
        with open(path, encoding="utf-8") as f:
            floor = json.load(f)["asr_noise_floor"]
        return {"p10": int(floor["p10"]), "median": int(floor["median"]),
                "n_faithful": int(floor["n_faithful"])}
    except (OSError, KeyError, ValueError):
        return dict(_FALLBACK)


def score_band(score: int, ref: dict) -> dict:
    """Place a score in the faithful-reading reference distribution."""
    p10, median = ref["p10"], ref["median"]
    if score >= median:
        return {"band": "typical_faithful", "low": median, "high": 100}
    if score >= p10:
        return {"band": "indeterminate", "low": p10, "high": median - 1}
    return {"band": "rare_for_faithful", "low": 0, "high": p10 - 1}
