"""Experiment 3 — G2P engine accuracy on a held-out evaluation set.

Evaluates src/g2p.py surface-form conversion against gold standard
pronunciations (표준국어대사전) that were NOT used while developing the
engine or its unit tests. In-scope (core) and out-of-scope (morphology-
dependent) items are reported separately: the latter quantify documented
limitations rather than defects.

Usage:  python experiments/exp3_g2p_heldout.py [--check]
        --check: exit non-zero if in-scope accuracy drops below 100%
                 (used as a regression gate in CI)
Output: experiments/results/exp3_g2p_heldout.json + console table
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.g2p import to_surface  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "g2p_heldout.tsv")
OUT = os.path.join(os.path.dirname(__file__), "results", "exp3_g2p_heldout.json")


def load_items():
    items = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            scope, ortho, gold = line.split("\t")
            items.append({"scope": scope, "orthography": ortho, "gold": gold})
    return items


# scopes whose accuracy must stay at 100% (CI regression gate via --check)
_GATED_SCOPES = ("core", "morph")

_SCOPE_LABELS = {
    "core": "core  (context-free rules)",
    "morph": "morph (morphology-conditioned)",
    "oos": "oos   (semantics-dependent)",
}


def main():
    items = load_items()
    results = {}
    for item in items:
        bucket = results.setdefault(
            item["scope"], {"n": 0, "correct": 0, "failures": []}
        )
        pred = to_surface(item["orthography"])
        bucket["n"] += 1
        if pred == item["gold"]:
            bucket["correct"] += 1
        else:
            bucket["failures"].append(
                {"orthography": item["orthography"], "gold": item["gold"], "pred": pred}
            )

    for scope, r in results.items():
        r["accuracy"] = round(r["correct"] / r["n"], 4) if r["n"] else None

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for scope, r in results.items():
        label = _SCOPE_LABELS.get(scope, scope)
        print(f"{label}: {r['correct']}/{r['n']} = {r['accuracy']:.1%}")
        for fail in r["failures"]:
            print(f"  [{scope}] {fail['orthography']}: gold={fail['gold']} pred={fail['pred']}")

    if "--check" in sys.argv and any(
        results.get(s, {}).get("accuracy", 1.0) < 1.0 for s in _GATED_SCOPES
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
