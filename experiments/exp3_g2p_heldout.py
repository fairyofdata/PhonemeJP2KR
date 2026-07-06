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


def main():
    items = load_items()
    results = {"core": {"n": 0, "correct": 0, "failures": []},
               "oos": {"n": 0, "correct": 0, "failures": []}}
    for item in items:
        pred = to_surface(item["orthography"])
        bucket = results[item["scope"]]
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

    print(f"core (in-scope) : {results['core']['correct']}/{results['core']['n']}"
          f" = {results['core']['accuracy']:.1%}")
    print(f"oos  (morphology): {results['oos']['correct']}/{results['oos']['n']}"
          f" = {results['oos']['accuracy']:.1%}")
    for scope in ("core", "oos"):
        for fail in results[scope]["failures"]:
            print(f"  [{scope}] {fail['orthography']}: gold={fail['gold']} pred={fail['pred']}")

    if "--check" in sys.argv and results["core"]["accuracy"] < 1.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
