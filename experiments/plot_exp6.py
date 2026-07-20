# -*- coding: utf-8 -*-
"""Publication figures for Experiment 6 / 6b.

Reads the per-clip results (experiments/data/l2_aihub/, local-only by
AI-Hub license) and renders 300-dpi PNGs into docs/assets/:

  exp6_score_by_level.png   score distributions by speech-level rating,
                            one panel per scorer (System / GOP)
  exp6_roc.png              ROC curves for both scorers on the two
                            detection tasks (상 vs 하; transcriber-noted
                            deviation)

Colors: colorblind-validated pair — System #2a78d6 (blue), GOP #eb6834
(orange); sequential blues for the ordinal level fills. Identity is never
color-alone (axis labels, direct labels, legends).

Usage:  python experiments/plot_exp6.py
"""

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "l2_aihub")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")

BLUE, ORANGE = "#2a78d6", "#eb6834"
LEVEL_FILLS = {"하": "#a9c9ee", "중": "#639fe1", "상": "#2a78d6"}
INK, MUTED = "#1f2937", "#6b7280"

plt.rcParams.update({
    "font.sans-serif": ["Malgun Gothic", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def load_rows():
    meta = {}
    with open(os.path.join(DATA_DIR, "manifest.csv"), newline="",
              encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            meta[r["audio_file"]] = r
    for name in ("run_results.jsonl", "gop_results.jsonl"):
        with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d["audio_file"] in meta:
                        meta[d["audio_file"]].update(d)
    rows = [r for r in meta.values() if "system_score" in r and "gop" in r]
    print(f"{len(rows)} clips with all scores")
    return rows


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_score_by_level(rows):
    levels = ["하", "중", "상"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    panels = [("System (jamo alignment)", "system_score", "score (0–100)"),
              ("GOP (CTC likelihood ratio)", "gop", "GOP (0 = perfect)")]
    for ax, (title, key, ylabel) in zip(axes, panels):
        data = [[r[key] for r in rows if r["level"] == lv] for lv in levels]
        bp = ax.boxplot(
            data, tick_labels=[f"{lv}\n(n={len(d)})" for lv, d in zip(levels, data)],
            patch_artist=True, widths=0.55, showfliers=False,
            medianprops=dict(color=INK, linewidth=1.6),
            whiskerprops=dict(color=MUTED, linewidth=1.0),
            capprops=dict(color=MUTED, linewidth=1.0),
        )
        for patch, lv in zip(bp["boxes"], levels):
            patch.set_facecolor(LEVEL_FILLS[lv])
            patch.set_edgecolor(MUTED)
            patch.set_linewidth(1.0)
        # direct median labels
        for i, d in enumerate(data, 1):
            med = sorted(d)[len(d) // 2]
            ax.annotate(f"{med:.2f}" if key == "gop" else f"{med:.0f}",
                        (i, med), textcoords="offset points", xytext=(22, -4),
                        fontsize=8, color=INK)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("speech-level rating (SentenceSpeechLV)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=9)
        style_axis(ax)
    fig.suptitle("Score distributions on real Japanese-L1 Korean speech "
                 "(AI-Hub 131, n=615)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(ASSETS, "exp6_score_by_level.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def roc_points(scores, labels):
    """ROC via score-descending sweep (higher score = predicted positive)."""
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    n_pos = sum(1 for _, l in pairs if l)
    n_neg = len(pairs) - n_pos
    xs, ys = [0.0], [0.0]
    tp = fp = 0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            if pairs[j][1]:
                tp += 1
            else:
                fp += 1
            j += 1
        xs.append(fp / n_neg)
        ys.append(tp / n_pos)
        i = j
    return xs, ys


def auc_of(xs, ys):
    return sum((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2
               for i in range(1, len(xs)))


def fig_roc(rows):
    tasks = [
        ("상 vs 하 (speech level)",
         [r for r in rows if r["level"] in ("상", "하")],
         lambda r: r["level"] == "상"),
        ("deviation detection (transcriber-noted)",
         rows,
         lambda r: r["heard_score"] < 100),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
    for ax, (title, sub, is_pos) in zip(axes, tasks):
        labels = [is_pos(r) for r in sub]
        # for deviation detection, LOWER score = predicted positive → flip sign
        sign = -1 if "deviation" in title else 1
        for name, key, color in (("System", "system_score", BLUE),
                                 ("GOP", "gop", ORANGE)):
            xs, ys = roc_points([sign * r[key] for r in sub], labels)
            ax.plot(xs, ys, color=color, linewidth=2,
                    label=f"{name} (AUC {auc_of(xs, ys):.3f})")
        ax.plot([0, 1], [0, 1], color="#d1d5db", linewidth=1, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("false positive rate", fontsize=9)
        ax.set_ylabel("true positive rate", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=9)
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        style_axis(ax)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    fig.suptitle("ROC: jamo-alignment score vs CTC-GOP baseline "
                 "(AI-Hub 131 sample)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(ASSETS, "exp6_roc.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def main():
    rows = load_rows()
    if len(rows) < 50:
        sys.exit("not enough scored clips — run exp6 and exp6b first")
    os.makedirs(ASSETS, exist_ok=True)
    fig_score_by_level(rows)
    fig_roc(rows)


if __name__ == "__main__":
    main()
