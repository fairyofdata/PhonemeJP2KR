"""Dependency-free statistics helpers for the evaluation scripts.

Implements Pearson r, Spearman rho (average ranks for ties), and a
percentile bootstrap confidence interval. Pure Python so the evaluation
harness runs anywhere the app runs, including CI, without scipy.
"""

import math
import random


def mean(xs):
    return sum(xs) / len(xs)


def pearson_r(x, y):
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two sequences of equal length >= 2")
    mx, my = mean(x), mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def average_ranks(xs):
    """Ranks starting at 1; tied values receive the average of their ranks."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(x, y):
    return pearson_r(average_ranks(x), average_ranks(y))


def rank_auc(scores, labels):
    """Mann-Whitney AUC: P(score of a positive > score of a negative),
    ties counted as 0.5. `labels` are truthy for positives."""
    pos = [i for i, l in enumerate(labels) if l]
    neg = [i for i, l in enumerate(labels) if not l]
    if not pos or not neg:
        raise ValueError("need both positive and negative labels")
    ranks = average_ranks(scores)
    r_pos = sum(ranks[i] for i in pos)
    n1, n2 = len(pos), len(neg)
    return (r_pos - n1 * (n1 + 1) / 2) / (n1 * n2)


def bootstrap_ci(x, y, statistic, n_boot=2000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for a paired statistic(x, y)."""
    rng = random.Random(seed)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = [x[i] for i in idx]
        by = [y[i] for i in idx]
        try:
            stats.append(statistic(bx, by))
        except ValueError:
            continue
    stats.sort()
    lo = stats[int(len(stats) * (alpha / 2))]
    hi = stats[int(len(stats) * (1 - alpha / 2)) - 1]
    return lo, hi
