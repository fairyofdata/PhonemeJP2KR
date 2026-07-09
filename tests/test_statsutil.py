"""Tests for the dependency-free statistics helpers."""

import pytest

from experiments.statsutil import average_ranks, bootstrap_ci, pearson_r, spearman_rho


def test_pearson_perfect_positive():
    assert pearson_r([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def test_pearson_perfect_negative():
    assert pearson_r([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)


def test_pearson_zero_variance():
    assert pearson_r([1, 1, 1], [1, 2, 3]) == 0.0


def test_average_ranks_ties():
    # [10, 20, 20, 30] → ranks [1, 2.5, 2.5, 4]
    assert average_ranks([10, 20, 20, 30]) == [1, 2.5, 2.5, 4]


def test_spearman_monotonic_nonlinear():
    # monotonic but non-linear → Spearman 1.0
    x = [1, 2, 3, 4, 5]
    y = [1, 8, 27, 64, 125]
    assert spearman_rho(x, y) == pytest.approx(1.0)


def test_spearman_reversed():
    assert spearman_rho([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_known_value():
    # classic textbook example with one displaced pair
    x = [1, 2, 3, 4, 5]
    y = [1, 3, 2, 4, 5]
    # d = [0, -1, 1, 0, 0]; rho = 1 - 6*2 / (5*24) = 0.9
    assert spearman_rho(x, y) == pytest.approx(0.9)


def test_bootstrap_ci_brackets_estimate():
    x = list(range(30))
    y = [v + (1 if v % 3 == 0 else -1) for v in x]
    rho = spearman_rho(x, y)
    lo, hi = bootstrap_ci(x, y, spearman_rho, n_boot=500)
    assert lo <= rho <= hi
    assert lo > 0.8  # strongly monotonic data → tight positive CI
