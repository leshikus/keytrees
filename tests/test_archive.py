"""Checks of numbers taken from the archive drafts and of the first estimate for search trees."""
from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache
from math import comb

from keytrees.core import complete_tree_cost, entropy, huffman_cost


def test_early_table_entropy_and_huffman_columns():
    """The two recomputable columns of the early table in the gradient chapter."""
    table = {8: (3, 3), 9: (3.17, 3.22), 10: (3.32, 3.40), 11: (3.46, 3.55), 12: (3.58, 3.67)}
    for m, (h, cost) in table.items():
        uniform = [1 / m] * m
        assert round(entropy(uniform), 2) == h
        assert round(huffman_cost(uniform), 2) == cost
        assert round(complete_tree_cost(m), 2) == cost


@lru_cache(maxsize=None)
def t3(n: int, m: int) -> Fraction:
    """Exact t_{n,m}(R_3) from the recurrences of the search-tree chapter."""
    if m <= 1:
        return Fraction(0)
    if m == 2:
        return Fraction(1)
    if m == 3:
        return Fraction(5, 3)
    half = 2 ** (n - 1)
    total = Fraction(1)
    for k in range(m + 1):
        if k > half or m - k > half:
            continue
        c = Fraction(comb(half, k) * comb(half, m - k), comb(2 * half, m))
        total += c * (Fraction(k, m) * t3(n - 1, k) + Fraction(m - k, m) * t3(n - 1, m - k))
    return total


def test_first_estimate_holds_exactly():
    for n in range(1, 11):
        for m in range(2, min(2**n, 80) + 1):
            assert float(t3(n, m)) <= 4 / 3 * math.log2(m) + 1e-12


def test_entropy_bound_used_by_first_estimate():
    for i in range(1, 1000):
        x = i / 1000
        h = -x * math.log2(x) - (1 - x) * math.log2(1 - x)
        assert h >= 4 * x * (1 - x) - 1e-12


def test_variance_identity_used_by_first_estimate():
    for n in range(2, 9):
        for m in range(1, 2**n + 1):
            half = 2 ** (n - 1)
            mean = Fraction(0)
            for k in range(m + 1):
                if k <= half and m - k <= half:
                    c = Fraction(comb(half, k) * comb(half, m - k), comb(2 * half, m))
                    mean += c * 4 * (Fraction(k, m) - Fraction(1, 2)) ** 2
            assert mean == Fraction(2**n - m, m * (2**n - 1))


def test_criterion_example_is_concave_and_symmetric():
    """H(q, p-q) of the general criterion peaks at q = p/2."""
    p = 0.6
    f = lambda q: -q * math.log2(q) - (p - q) * math.log2(p - q)
    grid = [p * i / 200 for i in range(1, 200)]
    values = [f(q) for q in grid]
    assert max(range(len(values)), key=values.__getitem__) == 99
    assert all(abs(f(q) - f(p - q)) < 1e-12 for q in grid)
    assert all(values[i - 1] + values[i + 1] <= 2 * values[i] + 1e-12 for i in range(1, len(values) - 1))
