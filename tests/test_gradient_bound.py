"""Checks of the numbers in articles/gradient-bound/article.tex."""
from __future__ import annotations

import csv
import doctest
import math
import os
from fractions import Fraction
from pathlib import Path

import pytest

import keytrees.gradient_bound as gb

DATA = Path(__file__).resolve().parents[1] / "data" / "gradient-bound"


def rows(name):
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def test_doctests():
    result = doctest.testmod(gb, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0
    assert result.failed == 0


# ---------------------------------------------------------------- lemmas


def test_fork_identity_on_random_tables():
    """Cost - log m equals the sum over forks of (|v|/m)(1 - h(k_v/|v|))."""
    import random

    rnd = random.Random(5)
    for _ in range(50):
        n = rnd.randint(3, 7)
        m = rnd.randint(2, min(20, 2**n))
        cols = rnd.sample(range(2**n), m)
        total = 0.0

        def walk(cs):
            nonlocal total
            s = len(cs)
            if s <= 1:
                return
            best, bq = None, 0
            for r in range(n):
                k = sum((c >> r) & 1 for c in cs)
                if min(k, s - k) > bq:
                    best, bq = r, min(k, s - k)
            one = [c for c in cs if (c >> best) & 1]
            total += s / m * (1 - gb._h(len(one) / s))
            walk(one)
            walk([c for c in cs if not (c >> best) & 1])

        walk(cols)
        assert float(gb.gradient_cost(cols, n)) - math.log2(m) == pytest.approx(total, abs=1e-12)


def test_entropy_quadratic_bound():
    for i in range(1, 1000):
        x = i / 1000
        assert gb._h(x) >= 4 * x * (1 - x) - 1e-15


def test_balanced_lower_bound():
    assert all(gb.balanced_probability(s) >= 1 / math.sqrt(2 * s) for s in range(2, 400))


def test_gamma():
    assert gb.complete_redundancy_sup() == pytest.approx(0.0861, abs=5e-5)
    best = max(float(gb.complete_cost(m)) - math.log2(m) for m in range(1, 5000))
    assert best <= gb.complete_redundancy_sup()


# ---------------------------------------------------------------- exact data


def test_exact_against_python_and_monograph():
    for r in rows("exact.csv"):
        n, m = int(r["n"]), int(r["m"])
        t = Fraction(int(r["numerator"]), int(r["denominator"]))
        if (n <= 4 and m <= 6) or (n == 5 and m <= 3):
            assert t == gb.exact_average(n, m)
        if m == 4 and n >= 2:
            assert t == 2 + gb.quality_le1_probability(n, 4) / 4
        if m == 5 and n >= 3:
            assert t == Fraction(12, 5) + Fraction(2, 5) * gb.quality_le1_probability(n, 5)
        # Theorem 1: C(m) <= t <= C(m) + bound
        c = gb.complete_cost(m)
        assert c <= t <= c + gb.theorem1_excess_bound(n, m)


def test_exact_table_values():
    t = {(int(r["n"]), int(r["m"])): Fraction(int(r["numerator"]), int(r["denominator"])) for r in rows("exact.csv")}
    assert len(t) == 48
    # values quoted in the article (redundancy t - log m, 4 decimals)
    quoted = {(3, 6): 0.0817, (4, 6): 0.0877, (5, 6): 0.0862, (6, 6): 0.0845,
              (4, 8): 0.0480, (5, 8): 0.0431, (6, 8): 0.0340,
              (5, 12): 0.0911, (7, 5): 0.0783, (4, 11): 0.0864, (5, 11): 0.0920}
    for key, val in quoted.items():
        assert float(t[key]) - math.log2(key[1]) == pytest.approx(val, abs=5e-5), key
    # largest redundancy in the table
    worst = max(float(v) - math.log2(k[1]) for k, v in t.items())
    assert worst == pytest.approx(0.0920, abs=5e-5)


def test_theorem1_ratio_m4():
    for n, ratio in ((10, 1.037), (20, 1.0002)):
        exact = gb.quality_le1_probability(n, 4) / 4
        assert float(gb.theorem1_excess_bound(n, 4) / exact) == pytest.approx(ratio, abs=5e-4)


# ---------------------------------------------------------------- constants


def test_blocked_constants():
    c = {int(r["b"]): r for r in rows("blocked_constants.csv")}
    assert set(c) == {2, 3, 4}
    for b, r in c.items():
        assert int(r["S0"]) == 4000
        assert float(r["mu_upper"]) == pytest.approx(float(gb.tail_mu(4000, float(r["alpha"]))), rel=1e-6)
        assert float(r["c_b_upper"]) >= float(r["max_r_upper"]) + float(r["mu_upper"]) - 1e-8
    assert float(c[2]["c_b_upper"]) < 0.181
    assert float(c[3]["c_b_upper"]) < 0.130
    assert float(c[4]["c_b_upper"]) < 0.109
    # floating-point chain values quoted in the text
    for b, val in {1: 0.3327, 2: 0.1714, 3: 0.1196, 4: 0.0994}.items():  # max over s <= 8192
        r = [float(x["r_b"]) for x in rows(f"chain_b{b}.csv")]
        assert max(r) == pytest.approx(val, abs=5e-5)


def test_blocked_constants_small_recomputed():
    u = gb.chain_upper(2, 120)
    f = gb.chain(2, 120)
    assert all(f[s] <= float(u[s]) + 1e-12 for s in range(1, 121))


def test_gradient_constants():
    c = {int(r["K"]): r for r in rows("gradient_constants.csv")}
    assert float(c[30]["cylinder_tail_upper"]) == pytest.approx(float(gb.cylinder_tail(500, 1)), abs=1e-8)
    assert float(c[30]["c_K_upper"]) < 0.20
    assert float(c[40]["c_K_upper"]) < 0.16
    assert float(c[60]["c_K_upper"]) < 0.13
    assert float(c[20]["c_K_upper"]) < 0.32


def test_cylinder_side_condition():
    """n >= (2 + 0.61 log 2n) log m + 6 and m > 500 give (m/2) P_cyl <= 2^-10."""
    for lm in [x / 4 for x in range(36, 200)]:
        m = 2**lm
        n = 1
        while n < (2 + 0.61 * math.log2(2 * n)) * lm + 6:
            n += 1
        for nn in (n, n + 1, 2 * n, 10 * n):
            assert gb.large_failure_log2(nn, m, 500, 1) <= -10


def test_corollary_regime_is_dominated_by_K_log_m():
    """For K = 30 and n <= 2^40 the condition n >= 30 log m implies the side condition."""
    for lm in range(9, 200):
        n = 30 * lm
        while n <= 2**40 and n < 2**12:
            assert n >= (2 + 0.61 * math.log2(2 * n)) * lm + 6
            n *= 2


# ---------------------------------------------------------------- exact formula


def fractions_of(name):
    return {(int(r["n"]), int(r["m"])): Fraction(int(r["numerator"]), int(r["denominator"])) for r in rows(name)}


def test_formula_agrees_with_brute_force():
    brute, formula = fractions_of("exact.csv"), fractions_of("exact_formula.csv")
    assert len(brute) == 48 and all(formula[k] == v for k, v in brute.items())


def test_formula_cells_within_bounds():
    for (n, m), t in fractions_of("exact_formula.csv").items():
        c = gb.complete_cost(m)
        assert c <= t <= c + gb.theorem1_excess_bound(n, m)
        assert float(t) - math.log2(m) < 0.1


def test_python_formula_on_engine_cells():
    formula = fractions_of("exact_formula.csv")
    for key in ((4, 7), (6, 5), (8, 6)):
        assert gb.path_formula(*key) == formula[key]


def test_m11_not_monotone():
    t = fractions_of("exact_formula.csv")
    red = {n: float(t[n, 11]) for n in (4, 5, 6)}
    assert red[5] > red[4] and red[5] > red[6]


def test_many_rows_corollary():
    def smallest_n(m):  # ceil(sqrt(2) m^(3/2)), in integers
        return math.isqrt(2 * m**3 - 1) + 1

    for m in range(2, 40):
        assert smallest_n(m) <= 2**m
    formula = fractions_of("exact_formula.csv")
    for m in range(4, 13):
        n = smallest_n(m)
        assert (n, m) in formula
        q = gb.distinct_probability(n, m)
        bound = m / 2 * math.exp(-(1 - math.log(2)) * m) / float(q)
        assert float(gb.theorem1_excess_bound(n, m)) <= bound


def test_pair_count_at_8_32():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "gradient_bound_data.py"
    spec = importlib.util.spec_from_file_location("gradient_bound_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert 0.5e10 < mod.tuple_count(8, 32) < 2e10


def test_article_macros_are_upper_bounds():
    """constants.tex rounds the data upwards (generated by scripts/gradient_bound_data.py tex)."""
    import re

    codes = Path(os.environ.get("CODES_DIR", Path(__file__).resolve().parents[2] / "codes"))
    path = codes / "articles" / "gradient-bound" / "constants.tex"
    if not path.exists():
        pytest.skip("the codes repository (articles) is not checked out next to keytrees")
    tex = path.read_text()
    mac = {k: float(v) for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{([0-9.]+)\}", tex)}
    bc = {int(r["b"]): float(r["c_b_upper"]) for r in rows("blocked_constants.csv")}
    gc = {int(r["K"]): float(r["c_K_upper"]) for r in rows("gradient_constants.csv")}
    for b, name in ((2, "cbtwo"), (3, "cbthree"), (4, "cbfour")):
        assert bc[b] <= mac[name] <= bc[b] + 0.001
        if b < 4:
            cor = (bc[b] + 1 / 64) / (1 - 2**-13)
            assert cor <= mac[{"cbtwo": "cortwo", "cbthree": "corthree"}[name]] <= cor + 0.001
    for K, name in ((20, "cGtwenty"), (30, "cGthirty"), (40, "cGforty"), (60, "cGsixty")):
        assert gc[K] <= mac[name] <= gc[K] + 0.001
    assert gc[30] + 0.0017 <= mac["cGthirtyfull"] <= gc[30] + 0.003
    assert mac["cGthirtyfull"] < 0.33
    assert all(v > 0 for v in mac.values()), "stale placeholder in constants.tex"
    placeholders = re.findall(r"\\newcommand\{\\(\w+)\}\{0\.0+\}", tex)
    assert not placeholders, placeholders
