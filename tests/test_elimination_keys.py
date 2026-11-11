"""Checks of the numbers in articles/elimination-keys/article.tex.

Exact values come from keytrees.elimination and data/elimination-keys/exact.csv;
simulated values from data/elimination-keys/sweep.csv (src/cpp/elimination.cpp).
"""
from __future__ import annotations

import csv
import itertools
import math
import random
from pathlib import Path

import pytest

from keytrees.elimination import (
    MELLIN_CONSTANT,
    adversarial_order_cost,
    beta,
    exact_rows,
    fluctuation_bound,
    free_no_error_cost,
    harmonic_sum,
    kraft_start,
    kraft_threshold,
    oracle_bound,
    order_bounds,
    order_cost,
    survival,
    threshold,
    volume,
    x_hat,
)

STATED_DIST_SIZES = 15
DATA = Path(__file__).resolve().parents[1] / "data" / "elimination-keys"


def read(name: str) -> list[dict[str, str]]:
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


# ---------- Theorem 1: exact law, by brute force on tiny cases ----------

def brute_order_cost(m: int, n: int, t: int, errors: tuple[int, ...] = ()) -> float:
    """Mean tests of the fixed-order key over all codes of m words of length n
    (truth = word 0, answer = word 0 with the given positions flipped)."""
    total = count = 0
    words = range(2**n)
    for code in itertools.product(words, repeat=m):
        answer = code[0]
        for p in errors:
            answer ^= 1 << p
        e = [0] * m
        alive = m
        tests = 0
        while alive > 1 and tests < n:
            bit = (answer >> tests) & 1
            for j in range(m):
                if e[j] <= t and ((code[j] >> tests) & 1) != bit:
                    e[j] += 1
                    if e[j] > t:
                        alive -= 1
            tests += 1
        total += tests
        count += 1
    return total / count


@pytest.mark.parametrize("m,n,t,errors", [(3, 4, 0, ()), (3, 4, 1, ()), (3, 4, 1, (2,)), (2, 6, 1, (0,)),
                                          (3, 3, 1, (1,)), (2, 6, 2, (0, 5))])
def test_theorem1_exact_law_brute_force(m, n, t, errors):
    assert brute_order_cost(m, n, t, errors) == pytest.approx(order_cost(m, t, n), abs=1e-12)


def test_theorem1_simulation_agrees():
    # elimination order 1 1024 40 1 200 50 printed 15.302500 (stated in the article)
    assert order_cost(2**10, 1, 40) == pytest.approx(15.3030, abs=1e-4)
    assert abs(15.3025 - order_cost(2**10, 1, 40)) < 0.05


def test_beta_monotone_and_bound():
    for t in range(1, 5):
        values = [beta(r, t) for r in range(200)]
        assert all(a >= b - 1e-15 for a, b in itertools.pairwise(values))
        for r in range(t, 200):
            assert beta(r, t) <= t / (r - t + 1) + 1e-15
            assert survival(r + 1, t) == pytest.approx(survival(r, t) * (1 + beta(r, t)) / 2)


@pytest.mark.parametrize("t", range(4))
def test_theorem1_bounds_hold(t):
    for L in range(4, 41):
        m = 2**L
        lo, hi = order_bounds(m, t)
        assert lo <= order_cost(m, t) <= hi


def test_theorem1_limits():
    # deficit -> sum_k exp(-2^k) and upper -> r0 + 2
    limit = sum(math.exp(-(2.0**k)) for k in range(10))
    assert limit == pytest.approx(0.5219, abs=1e-4)
    lo, hi = order_bounds(2**200, 2)
    r0 = threshold(2**200, 2)
    assert r0 - lo == pytest.approx(limit, abs=0.02)
    assert hi - r0 == pytest.approx(2, abs=0.05)


def test_adversarial_is_threshold_2t_brute_force():
    # max over error patterns of weight <= t equals the threshold-2t key with no errors
    m, n, t = 3, 4, 1
    total = count = 0
    for code in itertools.product(range(2**n), repeat=m):
        worst = 0
        for k in range(t + 1):
            for flips in itertools.combinations(range(n), k):
                answer = code[0]
                for p in flips:
                    answer ^= 1 << p
                e = [0] * m
                alive, tests = m, 0
                while alive > 1 and tests < n:
                    bit = (answer >> tests) & 1
                    for j in range(m):
                        if e[j] <= t and ((code[j] >> tests) & 1) != bit:
                            e[j] += 1
                            alive -= e[j] > t
                    tests += 1
                worst = max(worst, tests)
        total += worst
        count += 1
    assert total / count == pytest.approx(adversarial_order_cost(m, t, n), abs=1e-12)


# ---------- Theorem 2: constants ----------

def test_mellin_constant_and_fluctuation():
    assert MELLIN_CONSTANT == pytest.approx(1.3327, abs=1e-4)
    bound = fluctuation_bound()
    assert bound < 1.6e-6
    for k in range(40):
        x = 30 + k / 40
        assert abs(harmonic_sum(x) - x - MELLIN_CONSTANT) <= bound


@pytest.mark.parametrize("t", [1, 2, 3])
def test_theorem2_error_decreases(t):
    errors = [abs(order_cost(2**L, t) - x_hat(2**L, t) - MELLIN_CONSTANT) for L in (16, 32, 64, 128, 256)]
    assert all(a > b for a, b in itertools.pairwise(errors))
    assert errors[-1] < 0.03


def test_exact_csv_matches_module():
    rows = read("exact.csv")
    fresh = exact_rows()
    assert len(rows) == len(fresh)
    for old, new in zip(rows, fresh):
        for key, value in new.items():
            assert float(old[key]) == pytest.approx(value, abs=2e-6)


def test_table_exact_values():
    rows = {(int(r["t"]), int(r["L"])): r for r in read("exact.csv")}
    stated = {  # (t, L): (order mean, Thm 2 approximation, adversarial mean, kraft r_m - 1)
        (1, 10): (15.30, 15.23, 18.78, 13), (1, 20): (26.06, 26.02, 30.16, 24), (1, 40): (46.90, 46.87, 51.72, 45),
        (2, 10): (18.78, 18.65, 25.11, 17), (2, 20): (30.16, 30.08, 37.51, 28), (2, 40): (51.72, 51.67, 60.31, 50),
        (3, 10): (22.02, 21.84, 31.00, 20), (3, 20): (33.94, 33.82, 44.25, 32), (3, 40): (56.14, 56.07, 68.13, 54),
    }
    for key, (mean, approx, adv, kraft) in stated.items():
        r = rows[key]
        assert float(r["order_mean"]) == pytest.approx(mean, abs=0.005)
        assert float(r["thm2_approx"]) == pytest.approx(approx, abs=0.005)
        assert float(r["adversarial_mean"]) == pytest.approx(adv, abs=0.005)
        assert int(r["kraft_bound"]) == kraft


# ---------- Theorem 3: Kraft lemma ----------

@pytest.mark.parametrize("t", range(4))
def test_kraft_supporting_line(t):
    # p_D >= p_{D0} + (p_{D0+1} - p_{D0})(D - D0) for all D >= 0 once D0 >= D_t
    start = kraft_start(t)
    assert start >= 5 * t
    for d0 in range(start, start + 60):
        slope = survival(d0 + 1, t) - survival(d0, t)
        for d in range(0, d0 + 200):
            assert survival(d, t) >= survival(d0, t) + slope * (d - d0) - 1e-300


def test_kraft_start_values():
    assert [kraft_start(t) for t in range(4)] == [0, 9, 15, 21]


def test_kraft_bound_below_free_upper():
    # free questions, worst case: the volume bound q with 2^q >= m V(q,t) is itself >= r_m
    for t in range(4):
        for L in (8, 16, 32):
            assert kraft_threshold(2**L, t) - 1 < order_cost(2**L, 2 * t)


# ---------- Theorem 4: oracle bound ----------

def test_oracle_bound_brute_force_small():
    # exact minimum certifying set sizes for tiny random codes stay above the bound on average
    rng = random.Random(5)
    m, n, t = 8, 10, 1
    costs = []
    for _ in range(300):
        code = [rng.getrandbits(n) for _ in range(m)]
        i = 0
        best = n
        for d in range(n + 1):
            found = False
            for cols in itertools.combinations(range(n), d):
                if all(sum(((code[i] ^ code[j]) >> c) & 1 for c in cols) > t for j in range(1, m)):
                    found = True
                    break
            if found:
                best = d
                break
        costs.append(best)
    mean = sum(costs) / len(costs)
    assert oracle_bound(m, n, t) <= mean


def test_oracle_bound_regimes():
    rows = {(int(r["t"]), int(r["L"])): r for r in read("exact.csv")}
    assert float(rows[(3, 16)]["oracle_bound_4L"]) == pytest.approx(22.0, abs=0.01)
    assert float(rows[(3, 16)]["oracle_bound_linear"]) == pytest.approx(19.0, abs=0.01)


# ---------- Proposition 5: free questions ----------

def free_key_no_error(L: int, t: int, i: int) -> int:
    """Run the construction of Proposition 5 for species i of m = 2^L, no errors."""
    m = 2**L
    e = [0] * m
    tests = 0
    for b in range(L):  # binary digits
        answer = (i >> b) & 1
        for j in range(m):
            e[j] += ((j >> b) & 1) != answer
        tests += 1
    leader = min(range(m), key=lambda j: e[j])
    for _ in range(t):  # "is it one of the other survivors?" answered no
        for j in range(m):
            if j != leader and e[j] <= t:
                e[j] += 1
        tests += 1
    assert [j for j in range(m) if e[j] <= t] == [i]
    return tests


def test_free_construction():
    for L, t in [(4, 1), (5, 2), (6, 3)]:
        for i in range(2**L):
            assert free_key_no_error(L, t, i) == free_no_error_cost(2**L, t)


# ---------- simulations ----------

def sweep_rows():
    return read("sweep.csv")


def test_sweep_no_wrong_identification():
    assert all(int(r["wrong"]) == 0 for r in sweep_rows())


def test_sweep_order_matches_theorem1():
    for r in sweep_rows():
        if r["rule"] == "order":
            m, n, t = int(r["m"]), int(r["n"]), int(r["t"])
            exact = order_cost(m, t, n)
            # sample of 8 codes x 75 specimens; allow 5 standard errors plus 0.15
            assert abs(float(r["mean_tests"]) - exact) <= 5 * float(r["stderr"]) + 0.15, r


def test_sweep_respects_oracle_bound():
    for r in sweep_rows():
        if r["errors"] == "none":
            m, n, t = int(r["m"]), int(r["n"]), int(r["t"])
            assert float(r["mean_tests"]) >= oracle_bound(m, n, t) - 5 * float(r["stderr"]) - 0.15, r


# ---------- exact means over all tables ----------

def exact_rows_csv():
    return read("exact_means.csv")


def reference_walk(code, n, t, answer, truth, rule):
    """Independent reference for the three rules; `answer` is a bit list."""
    m = len(code)
    e = [0] * m
    asked = set()
    while sum(x <= t for x in e) > 1 and len(asked) < n:
        live = [j for j in range(m) if e[j] <= t]
        unasked = [p for p in range(n) if p not in asked]
        if rule == "order":
            pick = unasked[0]
        else:
            r = n - len(asked) - 1

            def score(p):
                if rule == "weighted":
                    side = [0, 0]
                    for j in live:
                        side[code[j][p]] += volume(r, t - e[j])
                        side[1 - code[j][p]] += volume(r, t - e[j] - 1)
                    return min(side)
                least = min(e[j] for j in live)
                group = [j for j in live if e[j] == least]
                if len(group) > 1:
                    ones = sum(code[j][p] for j in group)
                    return min(ones, len(group) - ones)
                return sum(code[j][p] != code[group[0]][p] for j in live)

            pick = max(unasked, key=lambda p: (score(p), -p))
        asked.add(pick)
        for j in live:
            e[j] += code[j][pick] != answer[pick]
    return len(asked)


def optimal_total(code, n, t):
    """Least total no-error depth over all rules on one table, by plain recursion."""
    from functools import lru_cache

    m = len(code)

    @lru_cache(maxsize=None)
    def f(asked, group):
        z = group[0]
        alive = sum(sum(code[j][p] != code[z][p] for p in asked) <= t for j in range(m))
        if alive <= 1 or len(asked) == n:
            return 0
        best = None
        for p in range(n):
            if p in asked:
                continue
            nxt = tuple(sorted(asked + (p,)))
            value = len(group)
            for b in (0, 1):
                side = tuple(j for j in group if code[j][p] == b)
                if side:
                    value += f(nxt, side)
            best = value if best is None else min(best, value)
        return best

    return f((), tuple(range(m)))


def brute_means(m, n, t, model):
    """All tables, all species, all error sets, with no symmetry; exact Fractions."""
    from fractions import Fraction

    sums: dict = {}
    counts: dict = {}

    def add(key, value, weight=1):
        sums[key] = sums.get(key, 0) + value
        counts[key] = counts.get(key, 0) + weight

    words = list(itertools.product((0, 1), repeat=n))
    for code in itertools.product(words, repeat=m):
        if model == "dist" and any(
            sum(a != b for a, b in zip(x, y)) < 2 * t + 1 for x, y in itertools.combinations(code, 2)
        ):
            continue
        add(("0", "optimal"), Fraction(optimal_total(code, n, t), m))
        for truth in range(m):
            star = n
            for d in range(n + 1):
                if any(all(sum(code[truth][p] != code[j][p] for p in s) > t for j in range(m) if j != truth)
                       for s in itertools.combinations(range(n), d)):
                    star = d
                    break
            add(("0", "oracle"), star)
            for rule in ("order", "weighted", "leader"):
                worst = 0
                for w in range(min(t, n) + 1):
                    for flips in itertools.combinations(range(n), w):
                        answer = [code[truth][p] ^ (p in flips) for p in range(n)]
                        k = reference_walk(code, n, t, answer, truth, rule)
                        add((str(w), rule), k)
                        worst = max(worst, k)
                add(("worst", rule), worst)
    out = {}
    for key, value in sums.items():
        out[key] = Fraction(value) / counts[key]
    return out


@pytest.mark.parametrize("model,m,n,t", [("all", 2, 4, 1), ("all", 3, 3, 0), ("all", 3, 3, 1), ("all", 3, 4, 1),
                                         ("all", 4, 3, 1), ("all", 3, 3, 2), ("all", 4, 2, 0),
                                         ("dist", 4, 5, 1)])
def test_exact_means_against_brute_force(model, m, n, t):
    from fractions import Fraction

    rows = [r for r in exact_rows_csv()
            if (r["model"], int(r["m"]), int(r["n"]), int(r["t"])) == (model, m, n, t)]
    assert rows
    brute = brute_means(m, n, t, model)
    assert {(r["errors"], r["rule"]) for r in rows} == set(brute)
    for r in rows:
        assert Fraction(int(r["num"]), int(r["den"])) == brute[(r["errors"], r["rule"])], r


def test_exact_order_equals_formula():
    from fractions import Fraction

    from keytrees.elimination import exact_order_mean

    checked = 0
    for r in exact_rows_csv():
        if r["model"] != "all" or r["rule"] != "order":
            continue
        m, n, t = int(r["m"]), int(r["n"]), int(r["t"])
        value = Fraction(int(r["num"]), int(r["den"]))
        if r["errors"] == "worst":
            assert value == exact_order_mean(m, n, 2 * t), r
        else:
            assert value == exact_order_mean(m, n, t), r
        checked += 1
    assert checked > 100


def test_exact_inequalities_and_oracle_bound():
    from fractions import Fraction

    index = {}
    for r in exact_rows_csv():
        index[(r["model"], int(r["m"]), int(r["n"]), int(r["t"]), r["errors"], r["rule"])] = Fraction(
            int(r["num"]), int(r["den"]))
    for key, value in index.items():
        model, m, n, t, errors, rule = key
        if rule == "oracle":
            optimal = index.get((model, m, n, t, "0", "optimal"))
            if optimal is not None:
                assert value <= optimal
            for other in ("order", "weighted", "leader"):
                assert value <= index[(model, m, n, t, "0", other)]
            if model == "all":
                assert float(value) >= oracle_bound(m, n, t) - 1e-9
        if rule == "optimal":
            for other in ("order", "weighted", "leader"):
                assert value <= index[(model, m, n, t, "0", other)]


def test_free_values_against_proposition():
    # data/omn belongs to the omn-trees article and may live in the public keytrees repository
    candidates = [DATA.parent / "omn" / "average.csv",
                  Path(__file__).resolve().parents[2] / "keytrees" / "data" / "omn" / "average.csv"]
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        pytest.skip("data/omn/average.csv (omn-trees article) not available")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    checked = 0
    for r in rows:
        o, a, b, total = int(r["o"]), int(r["m"]), int(r["n"]), int(r["total"])
        if a == b and o >= 2:
            assert total <= o * free_no_error_cost(o, a)
            checked += 1
    assert checked > 10


def exact_index():
    from fractions import Fraction

    return {(r["model"], int(r["m"]), int(r["n"]), int(r["t"]), r["errors"], r["rule"]):
            Fraction(int(r["num"]), int(r["den"])) for r in exact_rows_csv()}


def test_exact_examples_in_text():
    from fractions import Fraction

    I = exact_index()
    assert I[("all", 3, 4, 1, "0", "weighted")] == Fraction(435, 128)
    assert I[("all", 3, 4, 1, "0", "order")] == Fraction(59, 16)
    assert I[("all", 3, 4, 1, "0", "optimal")] == Fraction(53, 16)
    assert round(float(I[("dist", 8, 6, 1, "0", "leader")]), 3) == 4.267
    assert round(float(I[("dist", 8, 6, 1, "0", "weighted")]), 3) == 4.800
    assert round(float(I[("dist", 4, 6, 1, "0", "order")]), 3) == 4.081
    assert round(float(I[("dist", 4, 6, 1, "1", "order")]), 3) == 4.641


def test_exact_observations_in_text():
    I = exact_index()
    sizes = sorted({k[:4] for k in I})
    dist = [s for s in sizes if s[0] == "dist" and s[3] >= 1]
    assert len(dist) == STATED_DIST_SIZES
    for s in dist:
        for w in [str(x) for x in range(s[3] + 1)] + ["worst"]:
            assert I[s + (w, "weighted")] <= I[s + (w, "order")]
    gaps = [I[s + ("0", "weighted")] - I[s + ("0", "optimal")] for s in dist if s + ("0", "optimal") in I]
    assert gaps and max(gaps) == I[("dist", 4, 7, 1, "0", "weighted")] - I[("dist", 4, 7, 1, "0", "optimal")]
    assert round(float(max(gaps)), 4) == 0.0055
    assert [s for s in dist if I[s + ("0", "weighted")] > I[s + ("0", "leader")]] == [("dist", 8, 6, 1)]
    full = [s for s in sizes if s[0] == "all" and s[2] > s[3] >= 1]
    worse = [s for s in full if I[s + ("0", "weighted")] > I[s + ("0", "order")]]
    assert (len(worse), len(full)) == (31, 109)
    assert all(s[2] <= 2 * s[3] + 2 for s in worse)


def test_sample_statements_in_text():
    from keytrees.elimination import overhead_slope, read_sweep

    rows = read_sweep(str(DATA / "sweep.csv"))
    assert [round(overhead_slope(rows, "4log", "weighted", t), 2) for t in (1, 2, 3)] == [0.16, 1.07, 1.63]
    exact = [((order_cost(2**16, t, 64) - 16) - (order_cost(2**8, t, 32) - 8)) for t in (1, 2, 3)]
    assert [round(x, 2) for x in exact] == [0.73, 1.31, 1.81]
    get = {(r["regime"], int(r["m"]), int(r["t"]), r["errors"], r["rule"]): float(r["mean_tests"]) for r in rows}
    assert round(get[("4log", 2**16, 3, "none", "weighted")], 2) == 24.00
    assert round(get[("4log", 2**16, 3, "none", "oracle")], 2) == 24.12
    assert round(oracle_bound(2**16, 64, 3), 2) == 22.00
