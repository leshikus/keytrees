"""Checks of every number stated in articles/random-algorithm-bound/article.tex."""
from __future__ import annotations

import csv
import doctest
import math
from fractions import Fraction
from math import comb
from pathlib import Path

import pytest

import keytrees.random_bound as rb
from keytrees.random_bound import (
    ALPHA,
    C_PATRICIA,
    C_STAR,
    convex_order_gaps,
    finite_costs,
    gap_certificate,
    invariants,
    limit_costs,
    log2_bounds,
    ln2_bounds,
    mellin_offset,
    patricia_closed_form,
    second_differences,
    split_weights,
    step_certificate,
    step_function,
    step_holds,
    verify_base,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "random-algorithm-bound"
BASE_N = 8  # base of the induction in the monotonicity theorem
T = finite_costs(BASE_N + 1)


def rows(name: str) -> list[dict[str, str]]:
    with open(DATA / name, newline="") as fh:
        return list(csv.DictReader(fh))


def test_doctests():
    result = doctest.testmod(rb, optionflags=doctest.ELLIPSIS)
    assert result.attempted > 0 and result.failed == 0


# --- Section 2: recurrence and small values ---------------------------------


def test_small_values():
    for n in range(2, BASE_N + 1):
        assert T[n][2] == 1 and T[n][3] == Fraction(5, 3) and T[n][2**n] == n
    lim = limit_costs(6)
    assert lim[4] == Fraction(16, 7) and lim[5] == Fraction(274, 105)
    assert T[5][5] == Fraction(2593, 1015)
    assert T[7][6] == Fraction(1915322, 667275)


def test_kraft_lower_bound():
    assert all(T[n][m] >= math.log2(m) for n in range(1, BASE_N + 1) for m in range(1, 2**n + 1))


# --- Section 3: convex order --------------------------------------------------


@pytest.mark.parametrize("n", range(1, 8))
def test_convex_order_gaps(n):
    for m in range(2, 2**n + 1):
        D = convex_order_gaps(n, m)
        assert D[0] == 0 and D[m] == 0
        assert all(d >= 0 for d in D)
        assert all(D[j] == D[m - j] for j in range(m + 1))
        assert all(D[j] <= D[j + 1] for j in range((m + 1) // 2))


def test_gap_identity():
    """Sum (q_k - p_k) phi(k) = sum_j D(j) Delta^2 phi(j)."""
    n, m = 4, 9
    p, q = split_weights(n, m), split_weights(n + 1, m)
    phi = [Fraction(k**3 - 7 * k, 5) for k in range(m + 1)]
    D = convex_order_gaps(n, m)
    lhs = sum((q[k] - p[k]) * phi[k] for k in range(m + 1))
    rhs = sum(D[j] * (phi[j - 1] - 2 * phi[j] + phi[j + 1]) for j in range(1, m))
    assert lhs == rhs


# --- Section 4: upper bounds for m = 4, 5 --------------------------------------


def test_small_m_upper_bounds():
    b = limit_costs(5)
    for m, top in ((4, Fraction(16, 7)), (5, Fraction(274, 105))):
        psi = [(k * b[k] + (m - k) * b[m - k]) / m for k in range(m + 1)]
        assert all(psi[j - 1] - 2 * psi[j] + psi[j + 1] > 0 for j in range(1, m))
        assert 1 + sum(Fraction(comb(m, k), 2**m) * psi[k] for k in range(m + 1)) == top
        assert all(T[n][m] <= top for n in range(3, BASE_N + 1))
    b5 = [(k * b[k] + (5 - k) * b[5 - k]) / 5 for k in range(6)]
    assert b5 == [Fraction(274, 105), Fraction(64, 35), Fraction(7, 5), Fraction(7, 5),
                  Fraction(64, 35), Fraction(274, 105)]


# --- Section 5: the joint induction -------------------------------------------


@pytest.mark.parametrize("n", range(2, BASE_N + 1))
def test_base_monotone(n):
    assert all(T[n][m] >= T[n - 1][m] for m in range(2, 2 ** (n - 1) + 1))


@pytest.mark.parametrize("n", range(1, BASE_N + 1))
def test_base_convexity(n):
    inv = invariants(T[n])
    assert inv["negative"] == ([4] if n >= 4 else [])
    if n >= 3:
        assert inv["F1"] >= 0 and inv["F2"] >= 0


def test_second_difference_identity():
    """Delta^2 u_{n+1}(j) = 2 P(XY = 1) E[Delta^2 u_n(S+1) | XY = 1] for j >= 5."""
    n = 5
    d_n = second_differences(T[n])
    d_next = second_differences(T[n + 1])
    big, ones = 2 ** (n + 1), 2**n
    both = Fraction(ones * (ones - 1), big * (big - 1))
    for j in range(5, 2 ** (n + 1)):
        draws = j - 1
        total = comb(big - 2, draws)
        expect = sum(Fraction(comb(ones - 2, s) * comb(ones, draws - s), total) * d_n[s + 1]
                     for s in range(0, min(draws, ones - 2) + 1) if draws - s <= ones)
        assert d_next[j] == 2 * both * expect


def test_weight_ratios():
    for n in range(3, 12):
        a, b = 2**n - 2, 2**n
        for d in range(4, 8):
            assert Fraction((a - 2) * (d - 2), 3 * (b - d + 3)) <= Fraction(5, 3)
        if 2**n >= 9:
            assert all(Fraction((a - 3) * (d - 3), 4 * (b - d + 4)) >= 1 for d in range(8, b + 4))


def test_step_constants():
    t55, t76 = T[5][5], T[7][6]
    assert 10 * t55 - 12 * Fraction(16, 7) + 2 == Fraction(120, 1015)
    f2 = 5 - 4 * Fraction(16, 7) - 5 * Fraction(274, 105) + 6 * t76
    assert f2 > 0 and abs(float(f2) - 0.0317) < 1e-4
    assert 4 - 2 * Fraction(16, 7) == Fraction(-4, 7)


def test_gap_certificate():
    cert = gap_certificate(m=8, j=3, i=4, kappa=Fraction(1, 2), h0=128)
    assert len(cert) - 1 == 17 and all(c >= 0 for c in cert)
    for n in range(8, 13):
        D = convex_order_gaps(n, 8)
        assert 2 * D[3] >= D[4]


def test_finite_data_to_n12():
    data = rows("finite_r3.csv")
    assert [int(r["n"]) for r in data] == list(range(1, 13))
    for r in data:
        n = int(r["n"])
        assert int(r["monotone_violations"]) == 0
        assert r["negative_second_differences"] == ("4" if n >= 4 else "")
        if n >= 3:
            assert float(r["F1"]) > 0.5 and float(r["F2"]) > 0.14
    assert abs(float(data[-1]["d2u_4"]) + 0.2379) < 1e-4


# --- Section 7: the uniform bound ----------------------------------------------


def test_base_range():
    for s, c in ((3, C_STAR), (None, C_PATRICIA)):
        r = verify_base(c, ALPHA, 8, rb.BASE_RANGE, s)
        assert r["ok"]
    assert C_STAR == Fraction(4311, 10000) and C_PATRICIA == Fraction(33276, 100000)
    data = {r["variant"]: r for r in rows("base_check.csv")}
    assert data["s3"]["ok"] == "True" and data["s3"]["argmin"] == "92"
    assert data["patricia"]["ok"] == "True" and data["patricia"]["argmin"] == "139"
    assert abs(float(data["s3"]["min_margin"]) - 7.72e-6) < 1e-8


def test_step_lemma():
    assert all(c >= 0 for c in step_certificate(16))
    l2lo, _ = ln2_bounds(64)
    log2e = 1 / l2lo
    lhs = log2e * (Fraction(1, 2) + Fraction(1, 2050))
    assert lhs < Fraction(72206, 100000) < ALPHA
    # tail: alpha m P(Bin(m-1,1/2) <= 6) <= 7 alpha m^7 / (720 * 2^(m-1)), tiny for m >= 1025
    m = 1025
    assert Fraction(7 * m**7, 720 * 2 ** (m - 1)) < Fraction(1, 10**280)
    assert all(step_holds(m, ALPHA) for m in (1025, 1500, 4096))
    assert not step_holds(200, ALPHA)
    # threshold of the step with these constants
    assert abs(float(step_function(10**6)) - 0.5 - 0.25e-6) < 1e-9


def test_rigorous_sup_lower_bound():
    data = [r for r in rows("enclosures.csv") if r["variant"] == "s3"]
    assert int(data[-1]["m"]) == 8192
    near = [r for r in data if 6400 <= int(r["m"]) <= 6470]  # floats locate the peak; certify exactly
    best = max(near, key=lambda r: Fraction(int(r["lo"]), 10**30) - log2_bounds(int(r["m"]))[1])
    m = int(best["m"])
    _, hi = log2_bounds(m)
    lower = Fraction(int(best["lo"]), 10**30) - hi
    assert m == 6434 and Fraction(4309616, 10**7) < lower < C_STAR
    # the enclosures agree with the Python ones where both exist
    lo, hi2, bits = rb.limit_enclosures(200, 3, 96)
    assert int(data[198]["m"]) == 200
    assert Fraction(int(data[198]["lo"]), 10**30) <= Fraction(hi2[200], 2**bits)
    assert Fraction(lo[200], 2**bits) <= Fraction(int(data[198]["hi"]), 10**30)


def test_limit_profile():
    data = rows("limit_profile.csv")
    r3 = {int(r["m"]): float(r["r_s3"]) for r in data}
    rp = {int(r["m"]): float(r["r_patricia"]) for r in data}
    assert max(r3.values()) < float(C_STAR) and max(rp.values()) < float(C_PATRICIA)
    best = max((r3[m] + 0.723 / m, m) for m in r3 if m >= 8)
    assert best[1] == 92 and abs(best[0] - 0.4310923) < 1e-6
    bestp = max((rp[m] + 0.723 / m, m) for m in rp if m >= 8)
    assert bestp[1] == 139 and abs(bestp[0] - 0.3327540) < 1e-6
    assert abs(r3[32768] - 0.43103) < 1e-4


def test_markov_corollary():
    assert abs(1 - float(C_STAR) / (0.5 * 10) - 0.91378) < 1e-5
    assert abs(1 - (29 / 28) / (0.5 * 10) - 0.793) < 1e-3


# --- PATRICIA -------------------------------------------------------------------


def test_patricia_closed_form_and_monotone():
    tp = finite_costs(7, s=None)
    assert all(patricia_closed_form(n, m) == tp[n][m] for n in range(1, 8) for m in range(1, 2**n + 1))
    for n in range(2, 10):
        assert all(patricia_closed_form(n, m) >= patricia_closed_form(n - 1, m)
                   for m in range(2, 2 ** (n - 1) + 1))


# --- Section 8: asymptotics ---------------------------------------------------------


def test_mellin_constants():
    gamma = 0.57721566490153286061
    ln2 = math.log(2)
    assert [mellin_offset(s) for s in (1, 2, 3)] == [0, Fraction(-1, 2), Fraction(-5, 8)]
    kappa = {s: gamma / ln2 + 0.5 + float(mellin_offset(s)) / ln2 for s in (1, 2, 3)}
    # values printed in Prodinger and Szpankowski (preprint of 27 May 2001)
    assert abs(kappa[1] - 1.332746177) < 1e-9
    assert abs(kappa[2] - 0.6113986565) < 1e-9
    assert abs(kappa[3] - 0.4310617764) < 1e-9
    assert abs((gamma / ln2 - 0.5) - 0.332746177) < 1e-9
    data = {r["variant"]: r for r in rows("constants.csv")}
    assert abs(float(data["s3"]["max_P"]) - 1.709e-5) < 1e-8
    assert abs(float(data["s1"]["max_P"]) - 1.573e-6) < 1e-9
    assert abs(kappa[3] + float(data["s3"]["max_P"]) - 0.4310789) < 1e-7


def test_asymptotic_residual():
    """(r_m - kappa - P(log m)) m is close to -0.722 for 100 <= m <= 8192."""
    import cmath

    data = {int(r["m"]): float(r["r_s3"]) for r in rows("limit_profile.csv")}
    gamma, ln2 = 0.57721566490153286061, math.log(2)
    kappa = gamma / ln2 + 0.5 - 5 / (8 * ln2)
    coef = rb.fourier_coefficients(3, 3)
    for m in (128, 1024, 4096, 8192):
        x = math.log2(m)
        p = 2 * sum((c * cmath.exp(-2j * math.pi * (k + 1) * x)).real for k, c in enumerate(coef))
        assert abs((data[m] - kappa - p) * m + 0.722) < 2e-3


def test_remaining_numbers():
    lim = limit_costs(6)
    assert 5 - 8 * lim[4] + 5 * lim[5] == Fraction(-5, 21)
    assert abs(float(-5 / 21) + 0.238) < 1e-3
    gamma, ln2 = 0.57721566490153286061, math.log(2)
    data = {r["variant"]: r for r in rows("constants.csv")}
    assert abs(gamma / ln2 - 0.5 + float(data["s1"]["max_P"]) - 0.3327478) < 1e-7
    assert abs(float(data["s2"]["max_P"]) - 7.30e-6) < 1e-8
    assert Fraction(1, 2**1024) < Fraction(1, 10**300)
    assert 1.4427 * 0.50049 < 0.72206 and 1 / math.log(2) < 1.4427
    kappa2 = gamma / ln2 + 0.5 - 1 / (2 * ln2)
    kappa3 = gamma / ln2 + 0.5 - 5 / (8 * ln2)
    assert f"{kappa2:.10f}" == "0.6113986568" and f"{kappa3:.10f}" == "0.4310617767"
    assert abs(kappa2 - 0.6113986565) < 4e-10 and abs(kappa3 - 0.4310617764) < 4e-10
