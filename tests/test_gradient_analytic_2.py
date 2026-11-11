"""Exact checks of articles/gradient-bound/analytic/02-hypergeometric.tex.

Probabilities are fractions; exp, sqrt and pi are enclosed by rational intervals,
so every comparison below is exact. A grid check is a check, not a proof.
"""
from __future__ import annotations

import doctest
from fractions import Fraction as F
from math import comb

import pytest

import keytrees.gradient_analytic_2 as g

THETA = F(3, 5)
BS = [F(1, 20), F(1, 10), F(1, 5), F(1, 3), F(49, 100)]
TS = [F(0), F(1, 4), F(1, 2), F(1), F(2)]


def a_range(s):
    return [a for a in range(1, s) if (1 - THETA) * s <= a <= THETA * s]


def k_range(s, T):
    return [k for k in range(s + 1) if (k - F(s, 2)) ** 2 <= T * T * s]


def half_lattice(n):
    """All z = j - n/2, j = 0..n."""
    return [j - F(n, 2) for j in range(n + 1)]


def test_doctests():
    result = doctest.testmod(g, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0
    assert result.failed == 0


def test_part_a_identity():
    for s in range(2, 41):
        for k in range(s + 1):
            for a in range(1, s + 1):
                assert g.second_moment(s, k, a) == g.second_moment_formula(s, k, a)


def test_central_binomial_two_sided():
    for n in range(1, 241):
        for z in half_lattice(n):
            if z < 0:
                continue
            value = F(comb(n, int(F(n, 2) + z)), 2**n)
            assert g.central_lower(n, z).hi <= value <= g.central_upper(n, z).lo


def local_bounds(s, k, a, x):
    """Right sides of the two local inequalities at x."""
    e = k - F(s, 2)
    y = x - F(a, 2)
    w = e - y
    ap, bp = a + 1, s - a + 1
    kappa = F(2 * (s + 2), ap * bp)
    m = F(ap, s + 2) * e
    quad = kappa * (y - m) ** 2
    ws = max(2 * abs(e) - 1, F(0)) / (s + 1)
    up_err = 2 * e * e / ((s + 1) * (s + 2)) + (2 * e * e / (s + 1)) * ws**2 / (1 - ws**2)
    upper = g.SQRT_2_OVER_PI * g.iv_sqrt(g.Iv.of(F(s + 1, a * (s - a)))) * g.iv_exp(g.Iv.of(up_err - quad))
    w1 = max(2 * abs(y) - 1, F(0)) / ap
    w2 = max(2 * abs(w) - 1, F(0)) / bp
    lo_err = (2 * y * y / ap) * w1**2 / (1 - w1**2) + (2 * w * w / bp) * w2**2 / (1 - w2**2)
    lower = g.SQRT_2_OVER_PI * g.iv_sqrt(g.Iv.of(F(s, ap * bp))) * g.iv_exp(g.Iv.of(-quad - lo_err))
    return lower, upper


@pytest.mark.parametrize("s", list(range(2, 25)) + [47, 80])
def test_local_bounds(s):
    step = 1 if s < 25 else 7
    for a in range(1, s, step):
        for k in range(0, s + 1, step):
            for x, p in g.pmf(s, k, a).items():
                lower, upper = local_bounds(s, k, a, x)
                assert lower.hi <= p <= upper.lo


@pytest.mark.parametrize("T", TS)
def test_part_b(T):
    for s in list(range(max(2, int(4 * T * T) + 1), 121)) + [300]:
        if s <= 4 * T * T:
            continue
        ks, As = k_range(s, T), a_range(s)
        if not ks or not As:
            continue
        worst = max(max(g.pmf(s, k, a).values()) for k in ks for a in As)
        assert worst**2 * s <= g.amax(T, s).lo ** 2


def test_part_b_simple_constant():
    # A(T, s) <= 1.81 for s >= 100 (1+T^2)^2; <= 1.64 for s >= 2000 (1+T)^4.
    for num in range(0, 401):
        T = F(num, 20)
        assert g.amax(T, int(100 * (1 + T * T) ** 2) + 1).hi <= F(181, 100)
        assert g.amax(T, int(2000 * (1 + T) ** 4) + 1).hi <= F(164, 100)
    assert g.amax(F(0), 100).hi <= F(181, 100)


def in_range(b, T, s):
    if b * b * (1 - THETA) * s < F(1, 4):
        return False
    R = g._r_upper(b, T, THETA)
    gap = g.Iv.of(1 - THETA)
    return (g.Iv.of(4) * R * R / (gap * gap * g.Iv.of(s))).hi < 1


def window_minima(s, a_step=1):
    """For each (a, b, T): the least window probability over k with e^2 <= T^2 s."""
    tmax = max(TS)
    out = {}
    for a in a_range(s)[::a_step]:
        total = comb(s, a)
        for k in k_range(s, tmax):
            nums = {x: comb(k, x) * comb(s - k, a - x) for x in range(max(0, a + k - s), min(a, k) + 1)}
            e2 = (k - F(s, 2)) ** 2
            for b in BS:
                p = F(sum(n for x, n in nums.items() if (x - F(a, 2)) ** 2 <= b * b * a), total)
                for T in TS:
                    if e2 <= T * T * s:
                        key = (a, b, T)
                        out[key] = min(out.get(key, p), p)
    return out


@pytest.mark.parametrize("s", list(range(2, 101)) + [160, 257])
def test_part_c_grid(s):
    minima = window_minima(s, 1 if s <= 100 else 9)
    for b in BS:
        for T in TS:
            if not in_range(b, T, s):
                continue
            lam = g.correction(b, T, s)
            c = g.c_bound(b, T, s)
            for a in a_range(s)[:: 1 if s <= 100 else 9]:
                p = minima.get((a, b, T))
                if p is None:  # no k with |k - s/2| <= T sqrt(s)
                    continue
                assert c.hi <= p
                assert (g.Iv.of(2 * b) * g.SQRT_2_OVER_PI * g.g_alpha(F(a, s), b, T) * lam).hi <= p


@pytest.mark.parametrize("b", [F(1, 10), F(1, 3), F(49, 100)])
@pytest.mark.parametrize("T", [F(0), F(1, 2), F(1)])
def test_part_c_master(b, T):
    for s in (120, 201, 400):
        if not in_range(b, T, s):
            continue
        for a in a_range(s)[::7]:
            for k in k_range(s, T)[::3]:
                assert g.master_lower(s, k, a, b, T).hi <= g.window_probability(s, k, a, b)


def test_part_c_simple_correction():
    # Lambda >= 4/5 at s = max(100/b^2, 2000 (1+T)^4); Lambda increases with s.
    for b in [F(i, 100) for i in range(1, 50)]:
        for T in [F(i, 4) for i in range(0, 41)]:
            s = -int(-max(100 / b**2, 2000 * (1 + T) ** 4) // 1)
            assert g.correction(b, T, s).lo >= F(4, 5)


def test_part_c_closed_form_at_threshold():
    # Full-size instances at the threshold: s = 2002 (T = 0) and 2432 (T = 1/20), b = 1/4.
    b = F(1, 4)
    for T in (F(0), F(1, 20)):
        s = int(2000 * (1 + T) ** 4) + 1
        s += s % 2  # even s, so that k = s/2 exists
        c = g.c_bound(b, T, s)
        closed = F(127, 100) * b * g.g_min(b, T).hi
        assert closed <= c.lo
        for a in (-(-2 * s // 5), s // 2, 3 * s // 5):
            for k in (k_range(s, T)[0], k_range(s, T)[-1]):
                assert c.hi <= g.window_probability(s, k, a, b)


def test_g_min_at_ends():
    for b in BS:
        for T in TS + [F(7, 20), F(3, 10)]:
            m = g.g_min(b, T)
            for i in range(40, 61):
                assert m.lo <= g.g_alpha(F(i, 100), b, T).hi


def test_simple_c_form():
    # g >= theta^{-1/2} exp(-3T^2 - 5b^2/3), and 1.27 / sqrt(.6) >= 1.6.
    assert g.iv_sqrt(g.Iv.of(THETA)).hi * F(16, 10) <= F(127, 100)
    for b in BS:
        for T in TS:
            simple = g.iv_exp(g.Iv.of(-3 * T * T - F(5, 3) * b * b)) / g.iv_sqrt(g.Iv.of(THETA))
            assert simple.hi <= g.g_min(b, T).lo + F(1, 2**120)


def test_window_can_be_empty():
    # b sqrt(a) < 1/2 with a odd: no lattice point, so s_* must depend on b.
    b = F(1, 10)
    assert g.window_probability(50, 25, 23, b) == 0
    # k far from s/2: the support misses the window, so s_* must depend on T.
    assert g.window_probability(100, 100, 60, F(1, 2)) == 0  # e = 50 = 5 sqrt(s)


def test_part_d_proxy():
    for s in range(2, 200):
        for a in range(1, s):
            v = g.variance_proxy(s, a)
            assert v <= a
            u = s - a
            assert v <= F(a * u * u * 4, (2 * u - 1) * (2 * s - 1))
            if a >= 2 and u >= 3:
                assert v <= F(a * (u + 1), s) <= F((s + 1) ** 2, 4 * s)


@pytest.mark.parametrize("s", list(range(2, 41)) + [64, 101])
def test_part_d_tail(s):
    for a in range(1, s):
        v = min(g.variance_proxy(s, a), g.variance_proxy(s, s - a))
        for k in range(s + 1):
            law = g.pmf(s, k, a)
            mu = F(a * k, s)
            for t in (F(1, 2), F(1), F(3, 2), F(2), F(3), F(5)):
                bound = g.iv_exp(g.Iv.of(-2 * t * t / v)).lo
                assert sum((p for x, p in law.items() if x - mu >= t), F(0)) <= bound
                assert sum((p for x, p in law.items() if mu - x >= t), F(0)) <= bound


def test_binomial_window_upper():
    # P(|Bin(s,1/2) - s/2| <= u) <= (2u+1) sqrt(2/(pi s)).
    for s in range(1, 200):
        for u2 in range(0, 2 * s + 1, 3):
            u = F(u2, 4)
            p = sum(F(comb(s, j), 2**s) for j in range(s + 1) if abs(j - F(s, 2)) <= u)
            assert p <= (2 * u + 1) * (g.SQRT_2_OVER_PI / g.iv_sqrt(g.Iv.of(s))).lo
