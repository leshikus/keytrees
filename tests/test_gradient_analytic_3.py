"""Checks of Step 3 (articles/gradient-bound/analytic/03-filtering.tex).

Exact checks (rational arithmetic) of the identities used in the proofs, and floating-point
stress tests of the bounds on filtered laws. The floating-point checks are evidence for the
statements, not part of their proofs.
"""
from __future__ import annotations

import doctest
import math
import random
from fractions import Fraction as F

import numpy as np
import pytest
from scipy.special import erf
from scipy.stats import norm

import keytrees.gradient_analytic_3 as g3


def test_doctests():
    result = doctest.testmod(g3, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0
    assert result.failed == 0


# ------------------------------------------------------------------- exact lemmas

@pytest.mark.parametrize("s", [2, 3, 5, 8, 13, 20])
def test_one_step_mean_identity(s):
    # Lemma: E[4(X-a/2)^2/a] = 1 + (a-1)/(s-1) (v-1), v = 4(k-s/2)^2/s.
    for a in range(1, s):
        for k in range(s + 1):
            law = g3.hyp_pmf_exact(s, k, a)
            lhs = sum(p * 4 * (x - F(a, 2)) ** 2 / a for x, p in law.items())
            v = 4 * (k - F(s, 2)) ** 2 / s
            assert lhs == 1 + F(a - 1, s - 1) * (v - 1)


@pytest.mark.parametrize("s", [10, 31, 60])
def test_subgaussian_mgf(s):
    # Lemma: E exp(t v') <= (1-2t)^{-1/2} exp(t alpha v/(1-2t)).
    for a in (s // 3, s // 2, s - s // 3):
        for k in range(s + 1):
            law = g3.hyp_pmf_exact(s, k, a)
            v = 4 * (k - s / 2) ** 2 / s
            for t in (0.05, 0.2, 0.4):
                lhs = sum(float(p) * math.exp(t * 4 * (x - a / 2) ** 2 / a) for x, p in law.items())
                rhs = (1 - 2 * t) ** -0.5 * math.exp(t * (a / s) * v / (1 - 2 * t))
                assert lhs <= rhs * (1 + 1e-12)


@pytest.mark.parametrize("s,a", [(40, 20), (101, 50), (300, 140)])
def test_berry_esseen_and_logconcavity(s, a):
    for k in range(1, s):
        law = g3.hyp_pmf_exact(s, k, a)
        xs = sorted(law)
        mean = F(a * k, s)
        var = F(a * k * (s - k) * (s - a), s * s * (s - 1))
        sd = math.sqrt(var)
        if sd == 0:
            continue
        cdf = 0.0
        for x in xs:
            left = cdf
            cdf += float(law[x])
            z = (x - float(mean)) / sd
            assert abs(cdf - norm.cdf(z)) <= g3.BE_CONST / sd
            assert abs(left - norm.cdf(z)) <= g3.BE_CONST / sd
        for x0, x1, x2 in zip(xs, xs[1:], xs[2:]):
            assert law[x1] ** 2 >= law[x0] * law[x2]


def test_exact_recursion_small():
    # N_{l+1} = 1 + alpha'_l (M_l - 1) exactly, and M_l <= N_l/(1 - eta_l).
    rng = random.Random(3)
    for _ in range(30):
        m = rng.randint(8, 40)
        sizes = [m]
        while sizes[-1] > 3:
            s = sizes[-1]
            sizes.append(rng.randint(max(2, s // 2 - 2), min(s - 1, s // 2 + 2)))
        deltas = [F(rng.randint(0, 2)) + (F(1, 2) if s % 2 else 0) for s in sizes[:-1]]
        strict = [rng.random() < 0.5 for _ in deltas]
        lv = g3.filter_exact(m, sizes, deltas, strict)
        for l in range(len(sizes) - 1):
            assert lv[l]["M"] * (1 - lv[l]["eta"]) <= lv[l]["N"]
            if l + 1 < len(sizes) - 1:
                s, a = sizes[l], sizes[l + 1]
                assert lv[l + 1]["N"] == 1 + F(a - 1, s - 1) * (lv[l]["M"] - 1)


def test_float_filter_matches_exact():
    sizes, deltas = [60, 31, 16, 8], [F(1), F(1, 2), F(1)]
    strict = [True, False, True]
    ex = g3.filter_exact(60, sizes, deltas, strict)
    fl = g3.filter_float(60, sizes, [float(d) for d in deltas], strict, width=60)
    for l in range(3):
        assert abs(float(ex[l]["M"]) - fl[l]["M"]) < 1e-10
        assert abs(float(ex[l]["eta"]) - fl[l]["eta"]) < 1e-10


# ------------------------------------------------------------------- bounds

@pytest.mark.parametrize("s", [1024, 4096])
@pytest.mark.parametrize("b", [0.05, 0.1, 0.2])
def test_hole_sup_below_eta_bar(s, b):
    # sup over all parent counts of the hole mass (closed hole, largest delta), both parities.
    for a in (s // 2, s // 2 + 1):
        d = math.floor(b * math.sqrt(a) - (a % 2) / 2) + (a % 2) / 2
        hs = g3.hole_sup(s, a, d, True)
        assert hs <= g3.eta_bar(b, s - 1, 4.0)
        # the sup is close to erf(2b), not to the stationary value 1.6 b
        assert abs(hs - erf(2 * b)) < 0.06


@pytest.mark.parametrize("b", [0.1, 0.2])
@pytest.mark.parametrize("side", ["big", "small", "random"])
def test_filtered_moments_adversarial(b, side):
    rng = random.Random(7)
    m, s_end = 2 ** 13, 64
    for pat in ("strict", "loose", "random"):
        sizes, deltas = g3.adversarial_sequence(m, b, s_end, rng, side)
        strict = [pat == "strict" or (pat == "random" and rng.random() < 0.5) for _ in deltas]
        lv = g3.filter_float(m, sizes, [float(x) for x in deltas], strict, b_good=b, width=12)
        assert lv[-1]["lost"] < 1e-9
        Ms = [r["M"] for r in lv[:-1]]
        goods = [r["good"] for r in lv[1:]]
        etas = [r["eta"] for r in lv[:-1]]
        # limit constants (evidence at finite sizes, not a proof)
        assert max(Ms) <= 1 / (1 - 2 * erf(2 * b))
        assert max(etas) <= erf(2 * b) + 0.03
        assert min(goods) >= g3.c1_limit(b)


@pytest.mark.parametrize("b", [0.05, 0.1, 0.15, 0.2])
def test_gaussian_limit_above_proved_constant(b):
    eta, M, good = g3.gaussian_filter_limit(b)
    assert M <= 1 / (1 - 2 * erf(2 * b))
    assert good >= g3.c1_limit(b)
    assert eta <= erf(2 * b)


def test_c1_values_stated():
    # values quoted in 03-filtering.tex and STATUS-3.md
    for cited, b_best, c_best in ((False, 0.1095, 0.0905), (True, 0.1125, 0.0920)):
        bs = np.linspace(0.005, 0.24, 471)
        vals = [g3.c1_limit(b, cited) for b in bs]
        i = int(np.argmax(vals))
        assert abs(bs[i] - b_best) < 0.002 and abs(vals[i] - c_best) < 1e-4
    assert 0.0894 < g3.c1_limit(0.1) < 0.0896
    b_free, b_cited = math.sqrt(math.pi) / 8, 0.5 * 0.4769362762044699
    assert g3.c1_limit(b_free - 0.01) > 0 and g3.c1_limit(b_free + 1e-6) == 0
    assert g3.c1_limit(b_cited - 0.01, True) > 0 and g3.c1_limit(b_cited + 1e-6, True) == 0
    for s1, b, T, lo in ((1e4, 0.106, 1.5, 0.0743), (1e5, 0.108, 1.5, 0.0856), (1e6, 0.11, 2.0, 0.0891)):
        c = g3.c1_bound(b, s1, T, g3.m_star(b, s1, T))
        assert lo < c < lo + 0.0002
    assert g3.m_star(0.1, 1e12, 10, cited=True) < 1.81


def test_step2_helpers_agree():
    # the float versions of A and Lambda agree with the interval versions of Step 2
    from fractions import Fraction as Fr
    import keytrees.gradient_analytic_2 as g2
    for b, T, s, th in ((Fr(1, 10), Fr(2), 10**6, Fr(3, 5)), (Fr(1, 4), Fr(1), 10**5, Fr(11, 20))):
        lam = g2.correction(b, T, s, th)
        assert float(lam.lo) - 1e-12 <= g3.big_lambda(float(b), float(T), s, float(th)) <= float(lam.hi) + 1e-12
        A = g2.amax(T, s, th)
        assert float(A.lo) - 1e-12 <= g3.a_max(float(T), s, float(th)) <= float(A.hi) + 1e-12


def test_c1_bound_below_exact_window():
    # Theorem filter-c1 pointwise ingredient: P(W) >= 2b sqrt(2/pi) Lambda g_hat(x^2), exact.
    s, b, T = 4000, 0.1, 1.5
    for a in (1990, 2000, 2013):
        for k in range(int(s / 2 - T * math.sqrt(s)), int(s / 2 + T * math.sqrt(s)) + 1, 7):
            x2 = (k - s / 2) ** 2 / s
            law = g3.hyp_pmf_exact(s, k, a)
            w = sum(float(p) for x, p in law.items() if abs(x - a / 2) <= b * math.sqrt(a))
            th = max(a / s, 1 - a / s)
            ap, am = th, 1 - th
            ghat = math.exp(-(2 * ap * x2 + 2 * b * b / 3) / am) / math.sqrt(ap)
            assert w >= 2 * b * math.sqrt(2 / math.pi) * g3.big_lambda(b, T, s, th) * ghat


# ------------------------------------------------------------------- fixed-set variant (Step 7)

def test_subset_form_fails():
    # C'_0 = {+-s/2}: Y constant on S'_0, so the next count is 0 or a and the window is empty.
    s, a, b = 40, 20, 0.2
    law = {0: F(1, 2), s: F(1, 2)}
    m2 = sum(p * F(4) * (k - F(s, 2)) ** 2 / s for k, p in law.items())
    assert m2 == s
    nxt = {}
    for k, p in law.items():
        for x, q in g3.hyp_pmf_exact(s, k, a).items():
            nxt[x] = nxt.get(x, 0) + p * q
    assert sum(p for x, p in nxt.items() if abs(x - a / 2) <= b * math.sqrt(a)) == 0


@pytest.mark.parametrize("pattern", ["low", "high", "alternate"])
def test_variant_ratios_04_06(pattern):
    # exact filtered laws with ratios near 0.4 / 0.6 and holes of radius b_h sqrt(s)
    b = 0.068
    bh = b / math.sqrt(0.99)
    sizes = [2 ** 14]
    while sizes[-1] > 200:
        s = sizes[-1]
        r = {"low": 0.41, "high": 0.59}.get(pattern, 0.41 if len(sizes) % 2 else 0.59)
        sizes.append(int(r * s))
    deltas = [math.floor(bh * math.sqrt(s)) + (s % 2) / 2 for s in sizes[:-1]]
    deltas = [d if d <= bh * math.sqrt(s) else d - 1 for d, s in zip(deltas, sizes)]
    lv = g3.filter_float(sizes[0], sizes, deltas, [True] * len(deltas), b_good=b, width=12)
    Ms = [r["M"] for r in lv[:-1]]
    goods = [r["good"] for r in lv[1:]]
    assert max(Ms) <= g3.m_star_theta(bh, 1e12, 10, 0.6) + 1e-9   # evidence, limit constant
    assert min(goods) >= g3.c1_bound_theta(b, 1e12, 2.5, g3.m_star_theta(bh, 1e12, 2.5, 0.6), 0.6)


def test_variant_constants_stated():
    for s1, T, lo in ((1e12, 2.5, 0.0371), (1e6, 2.0, 0.0363), (1e5, 2.0, 0.0339)):
        M = g3.m_star_theta(0.068 / math.sqrt(0.99), s1, T, 0.6)
        assert lo < g3.c1_bound_theta(0.068, s1, T, M, 0.6) < lo + 0.0002
    assert abs(0.4 / (2 * math.sqrt(0.6) * math.sqrt(2 / (math.pi * 0.24))) - 0.1585) < 1e-4
    assert abs(g3.m_star_theta(0.022, 1e12, 10, 0.6) - 1.16) < 0.01
