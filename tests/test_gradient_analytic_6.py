"""Checks of articles/gradient-bound/analytic/06-assembly.tex (in the codes repo).

Every constant stated in Theorem C and Corollary C is recomputed here in interval
arithmetic (mpmath.iv, outward rounding) or exact integers. The parameter search
that chose s_1 = 210, b = 1/20, T = 5/4 is not repeated; only the chosen values are
certified.
"""
from __future__ import annotations

import math
from fractions import Fraction as F

import pytest
from mpmath import iv

from keytrees.gradient_analytic_6 import (
    GAMMA, LOG2E, budget_upper, c1_iv, check_P, delta_needed, error_terms, eta_bar_iv,
    h_iv, hi, ivq, lipschitz_f, lo, m_star_iv, p_certify, p_float_scan, sigma_constant,
    _varpi_int,
)

B_, T_, S1 = F(1, 20), F(5, 4), 210
P_LO = F(10129, 10**6)
T2 = F(1012913, 10**6)
ELL = 8


def theta1():
    return ivq('1/2') + ivq(B_) / iv.sqrt(ivq(S1))


def beta():
    return ivq(B_) / iv.sqrt(ivq(S1))


def zeta():
    return 5 * beta() * LOG2E / (2 * (1 - iv.sqrt(theta1())))


def xi():
    return 2 * beta() * LOG2E / (1 - iv.sqrt(theta1()))


def w_l():
    return 4 * ivq(B_) ** 2 / (ivq(S1) * (1 - theta1()))


def test_gamma():
    assert F(86071, 10**6) < lo(GAMMA) and hi(GAMMA) < F(86072, 10**6)


def test_step3_constants_theorem_C():
    assert check_P(B_, T_, S1)
    assert ELL == math.ceil(math.log2(S1))
    assert hi(eta_bar_iv(B_, T_, S1)) <= F(25144, 10**5)
    M = m_star_iv(B_, T_, S1)
    assert hi(M) <= F(202583, 10**5)
    assert hi(M) / 2 <= T2
    c1 = c1_iv(B_, T_, S1)
    assert lo(c1) >= F(71660, 10**7)
    assert hi(w_l()) <= F(959, 10**7)
    assert hi(xi()) <= F(343, 10**4)
    assert hi(zeta()) <= F(429, 10**4)
    assert hi(lipschitz_f() * zeta()) <= F(166, 10**4)
    assert F(3862, 10**4) < lo(lipschitz_f()) and hi(lipschitz_f()) < F(3863, 10**4)


def test_p_certified():
    # Exact integer comparison on the whole (enlarged) range of Step 5, Lemma p.
    assert p_certify(S1, T2, B_, P_LO)
    v, (N, k, u) = p_float_scan(S1, T2, B_)
    num, den = _varpi_int(N, k, u)
    assert P_LO <= F(num, den) < P_LO + F(1, 10**6)


def test_budget_affine():
    u = budget_upper(0)
    v = budget_upper(1) - u
    assert hi(u) <= F(1714, 100) and lo(u) > 0
    assert hi(v) <= F(4548, 10**4)
    assert sigma_constant('3/4', 2, 2) == 34
    # B'(m) of Step 4 with the floor, against the affine bound, for m = 2^8 .. 2^60.
    for lam in range(8, 61):
        dstar = (lam - math.log2(34)) / math.log2(4 / 3)
        exact = (1 - h_iv(ivq('3/4'))) * (math.floor(dstar) + 1) + sum(
            (1 - h_iv(F(1, s)) for s in range(3, 34)), ivq(0))
        assert hi(exact) <= lo(budget_upper(lam)) + F(1, 10**30)


def test_delta0_values():
    eps = F(1, 250)
    for lam, bound in [(20, 17176), (64, 18071), (640, 21068)]:
        d = delta_needed(lam, B_, T_, S1, P_LO, eps, eps)
        assert hi(d) <= bound
        assert lo(d) > bound - 2


@pytest.mark.parametrize("eps,c,cp", [(F(1, 250), F(944, 10**4), F(249, 10**4)),
                                      (F(1, 20), F(1864, 10**4), F(117, 10**3))])
def test_corollary_constants(eps, c, cp):
    q = 1 - ivq(2) ** -11
    cat = ivq(1) / (34 * 211)
    lz = lipschitz_f() * zeta()
    cval = w_l() + (GAMMA + 2 * ivq(eps) + cat) / q
    cpval = w_l() + lz + (GAMMA + lz) * ivq(2) ** -11 / q + (2 * ivq(eps) + cat) / q
    assert hi(cval) < c
    assert hi(cpval) < cp


def _lambda0(K, eps):
    c1 = c1_iv(B_, T_, S1)
    alpha = max(F(ELL) / (P_LO / 2), 1 / lo(c1))
    start = max(10, math.ceil(alpha / (K - 1)))

    def ok(lam):
        return (K - 1) * lam >= hi(delta_needed(lam, B_, T_, S1, P_LO, eps, eps))
    lo_, hi_ = start, start
    while not ok(hi_):
        hi_ *= 2
    if ok(lo_):
        return lo_
    while hi_ - lo_ > 1:
        mid = (lo_ + hi_) // 2
        lo_, hi_ = (lo_, mid) if ok(mid) else (mid, hi_)
    return hi_


@pytest.mark.parametrize("K,l1,l2", [(3, 12861, 10723), (10, 2578, 2099), (30, 734, 585),
                                     (60, 342, 269), (100, 196, 153)])
def test_lambda0_table(K, l1, l2):
    assert _lambda0(K, F(1, 250)) == l1
    assert _lambda0(K, F(1, 20)) == l2


def test_master_inequality_at_threshold():
    # Prop. master evaluated directly at n - log m = Delta_0(64): below c of Corollary C.
    lam = 64
    d = math.ceil(hi(delta_needed(lam, B_, T_, S1, P_LO, F(1, 250), F(1, 250))))
    t = error_terms(lam, lam + d, B_, T_, S1, P_LO)
    assert hi(t["total"]) < F(944, 10**4)
    assert hi(t["B"] * t["PF"]) <= F(1, 250) and hi(t["B"] * t["PL"]) <= F(1, 250)


def test_monotone_in_s1_fixed_angle():
    # Lemma mono: with theta^0 fixed, c_1 does not decrease and M* does not increase
    # (grid check of the proved statement).
    th0 = theta1()
    prev_c, prev_m = None, None
    for s1 in [210, 300, 1000, 10**4, 10**6]:
        assert check_P(B_, T_, s1, th0)
        c = c1_iv(B_, T_, s1, th0)
        M = m_star_iv(B_, T_, s1, th0)
        if prev_c is not None:
            assert lo(c) >= hi(prev_c) - F(1, 10**40)
            assert hi(M) <= lo(prev_m) + F(1, 10**40)
        prev_c, prev_m = c, M


def _C(s):
    a = s.bit_length() - 1
    return a + 2 - F(2 ** (a + 1), s)


def test_f_is_periodic_lipschitz_function_of_log():
    L = float(hi(lipschitz_f()))
    vals = {}
    for s in range(1, 2**12):
        f = float(_C(s)) - math.log2(s)
        x = s / 2 ** (s.bit_length() - 1)
        assert abs(f - (2 - 2 / x - math.log2(x))) < 1e-12
        vals[s] = f
    for s in range(1, 2**11):
        assert abs(vals[2 * s] - vals[s]) < 1e-12
    ss = sorted(range(2**10, 2**11))
    for a, b in zip(ss, ss[1:]):
        assert abs(vals[b] - vals[a]) <= L * (math.log2(b) - math.log2(a)) + 1e-12
    assert max(vals.values()) <= float(hi(GAMMA))


def test_remark_compare_numbers():
    # Remark compare: ell/pi about 1580 dominates; the Step 3 term alone is about 2e3.
    c1 = c1_iv(B_, T_, S1)
    assert 1579 < ELL / (P_LO / 2) < 1580
    assert P_LO / 2 == F(50645, 10**7)
    B = budget_upper(64)
    dL = iv.log(B / (c1 * ivq(F(1, 250)))) / c1 - iv.log(ivq(S1)) / iv.log(2) + xi()
    assert 1900 < lo(dL) and hi(dL) < 2100
    # Theorem B remark: 1 - M*/100 >= 0.98 for M* = 1.76.
    assert 1 - F(176, 10**4) >= F(98, 100)


def test_theorem_b_constants_certified():
    """Theorem B (appendix of the article; Step 7 of the notes): b = 0.068, T = 2,
    theta_1 = 3/5, sigma_1 = 0.99 * 10^5, hole radius b_h = b/sqrt(0.99), window b_w = b.
    In interval arithmetic: eta_bar < 0.178 < 2/5, M* <= 1.81 <= 2.5, c_1 >= 0.0338.
    These bound the constants for every s_1 >= 10^5, as they are monotone in sigma_1."""
    from keytrees.gradient_analytic_6 import filter_variant_iv
    b = ivq('0.068')
    eb, M, c1 = filter_variant_iv(b / iv.sqrt(ivq('0.99')), F('0.068'), 2, 99000, F(3, 5))
    assert eb.b < 0.178
    assert M.b <= 1.81
    assert c1.a >= 0.0338
    # monotone in sigma_1: larger sigma_1 gives smaller eta_bar and M*, larger c_1
    eb2, M2, c12 = filter_variant_iv(b / iv.sqrt(ivq('0.99')), F('0.068'), 2, 10**7, F(3, 5))
    assert eb2.b <= eb.b and M2.b <= M.b and c12.a >= c1.a
