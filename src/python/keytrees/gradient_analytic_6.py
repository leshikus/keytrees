r"""
Step 6 of the analytic estimate for the gradient key: the constants of the assembly.

See ``articles/gradient-bound/analytic/06-assembly.tex``. Every constant that enters a
stated inequality is computed here either exactly (fractions, integers) or as an
``mpmath.iv`` interval with outward rounding; a claim ``x <= c`` is checked as
``x.b <= c`` (upper end) and ``x >= c`` as ``x.a >= c`` (lower end).

The functions follow the step files:

- Step 2 / Step 3: :func:`a_iv`, :func:`lambda_iv`, :func:`eta_bar_iv`, :func:`m_star_iv`,
  :func:`c1_iv` (Theorem filter-c1 of Step 3, standing assumptions (P) checked by
  :func:`check_P`).
- Step 4: :func:`sigma_constant`, :func:`budget_upper` (Prop. cat-budget, with the
  floor of `d^*` dropped).
- Step 5: :func:`p_float_scan` and :func:`p_certify` (the exact minimum `p(s_1,T,b)` of
  Lemma p, certified by integer comparisons).
- Step 6: :func:`error_terms`, :func:`delta_needed`, :func:`lipschitz_f`.

EXAMPLES::

    >>> from fractions import Fraction as F
    >>> from keytrees.gradient_analytic_6 import c1_iv, GAMMA
    >>> c = c1_iv(F(1, 20), F(5, 4), 210)
    >>> 0.0071 < float(c.a) <= float(c.b) < 0.0072
    True
    >>> 0.0860 < float(GAMMA.a) <= float(GAMMA.b) < 0.0861
    True
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
from fractions import Fraction as F

from mpmath import iv

iv.dps = 50

__all__ = [
    "ivq", "lo", "hi", "GAMMA", "LOG2E", "a_iv", "lambda_iv", "eta_bar_iv", "m_star_iv", "c1_iv",
    "check_P", "h_iv", "sigma_constant", "budget_upper", "p_float_scan", "p_certify",
    "p_range", "lipschitz_f", "error_terms", "delta_needed", "q_d_lower",
]


def ivq(x) -> iv.mpf:
    r"""
    An exact rational (or integer) as a thin interval.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_6 import ivq
        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_6 import lo, hi
        >>> x = ivq('1/3'); lo(x) <= F(1, 3) <= hi(x)
        True
    """
    q = F(x)
    return iv.mpf(q.numerator) / q.denominator


def _raw_to_fraction(raw) -> F:
    sign, man, exp, _ = raw
    v = F(man) * (F(2) ** exp)
    return -v if sign else v


def lo(x) -> F:
    """Lower end point of an interval, as an exact fraction."""
    return _raw_to_fraction(x._mpi_[0])


def hi(x) -> F:
    """Upper end point of an interval, as an exact fraction."""
    return _raw_to_fraction(x._mpi_[1])


LN2 = iv.log(2)
LOG2E = 1 / LN2
#: `\gamma = 2 - 1/\ln 2 - \log_2(2\ln 2)`.
GAMMA = 2 - 1 / LN2 - iv.log(2 * LN2) / LN2


def _max(x, y):
    return iv.mpf([max(x.a, y.a), max(x.b, y.b)])


def h_iv(x) -> iv.mpf:
    """Binary entropy (base 2) of an interval in (0,1)."""
    x = x if isinstance(x, iv.mpf) else ivq(x)
    return -(x * iv.log(x) + (1 - x) * iv.log(1 - x)) / LN2


# ----------------------------------------------------------------- Steps 2 and 3

def a_iv(T, s, theta) -> iv.mpf:
    r"""
    `A(T,s)` of Step 2, Theorem hg-max:
    `\sqrt{2(1+1/s)/(\pi\theta(1-\theta))}\exp(2T^2/s + 8T^4/(s-4T^2))`, for `s > 4T^2`.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_6 import a_iv, ivq
        >>> 1.62 < float(a_iv(ivq(0), 10**9, ivq('3/5')).a) < 1.63
        True
    """
    T = T if isinstance(T, iv.mpf) else ivq(T)
    theta = theta if isinstance(theta, iv.mpf) else ivq(theta)
    s = ivq(s)
    assert (s - 4 * T * T).a > 0
    return iv.sqrt(2 * (1 + 1 / s) / (iv.pi * theta * (1 - theta))) * iv.exp(
        2 * T * T / s + 8 * T ** 4 / (s - 4 * T * T))


def lambda_iv(b, T, s, theta) -> iv.mpf:
    r"""
    `\Lambda(b,T,s)` of Step 2 (before Theorem hg-window), for its range
    `4b^2(1-\theta)s \ge 1`, `\rho < 1`.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_6 import lambda_iv, ivq
        >>> 0.98 < float(lambda_iv(ivq('9/100'), ivq(3), 10**6, ivq('1/2') + ivq('9/100000')).a)
        True
    """
    b = b if isinstance(b, iv.mpf) else ivq(b)
    T = T if isinstance(T, iv.mpf) else ivq(T)
    gap = 1 - theta
    s = ivq(s)
    R = T + b * iv.sqrt(theta)
    rho = 2 * R / (gap * iv.sqrt(s))
    assert rho.b < 1 and (4 * b * b * gap * s).a >= 1
    u0 = b * iv.sqrt(gap * s)
    nu = _max(ivq('1/2'), 1 - 1 / (2 * u0))
    return nu / (1 + 1 / (gap * s)) * iv.exp(
        -2 * b / (3 * gap ** ivq('3/2') * iv.sqrt(s))
        - (2 * b * b + 2 * R * R / gap) * rho * rho / (1 - rho * rho))


def _theta1(b, s1, theta1):
    if theta1 is None:
        return ivq('1/2') + ivq(b) / iv.sqrt(ivq(s1))
    return theta1 if isinstance(theta1, iv.mpf) else ivq(theta1)


def check_P(b, T, s1, theta1=None) -> bool:
    r"""
    The standing assumptions (P) of Step 3 that do not involve the path:
    `s_1 \ge 100`, `\beta = b/\sqrt{s_1} \le 1/10`, `\theta_1 \in (1/2, 7/10]`,
    `1 \le T \le \sqrt{s_1/8}`, the range of Step 2 for `(b, T, s_1, \theta_1)`, and
    `\bar\eta < 1 - \alpha_+`. With ``theta1=None``, `\theta_1 = 1/2 + \beta`.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_6 import check_P
        >>> check_P(F(1, 20), F(5, 4), 210), check_P(F(1, 20), F(5, 4), 99)
        (True, False)
    """
    b, T = F(b), F(T)
    if s1 < 100 or not (1 <= T and 8 * T * T <= s1) or 100 * b * b > s1:
        return False
    th = _theta1(b, s1, theta1)
    if not (th.a > 0.5 and th.b <= 0.7):
        return False
    gap = 1 - th
    if (4 * ivq(b) ** 2 * gap * s1).a < 1:
        return False
    if (2 * (ivq(T) + ivq(b) * iv.sqrt(th)) - gap * iv.sqrt(ivq(s1))).b >= 0:
        return False
    return bool((eta_bar_iv(b, T, s1, theta1) - (1 - th)).b < 0)


def eta_bar_iv(b, T, s1, theta1=None) -> iv.mpf:
    r"""
    `\bar\eta = \max\{(2b\sqrt{\alpha_+} + s_1^{-1/2})A(T,s_1),\ e^{-2(\sqrt{\alpha_-}T-b)_+^2}\}`
    of Step 3 (no citations), `\alpha_+ = \theta_1`, `A` for `\theta = \theta_1`.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_6 import eta_bar_iv
        >>> 0.251 < float(eta_bar_iv(F(1, 20), F(5, 4), 210).b) < 0.252
        True
    """
    th = _theta1(b, s1, theta1)
    bi, Ti = ivq(b), ivq(T)
    mod = (2 * bi * iv.sqrt(th) + 1 / iv.sqrt(ivq(s1))) * a_iv(Ti, s1, th)
    ra = iv.sqrt(1 - th) * Ti - bi
    assert ra.a > 0
    return _max(mod, iv.exp(-2 * ra * ra))


def m_star_iv(b, T, s1, theta1=None) -> iv.mpf:
    r"""
    `M^* = (1-\alpha_+)/(1-\bar\eta-\alpha_+)` of Step 3, Theorem filter-M.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_6 import m_star_iv
        >>> 2.02 < float(m_star_iv(F(1, 20), F(5, 4), 210).b) < 2.03
        True
    """
    th = _theta1(b, s1, theta1)
    eb = eta_bar_iv(b, T, s1, theta1)
    den = 1 - eb - th
    assert den.a > 0
    return (1 - th) / den


def c1_iv(b, T, s1, theta1=None) -> iv.mpf:
    r"""
    `c_1` of Step 3, Theorem filter-c1, with `\bar M = M^*`:
    `2b\sqrt{2/\pi}\,\Lambda(b,T,s_1)\,(\hat g(M^*/4) - \hat g(T^2) M^*/(4T^2))`,
    `\hat g(y) = \alpha_+^{-1/2}\exp(-(2\alpha_+y + \frac23 b^2)/\alpha_-)`.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_6 import c1_iv
        >>> 0.0882 < float(c1_iv(F(1, 10), 2, 10**6).a) < 0.0883
        True
    """
    assert check_P(b, T, s1, theta1)
    th = _theta1(b, s1, theta1)
    bi, Ti = ivq(b), ivq(T)
    ap, am = th, 1 - th
    M = m_star_iv(b, T, s1, theta1)

    def g(y):
        return iv.exp(-(2 * ap * y + 2 * bi * bi / 3) / am) / iv.sqrt(ap)
    lam = lambda_iv(bi, Ti, s1, th)
    return 2 * bi * iv.sqrt(2 / iv.pi) * lam * (g(M / 4) - g(Ti * Ti) * M / (4 * Ti * Ti))


# ----------------------------------------------------------------- Step 4

def sigma_constant(theta, K, beta) -> int:
    r"""
    The least integer `\sigma \ge (L+1)\ln 2/(2(\theta-1/2)^2)`,
    `L = (1+\beta+Kh(1/K))/(K-1)`, of Step 4, Theorem cat-constant.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_6 import sigma_constant
        >>> sigma_constant('3/4', 2, 2)
        34
    """
    theta, K, beta = ivq(theta), ivq(K), ivq(beta)
    L = (1 + beta + K * h_iv(1 / K)) / (K - 1)
    x = (L + 1) * LN2 / (2 * (theta - ivq('1/2')) ** 2)
    c = int(math.ceil(float(x.b)))
    assert x.b <= c and (x.a > c - 1)
    return c


def budget_upper(lam, theta='3/4', sigma=34) -> iv.mpf:
    r"""
    An upper bound for `B'` of Step 4, Prop. cat-budget, at `\log_2 m = \lambda`, with
    `\lfloor d^*\rfloor + 1 \le d^* + 1`:
    `(1-h(\theta))(d^*+1) + \sum_{s=3}^{\sigma-1}(1-h(1/s))`, `d^* = \log(m/\sigma)/\log(1/\theta)`.
    Affine in `\lambda`. Valid for `m \ge \sigma`.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_6 import budget_upper
        >>> 26.2 < float(budget_upper(20).b) < 26.3
        True
    """
    th = ivq(theta)
    dstar = (ivq(lam) - iv.log(ivq(sigma)) / LN2) / (-iv.log(th) / LN2)
    tail = sum((1 - h_iv(F(1, s)) for s in range(3, sigma)), ivq(0))
    return (1 - h_iv(th)) * (dstar + 1) + tail


# ----------------------------------------------------------------- Step 5

def p_range(s1, T2, b):
    r"""
    Yield `(N, k_{lo}, k_{hi}, u_{max})` for the finite range of Lemma p of Step 5,
    restricted by symmetry to `k \ge N/2`. ``T2`` is a rational upper bound for `T^2`,
    ``b`` rational; all tests are exact, so the yielded set contains the true range.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_6 import p_range
        >>> next(iter(p_range(100, F(1), F(1, 10))))
        (101, 51, 60, 51)
    """
    T2, b = F(T2), F(b)
    N = s1 + 1
    while True:
        over = F(N, 2) - s1
        if over > 0 and over * over > b * b * N:
            return
        klo = (N + 1) // 2
        khi = klo
        while (F(khi + 1) - F(N, 2)) ** 2 <= T2 * N:
            khi += 1
        umax = N // 2
        while umax + 1 <= s1 and (F(umax + 1) - F(N, 2)) ** 2 <= b * b * N:
            umax += 1
        umax = min(umax, s1)
        yield N, klo, khi, umax
        N += 1


def _varpi_int(N, k, u):
    """(numerator, denominator) of varpi(N, k, u) = P(X in {floor(u/2), ceil(u/2)})."""
    x, y = u // 2, u - u // 2
    num = 0
    for t in {x, y}:
        if t <= k and u - t <= N - k:
            num += math.comb(k, t) * math.comb(N - k, u - t)
    return num, math.comb(N, u)


def p_float_scan(s1, T2, b):
    r"""
    Floating-point minimum of `\varpi` over the range of :func:`p_range`, with its argument.
    Used only to choose the certified bound of :func:`p_certify`.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_6 import p_float_scan
        >>> v, arg = p_float_scan(47, F(9, 16), F(1, 10)); round(v, 5), arg
        (0.05568, (88, 51, 44))
    """
    best, arg = 2.0, None
    lg = math.lgamma
    for N, klo, khi, umax in p_range(s1, T2, b):
        for k in range(klo, khi + 1):
            base = lg(k + 1) + lg(N - k + 1) - lg(N + 1)
            for u in range(2, umax + 1):
                x, y = u // 2, u - u // 2
                tot = 0.0
                for t in {x, y}:
                    if t <= k and u - t <= N - k:
                        tot += math.exp(base - lg(t + 1) - lg(k - t + 1) - lg(u - t + 1)
                                        - lg(N - k - u + t + 1) + lg(u + 1) + lg(N - u + 1))
                if tot < best:
                    best, arg = tot, (N, k, u)
    return best, arg


def p_certify(s1, T2, b, p_lo) -> bool:
    r"""
    Exact check that `\varpi(N,k,u) \ge p_{lo}` on the whole range of :func:`p_range`
    (integer arithmetic, no rounding).

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_6 import p_certify
        >>> p_certify(47, F(9, 16), F(1, 10), F(5568, 10**5)), p_certify(47, F(9, 16), F(1, 10), F(5569, 10**5))
        (True, False)
    """
    p_lo = F(p_lo)
    a, c = p_lo.numerator, p_lo.denominator
    for N, klo, khi, umax in p_range(s1, T2, b):
        for k in range(klo, khi + 1):
            for u in range(2, umax + 1):
                num, den = _varpi_int(N, k, u)
                if num * c < a * den:
                    return False
    return True


# ----------------------------------------------------------------- Step 6

def lipschitz_f() -> iv.mpf:
    r"""
    `2\ln 2 - 1`: the Lipschitz constant of `u \mapsto f(2^u)`, `f(s) = C(s) - \log s`,
    as a 1-periodic function of `u = \log s`.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_6 import lipschitz_f
        >>> 0.386 < float(lipschitz_f().a) < 0.387
        True
    """
    return 2 * LN2 - 1


def q_d_lower(lam, n) -> iv.mpf:
    r"""
    `Q(D) \ge 1 - m(m-1)2^{-n-1} \ge 1 - 2^{2\lambda-n-1}` at `\lambda = \log_2 m`.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_6 import q_d_lower
        >>> float(q_d_lower(20, 50).a) > 1 - 2**-10
        True
    """
    return 1 - iv.mpf(2) ** (ivq(2 * F(lam)) - n - 1)


def error_terms(lam, n, b, T, s1, p_lo, theta4='3/4', sigma=34):
    r"""
    The terms of Theorem C of the step file, at `\log_2 m = \lambda`, `n`:
    returns a dict of intervals ``W`` (large good forks), ``PL`` (first non-good large
    fork), ``PF`` (imperfect small fork), ``B`` (budget), ``cat``, ``QD``, and ``total`` =
    `W + (\gamma + B(P_L+P_F) + \mathrm{cat})/Q(D)`. Requires `n \ge 2\lambda`, `m > s_1`.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_6 import error_terms
        >>> t = error_terms(20, 20000, F(1, 20), F(5, 4), 210, F(989, 100000))
        >>> float(t['total'].b) < 0.1
        True
    """
    assert n >= 2 * lam and 2 ** lam > s1
    bi = ivq(b)
    th = _theta1(b, s1, None)
    beta = bi / iv.sqrt(ivq(s1))
    xi = 2 * beta * LOG2E / (1 - iv.sqrt(th))
    W = 4 * bi * bi / (ivq(s1) * (1 - th))
    c1 = c1_iv(b, T, s1)
    delta = ivq(n) - ivq(lam)
    PL = (1 - c1) ** (delta + iv.log(ivq(s1)) / LN2 - xi) / c1
    ell = math.ceil(math.log2(s1))
    assert 2 ** (ell - 1) < s1 <= 2 ** ell
    pi = ivq(F(p_lo) / 2)
    PF = ell * (1 - pi) ** int(math.floor(float((delta / ell).a)))
    B = budget_upper(lam, theta4, sigma)
    m = iv.mpf(2) ** ivq(lam)
    cat = (m - 1) / (m * m * sigma)
    QD = q_d_lower(lam, n)
    total = W + (GAMMA + B * (PL + PF) + cat) / QD
    return {"W": W, "c1": c1, "PL": PL, "PF": PF, "B": B, "cat": cat, "QD": QD,
            "xi": xi, "total": total}


def delta_needed(lam, b, T, s1, p_lo, eps_L, eps_F, theta4='3/4', sigma=34):
    r"""
    The closed-form threshold `\Delta_0(\lambda) = \max(\Delta_L, \Delta_F)` of Corollary C,
    as an interval:
    `\Delta_F = \ell(1 + \ln(\ell B/\varepsilon_F)/\pi)`,
    `\Delta_L = \ln(B/(c_1\varepsilon_L))/c_1 - \log s_1 + \xi`; for `n - \log m \ge \Delta_0`,
    `B P_F \le \varepsilon_F` and `B P_L \le \varepsilon_L`.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_6 import delta_needed
        >>> d = delta_needed(20, F(1, 20), F(5, 4), 210, F(989, 100000), F(1, 250), F(1, 250))
        >>> 17000 < float(d.b) < 19000
        True
    """
    bi = ivq(b)
    th = _theta1(b, s1, None)
    beta = bi / iv.sqrt(ivq(s1))
    xi = 2 * beta * LOG2E / (1 - iv.sqrt(th))
    c1 = c1_iv(b, T, s1)
    B = budget_upper(lam, theta4, sigma)
    ell = math.ceil(math.log2(s1))
    pi = ivq(F(p_lo) / 2)
    dF = ell * (1 + iv.log(ell * B / ivq(eps_F)) / pi)
    dL = iv.log(B / (c1 * ivq(eps_L))) / c1 - iv.log(ivq(s1)) / LN2 + xi
    return _max(dF, dL)
