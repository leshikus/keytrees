r"""
Step 3 of the analytic estimate for the gradient key: filtered laws of an unchosen row.

See ``articles/gradient-bound/analytic/03-filtering.tex``. Along the path of a column the
forks have sizes `s_0 = m > s_1 > \dots`; an unchosen row is uniform, conditioned at each
fork `S_l` on `|e_l| \ge \delta_l` (rows after the chosen one) or `|e_l| > \delta_l` (rows
before it), where `e_l = k_r(S_l) - s_l/2` and `\delta_l = s_l/2 - q_l`. Its counts form a
Markov chain with hypergeometric steps, so its conditional law is computed by forward
filtering: restrict to the allowed counts, normalise, push through the kernel.

The functions fall into three groups.

- Exact filtering: :func:`filter_exact` (rational, small sizes) and :func:`filter_float`
  (floating point, truncated to `|e| \le w\sqrt s`, the truncated mass is returned).
- Worst cases computed exactly: :func:`hole_sup` (the largest probability that one kernel
  step lands in a hole, over all parent counts).
- The bounds proved in the step file: :func:`eta_bar`, :func:`m_star`, :func:`c1_bound`,
  :func:`c1_limit`, :func:`lyapunov_l_star`; and the Gaussian limit of the worst constant
  hole, :func:`gaussian_filter_limit`, which is numerics and not a proof.

Floating-point values are not proofs; they stress-test the statements.

EXAMPLES::

    >>> from fractions import Fraction
    >>> from keytrees.gradient_analytic_3 import filter_exact
    >>> levels = filter_exact(8, [8, 4, 2], [Fraction(0), Fraction(0)], [False, False])
    >>> [lv["eta"] for lv in levels[:-1]]
    [0, 0]
    >>> levels[0]["M"]
    Fraction(1, 1)
"""
# SPDX-License-Identifier: GPL-2.0-or-later
from __future__ import annotations

import math
from fractions import Fraction
from math import comb

import numpy as np
from scipy.special import erf, gammaln

__all__ = [
    "BE_CONST",
    "hyp_pmf_exact",
    "filter_exact",
    "filter_float",
    "hole_sup",
    "adversarial_sequence",
    "a_max",
    "big_lambda",
    "eta_bar",
    "m_star",
    "c1_bound",
    "c1_limit",
    "lyapunov_l_star",
    "gaussian_filter_limit",
]

#: Berry--Esseen constant for sums of independent, non-identical summands (Shevtsova 2010).
BE_CONST = 0.56


# ----------------------------------------------------------------- exact filtering

def hyp_pmf_exact(s, k, a):
    r"""
    The law of `X \sim \mathrm{Hyp}(s, k, a)` as a dict of Fractions.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_3 import hyp_pmf_exact
        >>> hyp_pmf_exact(4, 2, 2)
        {0: Fraction(1, 6), 1: Fraction(2, 3), 2: Fraction(1, 6)}
    """
    tot = comb(s, a)
    return {x: Fraction(comb(k, x) * comb(s - k, a - x), tot)
            for x in range(max(0, a + k - s), min(a, k) + 1)}


def _in_hole(e, delta, strict):
    # strict (row before the chosen row): allowed |e| > delta, hole |e| <= delta;
    # non-strict (row after it): allowed |e| >= delta, hole |e| < delta.
    return abs(e) <= delta if strict else abs(e) < delta


def filter_exact(m, sizes, deltas, strict):
    r"""
    Forward filtering in rational arithmetic.

    ``sizes`` is `s_0 = m, s_1, \dots, s_d`; ``deltas[l]`` and ``strict[l]`` describe the
    hole at level `l < d`. Returns one dict per level `l < d` with the pre-filter mean `N` of
    `v = 4e^2/s`, the hole mass ``eta``, the filtered mean ``M``, and at the last level the
    pre-filter law ``law`` (dict count -> probability).

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_3 import filter_exact
        >>> lv = filter_exact(6, [6, 3], [F(1)], [True])
        >>> lv[0]["eta"], lv[0]["M"]
        (Fraction(25, 32), Fraction(22, 7))
    """
    assert sizes[0] == m
    law = {k: Fraction(comb(m, k), 2 ** m) for k in range(m + 1)}
    out = []
    for l in range(len(sizes) - 1):
        s, a = sizes[l], sizes[l + 1]
        N = sum(p * Fraction(4) * (k - Fraction(s, 2)) ** 2 / s for k, p in law.items())
        hole = {k for k in law if _in_hole(k - Fraction(s, 2), deltas[l], strict[l])}
        eta = sum(law[k] for k in hole)
        kept = {k: p / (1 - eta) for k, p in law.items() if k not in hole}
        M = sum(p * Fraction(4) * (k - Fraction(s, 2)) ** 2 / s for k, p in kept.items())
        new = {}
        for k, p in kept.items():
            for x, q in hyp_pmf_exact(s, k, a).items():
                new[x] = new.get(x, 0) + p * q
        out.append({"N": N, "eta": eta, "M": M})
        law = new
    out.append({"law": law})
    return out


def _hyp_log_matrix(s, a, ks, xs):
    ks = ks[:, None]
    xs = xs[None, :]
    with np.errstate(invalid="ignore"):
        lp = (gammaln(ks + 1) - gammaln(xs + 1) - gammaln(ks - xs + 1)
              + gammaln(s - ks + 1) - gammaln(a - xs + 1) - gammaln(s - ks - a + xs + 1)
              - gammaln(s + 1) + gammaln(a + 1) + gammaln(s - a + 1))
    ok = (xs <= ks) & (a - xs <= s - ks) & (xs >= 0) & (xs <= a)
    return np.where(ok, lp, -np.inf)


def filter_float(m, sizes, deltas, strict, b_good=None, width=14.0):
    r"""
    Forward filtering in floating point, with counts truncated to `|e| \le` ``width``
    `\sqrt s`. Returns per level the pre-filter mean ``N``, hole mass ``eta``, filtered mean
    ``M``, the probability ``good`` that `|e| \le b\sqrt s` before filtering (if ``b_good``),
    and ``lost``, the total truncated mass so far (an upper bound on the error).

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_3 import filter_exact, filter_float
        >>> sz, de, st = [40, 21, 11], [F(1, 2) + 1, F(3, 2)], [True, False]
        >>> ex = filter_exact(40, sz, de, st)
        >>> fl = filter_float(40, sz, [float(d) for d in de], st, width=40)
        >>> abs(float(ex[1]["M"]) - fl[1]["M"]) < 1e-12
        True
    """
    def window(s):
        lo = max(0, math.floor(s / 2 - width * math.sqrt(s)))
        hi = min(s, math.ceil(s / 2 + width * math.sqrt(s)))
        return np.arange(lo, hi + 1, dtype=float)

    ks = window(m)
    logp = gammaln(m + 1) - gammaln(ks + 1) - gammaln(m - ks + 1) - m * math.log(2)
    p = np.exp(logp)
    lost = max(0.0, 1.0 - p.sum())
    out = []
    for l in range(len(sizes)):
        s = sizes[l]
        e = ks - s / 2
        v = 4 * e * e / s
        rec = {"N": float(p @ v) / p.sum()}
        if b_good is not None:
            rec["good"] = float(p[np.abs(e) <= b_good * math.sqrt(s) + 1e-9].sum() / p.sum())
        if l == len(sizes) - 1:
            rec["lost"] = lost
            out.append(rec)
            break
        d = deltas[l]
        hole = (np.abs(e) <= d + 1e-9) if strict[l] else (np.abs(e) < d - 1e-9)
        eta = float(p[hole].sum() / p.sum())
        q = np.where(hole, 0.0, p)
        q /= q.sum()
        rec.update(eta=eta, M=float(q @ v), lost=lost)
        out.append(rec)
        a = sizes[l + 1]
        xs = window(a)
        K = np.exp(_hyp_log_matrix(s, a, ks, xs))
        p = q @ K
        lost += max(0.0, 1.0 - p.sum())
        ks = xs
    return out


def hole_sup(s, a, delta, strict):
    r"""
    `\max_k \Pr(X \in \text{hole})` for `X \sim \mathrm{Hyp}(s, k, a)`, over all `k`, computed
    in floating point; the hole is `|X - a/2| < \delta` or `\le \delta`.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_3 import hole_sup
        >>> round(hole_sup(4, 2, 0, True), 4)
        0.6667
        >>> hole_sup(100, 50, 0, False)
        0.0
    """
    ks = np.arange(s + 1, dtype=float)
    xs = np.arange(a + 1, dtype=float)
    K = np.exp(_hyp_log_matrix(s, a, ks, xs))
    ex = xs - a / 2
    hole = (np.abs(ex) <= delta + 1e-9) if strict else (np.abs(ex) < delta - 1e-9)
    return float(K[:, hole].sum(axis=1).max()) if hole.any() else 0.0


def adversarial_sequence(m, b, s_end, rng=None, side="random"):
    r"""
    A path of good forks with the largest allowed imbalance `\delta = s/2 - \lceil s/2 - b\sqrt s\rceil`
    at every level, from `m` down to the first size `\le` ``s_end``. Returns ``sizes, deltas``.
    ``side`` chooses the larger part ("big"), the smaller ("small") or at random.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_3 import adversarial_sequence
        >>> adversarial_sequence(1000, 0.2, 100, side="big")[0]
        [1000, 506, 257, 131, 67]
    """
    sizes, deltas = [m], []
    s = m
    while s > s_end:
        q = math.ceil(s / 2 - b * math.sqrt(s))
        deltas.append(Fraction(s, 2) - q)
        if side == "big":
            s = s - q
        elif side == "small":
            s = q
        else:
            s = q if rng.random() < 0.5 else s - q
        sizes.append(s)
    return sizes, deltas


# ----------------------------------------------------------------- proved bounds

def _sigma_range(b, s1, T):
    beta = b / math.sqrt(s1)
    sig_lo2 = (0.25 - T * T / s1) * (0.5 - beta)
    sig_hi2 = 0.25 * (0.5 + beta) * (1 + 2 / s1)
    return beta, sig_lo2, sig_hi2


def a_max(T, s, theta):
    r"""
    `A(T,s)` of Step 2 (Theorem hg-max): `\max_x \Pr(X=x) \le A/\sqrt s` for `|e| \le T\sqrt s`.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_3 import a_max
        >>> round(a_max(0, 10**12, 0.5), 4)
        1.5958
    """
    assert s > 4 * T * T
    return math.sqrt(2 * (1 + 1 / s) / (math.pi * theta * (1 - theta))) * math.exp(
        2 * T * T / s + 8 * T ** 4 / (s - 4 * T * T))


def big_lambda(b, T, s, theta):
    r"""
    `\Lambda(b,T,s)` of Step 2 (Theorem hg-window), in floating point; ``0`` outside its range.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_3 import big_lambda
        >>> 0.98 < big_lambda(0.09, 3, 10**6, 0.5 + 0.09e-3) < 0.99
        True
    """
    gap = 1 - theta
    R = T + b * math.sqrt(theta)
    rho2 = 4 * R * R / (gap * gap * s)
    if rho2 >= 1 or 4 * b * b * gap * s < 1:
        return 0.0
    u = b * math.sqrt(gap * s)
    nu = max(0.5, 1 - 1 / (2 * u))
    size = 1 / (1 + 1 / (gap * s))
    t3 = (2 * b / 3) / (gap * math.sqrt(gap * s))
    err = (2 * b * b + 2 * R * R / gap) * rho2 / (1 - rho2)
    return nu * size * math.exp(-(t3 + err))


def eta_bar(b, s1, T, cited=False):
    r"""
    The uniform hole bound `\bar\eta` of the step file. Without citations (``cited=False``) the
    moderate part is `(2b\sqrt{\alpha_+} + s_1^{-1/2}) A(T, s_1)` (Step 2); with ``cited=True`` it is
    `\mathrm{erf}(b/(\sigma_-\sqrt2)) + 2C_0/\sigma_{\min}` (Berry--Esseen). The immoderate part is
    `e^{-2(\sqrt{\alpha_-}T-b)^2}`.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_3 import eta_bar
        >>> round(eta_bar(0.1, 10**12, 10, cited=True), 4)
        0.2227
        >>> round(eta_bar(0.1, 10**12, 10), 4)
        0.2257
    """
    beta, lo2, _ = _sigma_range(b, s1, T)
    if cited:
        smin = math.sqrt(lo2 * (0.5 - beta) * s1)
        mod = float(erf(b / math.sqrt(2 * lo2))) + 2 * BE_CONST / smin
    else:
        mod = (2 * b * math.sqrt(0.5 + beta) + 1 / math.sqrt(s1)) * a_max(T, s1, 0.5 + beta)
    ra = math.sqrt(0.5 - beta) * T - b
    imm = math.exp(-2 * ra * ra) if ra > 0 else 1.0
    return max(mod, imm)


def m_star(b, s1, T, cited=False):
    r"""
    `M^* = (1-\alpha_+)/(1-\bar\eta-\alpha_+)`, or ``inf`` if the recursion does not close.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_3 import m_star
        >>> round(m_star(0.1, 10**12, 10, cited=True), 3)
        1.803
    """
    beta = b / math.sqrt(s1)
    ap = 0.5 + beta
    eb = eta_bar(b, s1, T, cited)
    return (1 - ap) / (1 - eb - ap) if eb < 1 - ap else math.inf


def c1_bound(b, s1, T, M):
    r"""
    The lower bound of Theorem filter-c1 on the probability that an unchosen row is good on
    the next large fork, given a bound `M` on the filtered mean of `v` at the parent:
    `2b\sqrt{2/\pi}\,\Lambda(b,T,s_1)\,(\hat g(M/4) - \hat g(T^2) M/(4T^2))`, with
    `\hat g(y) = \alpha_+^{-1/2}\exp(-(2\alpha_+ y + 2b^2/3)/\alpha_-)`.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_3 import c1_bound, m_star
        >>> round(c1_bound(0.1, 10**12, 5, m_star(0.1, 10**12, 5)), 4)
        0.0895
    """
    if not math.isfinite(M):
        return 0.0
    beta = b / math.sqrt(s1)
    ap, am = 0.5 + beta, 0.5 - beta

    def g(y):
        return math.exp(-(2 * ap * y + 2 * b * b / 3) / am) / math.sqrt(ap)
    lam = big_lambda(b, T, s1, ap)
    return 2 * b * math.sqrt(2 / math.pi) * lam * (g(M / 4) - g(T * T) * M / (4 * T * T))


def c1_limit(b, cited=False):
    r"""
    The limit `s_1 \to \infty` of the proved constant:
    `c_1^\infty(b) = \frac{4b}{\sqrt\pi}\exp(-M^*_\infty/2 - 4b^2/3)`, `M^*_\infty = 1/(1-2\eta_\infty)`,
    with `\eta_\infty = 4b/\sqrt\pi` (no citations) or `\mathrm{erf}(2b)` (``cited``); ``0`` if
    `\eta_\infty \ge 1/2`.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_3 import c1_limit
        >>> round(c1_limit(0.1), 4), round(c1_limit(0.1, cited=True), 4)
        (0.0895, 0.0904)
        >>> c1_limit(0.3)
        0.0
    """
    e = float(erf(2 * b)) if cited else 4 * b / math.sqrt(math.pi)
    if e >= 0.5:
        return 0.0
    M = 1 / (1 - 2 * e)
    return 4 * b / math.sqrt(math.pi) * math.exp(-M / 2 - 4 * b * b / 3)


def lyapunov_l_star(eta, t, alpha=0.5):
    r"""
    The exponential-moment fixed point
    `L^* = (\frac12\ln\frac1{1-2t} + \ln\frac1{1-\eta})/(1-\kappa)`, `\kappa = \alpha/(1-2t)`;
    ``inf`` if `\kappa \ge 1`.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_3 import lyapunov_l_star
        >>> round(lyapunov_l_star(0.5, 0.1), 4)
        2.1459
    """
    kap = alpha / (1 - 2 * t)
    if kap >= 1 or eta >= 1:
        return math.inf
    return (0.5 * math.log(1 / (1 - 2 * t)) + math.log(1 / (1 - eta))) / (1 - kap)


def gaussian_filter_limit(b, levels=80, grid=4801, span=10.0):
    r"""
    Numerics, not a proof: the Gaussian limit `x' = x/\sqrt2 + N(0, 1/8)` of the chain in the
    scale `x = e/\sqrt s`, filtered at every level by the largest hole `|x| < b`, started from
    `N(0, 1/4)`. Returns ``(eta, M, good)`` at the last level: hole mass, filtered mean of
    `4x^2`, and the probability of `|x| \le b` before filtering.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_3 import gaussian_filter_limit
        >>> eta, M, good = gaussian_filter_limit(0.0)
        >>> round(M, 3), round(good, 3)
        (1.0, 0.0)
    """
    xs = np.linspace(-span, span, grid)
    dx = xs[1] - xs[0]
    sd = math.sqrt(1 / 8)
    K = np.exp(-((xs[None, :] - xs[:, None] / math.sqrt(2)) ** 2) / (2 * sd * sd))
    K /= K.sum(axis=1, keepdims=True)
    p = np.exp(-2 * xs * xs)
    p /= p.sum()
    hole = np.abs(xs) < b
    eta = M = good = 0.0
    for _ in range(levels):
        good = float(p[np.abs(xs) <= b].sum()) if b > 0 else 0.0
        eta = float(p[hole].sum())
        q = np.where(hole, 0.0, p)
        q /= q.sum()
        M = float(q @ (4 * xs * xs))
        p = q @ K
    return eta, M, good


# ----------------------------------------------------------------- variant for fixed sets (Step 7)

def eta_bar_theta(bh, s1, T, theta):
    r"""
    `\bar\eta` of Proposition filter-variant: hole radius `b_h\sqrt{s}`, ratios in `[1-\theta,\theta]`,
    sizes `> s_1`, Step 2 constants only.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_3 import eta_bar_theta
        >>> round(eta_bar_theta(0.05, 10**12, 10, 0.6), 4)
        0.1262
    """
    mod = (2 * bh * math.sqrt(theta) + 1 / math.sqrt(s1)) * a_max(T, s1, theta)
    ra = math.sqrt(1 - theta) * T - bh
    imm = math.exp(-2 * ra * ra) if ra > 0 else 1.0
    return max(mod, imm)


def m_star_theta(bh, s1, T, theta):
    r"""
    `M^* = (1-\theta)/(1-\bar\eta-\theta)` of Proposition filter-variant, or ``inf``.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_3 import m_star_theta
        >>> round(m_star_theta(0.05, 10**12, 10, 0.6), 4)
        1.4607
    """
    eb = eta_bar_theta(bh, s1, T, theta)
    return (1 - theta) / (1 - eb - theta) if eb < 1 - theta else math.inf


def c1_bound_theta(bw, s1, T, M, theta):
    r"""
    `c_1` of Proposition filter-variant: window `b_w\sqrt{s}`, given a bound `M` on the filtered mean.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_3 import c1_bound_theta, m_star_theta
        >>> round(c1_bound_theta(0.05, 10**12, 5, m_star_theta(0.05, 10**12, 5, 0.6), 0.6), 4)
        0.0343
    """
    if not math.isfinite(M):
        return 0.0
    ap, am = theta, 1 - theta

    def g(y):
        return math.exp(-(2 * ap * y + 2 * bw * bw / 3) / am) / math.sqrt(ap)
    lam = big_lambda(bw, T, s1, theta)
    return 2 * bw * math.sqrt(2 / math.pi) * lam * (g(M / 4) - g(T * T) * M / (4 * T * T))
