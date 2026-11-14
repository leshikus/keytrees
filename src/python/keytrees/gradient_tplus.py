r"""
Interval checks of the table function `t^+_{n,m}` of ``articles/gradient-bound``.

`t^+_{n,m}` is computed by the C++ engine ``gradient_tplus`` in double
precision with upward rounding. This module recomputes its ingredients
independently, in the interval arithmetic of ``mpmath.iv`` with exact
integer binomials, so that a sample of cells can be checked: the engine's
value must be at least the upper endpoint of the interval computed here.

The ingredients (see the engine's header and the article):

- :func:`fixed_bound` -- Theorem fixed-m;
- :func:`union_bound` -- Proposition union with requirement `\kappa_j`;
- :func:`explicit_bound` -- Corollary explicit;
- :func:`trivial_bound` -- `\min(n, C_{\max}(m))`.

Each returns an ``iv.mpf`` interval that contains the stated bound.

EXAMPLES::

    >>> from keytrees.gradient_tplus import fixed_bound, trivial_bound
    >>> b = fixed_bound(10, 4)
    >>> 2.00228 < float(b.a) <= float(b.b) < 2.00229
    True
    >>> trivial_bound(5, 6)
    Fraction(10, 3)
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
from fractions import Fraction
from math import comb

from mpmath import iv

from keytrees.gradient_bound import complete_cost

__all__ = [
    "PREC",
    "inv_distinct",
    "trivial_bound",
    "fixed_bound",
    "kappa_tau",
    "union_bound",
    "explicit_bound",
]

PREC = 160


def _iv(x):
    if isinstance(x, Fraction):
        return iv.mpf(x.numerator) / x.denominator
    return iv.mpf(x)


def _log2(m):
    return iv.log(iv.mpf(m)) / iv.log(2)


def _cmax(m):
    return Fraction(m + 1, 2) - Fraction(1, m)


def inv_distinct(n, m):
    r"""
    Return an interval containing `1/Q(D)`, `Q(D) = \prod_{i<m}(1 - i2^{-n})`.

    The product is taken directly for `m \le 4096`; otherwise
    `\sum x_i \le -\ln Q(D) \le \sum x_i/(1-x_i) \le m(m-1)2^{-n-1}/(1-(m-1)2^{-n})`.

    EXAMPLES::

        >>> float(inv_distinct(3, 2).b) == 8 / 7
        True
    """
    iv.prec = PREC
    if m <= 4096:
        p = iv.mpf(1)
        for i in range(1, m):
            p *= 1 - iv.mpf(i) / iv.mpf(2) ** n
        return 1 / p
    two = iv.mpf(2) ** n
    lo = iv.mpf(m) * (m - 1) / (2 * two)
    hi = lo / (1 - iv.mpf(m - 1) / two)
    return iv.exp(iv.mpf([lo.a, hi.b]))


def trivial_bound(n, m):
    r"""
    Return `\min(n, C_{\max}(m))` exactly.

    EXAMPLES::

        >>> trivial_bound(100, 4), trivial_bound(3, 64)
        (Fraction(9, 4), Fraction(3, 1))
    """
    return min(Fraction(n), _cmax(m))


def _fail(s, k):
    """f_k(s) as a Fraction."""
    return Fraction(2 * sum(comb(s, j) for j in range(k)), 2**s)


def fixed_bound(n, m):
    r"""
    Return the bound of Theorem fixed-m as an interval (all `s \le m`, so `m` should be small).

    EXAMPLES::

        >>> b = fixed_bound(30, 3)
        >>> abs(float((b - iv.mpf(5) / 3).b)) < 1e-40
        True
    """
    iv.prec = PREC
    c = complete_cost(m)
    total = iv.mpf(0)
    for s in range(4, m + 1):
        total += comb(m, s) * _iv(_fail(s, s // 2)) ** n
    return _iv(c) + _iv(_cmax(m) - c) * total * inv_distinct(n, m)


def kappa_tau(n, m, j, smax):
    r"""
    Return `\kappa_j(s)` for `s \le s_{\max}` (index `s`): the largest `k \le \lfloor s/2\rfloor`
    with `\binom ms f_k(s)^n \le 2^{-j}/m`, decided on interval endpoints as in the engine,
    and `1` if there is none.

    EXAMPLES::

        >>> kappa_tau(1000, 8, 10, 8)[2:]
        [1, 1, 2, 2, 3, 3, 4]
    """
    iv.prec = PREC
    thr = -(j * iv.log(2)) - iv.log(iv.mpf(m))
    kap = [0, 0]
    for s in range(2, smax + 1):
        lb = iv.log(iv.mpf(comb(m, s)))
        prefix = [0]
        for i in range(s // 2):
            prefix.append(prefix[-1] + comb(s, i))

        def ok(k):  # log f_k(s) increases with k, so binary search
            lf = iv.log(iv.mpf(2 * prefix[k])) - s * iv.log(2)
            return (lb + n * lf).b <= thr.a

        lo, hi = 1, s // 2
        if not ok(1):
            kap.append(1)
            continue
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if ok(mid) else (lo, mid - 1)
        kap.append(lo)
    return kap


def _rho_upper(kappa, smax):
    one = iv.mpf(1)
    ln2 = iv.log(2)
    up = [iv.mpf(0)] * (smax + 1)
    for s in range(2, smax + 1):
        best = iv.mpf(0)
        for k in range(kappa[s], s // 2 + 1):
            x = iv.mpf(k) / s
            y = one - x
            v = one + (x * iv.log(x) + y * iv.log(y)) / ln2 + x * up[k] + y * up[s - k]
            best = iv.mpf(max(best.b, v.b))
        up[s] = best
    return up


def union_bound(n, m, j, s1, x):
    r"""
    Return the bound of Proposition union as an interval, with `\kappa = \kappa_j` below `s_1`.

    If `s_1 \ge m`, no fork is large: `P_{\mathrm{cyl}} = 0` and the term `4x/(s_1(1-\theta))`
    is omitted. The maximum of `\rho_\kappa` is then over `s \le m`.

    EXAMPLES::

        >>> b = union_bound(4096, 64, 10, 64, 0.4)
        >>> 0.0860 < float(b.b) - 6 < 0.0862
        True
    """
    iv.prec = PREC
    S = min(s1, m)
    kap = kappa_tau(n, m, j, S)
    rho = _rho_upper(kap, S)
    mrho = max(r.b for r in rho[1:])
    pcol = iv.mpf(0)
    for s in range(4, S + 1):
        if kap[s] >= 2:
            pcol += comb(m, s) * _iv(_fail(s, kap[s])) ** n
    tail = iv.mpf(0)
    pcyl = iv.mpf(0)
    if m > s1:
        xi = iv.mpf(x)
        theta = iv.mpf(0.5) + iv.sqrt(xi / s1)
        assert theta.b < 1 and x > math.log(2) / 2
        tail = 4 * xi / (s1 * (1 - theta))
        ratio = iv.log(iv.mpf(m) / s1) / -iv.log(theta)
        dstar = int(math.floor(ratio.b))  # an upper bound of d*; extra terms only add
        lg = iv.log(2) - 2 * xi
        for d in range(0, min(dstar, n - 1) + 1):
            pcyl += iv.exp(d * iv.log(iv.mpf(2 * n)) + (n - d) * lg)
    rmax = _iv(_cmax(m)) - _log2(m)
    return _log2(m) + iv.mpf(mrho) + tail + rmax * inv_distinct(n, m) * (pcol + pcyl)


def explicit_bound(n, m, cK):
    r"""
    Return `\log m + c_K + (0.026m^{-2} + [m > 500]2^{-10})/Q(D)` (Corollary explicit), for a given
    upper bound ``cK`` of `c_K`; the caller checks the hypotheses.

    EXAMPLES::

        >>> 10.12 < float(explicit_bound(4096, 1024, 0.12645403).b) < 10.13
        True
    """
    iv.prec = PREC
    err = iv.mpf(26) / 1000 / (iv.mpf(m) ** 2)
    if m > 500:
        err += iv.mpf(2) ** -10
    return _log2(m) + _iv(Fraction(cK)) + err * inv_distinct(n, m)
