r"""
Vajda's tight lower bound and the refined Pinsker bounds

Vajda's bound `L(V)` is the infimum of the divergence `D(P\|Q)` in nats over pairs with
total variation `V`. Fedotov, Harremoes and Topsoe gave the parametrisation

.. MATH::

    V(t) = t(1-(\coth t - 1/t)^2), \qquad
    L(t) = \ln(t/\sinh t) + t\coth t - t^2/\sinh^2 t,

with `t = dL/dV`. This module computes the Taylor coefficients of `L(V)` exactly.

EXAMPLES::

    >>> c = taylor_coefficients(8)
    >>> c[2], c[4], c[6], c[8]
    (Fraction(1, 2), Fraction(1, 36), Fraction(1, 270), Fraction(221, 340200))
"""
from __future__ import annotations

from fractions import Fraction
from math import comb


def bernoulli_even(count: int) -> list[Fraction]:
    r"""
    Return `B_0, B_2, \dots, B_{2(count-1)}`.

    EXAMPLES::

        >>> bernoulli_even(3)
        [Fraction(1, 1), Fraction(1, 6), Fraction(-1, 30)]
    """
    top = 2 * count
    b = [Fraction(0)] * (top + 1)
    b[0] = Fraction(1)
    for m in range(1, top + 1):
        b[m] = -sum(comb(m + 1, k) * b[k] for k in range(m)) / (m + 1)
    return [b[2 * k] for k in range(count)]


def _coth_series(count: int) -> list[Fraction]:
    r"""
    Return `a_j` with `\coth t - 1/t = t\sum_j a_j t^{2j}`.

    TESTS::

        >>> _coth_series(2)
        [Fraction(1, 3), Fraction(-1, 45)]
    """
    bernoulli = bernoulli_even(count + 1)
    out = []
    factorial = 1
    for k in range(1, count + 1):
        factorial *= (2 * k - 1) * (2 * k)
        out.append(Fraction(4**k, factorial) * bernoulli[k])
    return out


def taylor_coefficients(degree: int) -> dict[int, Fraction]:
    r"""
    Return the Taylor coefficients `t_\nu` of `L(V)` for even `\nu\le` ``degree``.

    ALGORITHM:

    With `s=t^2` and `\coth t-1/t=t\,a(s)`, we have `t/V(t)=f(s)^{-1}`, `f=1-s\,a(s)^2`.
    Lagrange inversion gives `[V^n]\,t=\frac1n[s^{(n-1)/2}]f^{-n}`, and `L'(V)=t` then gives
    `t_{n+1}=[V^n]\,t/(n+1)`. Powers of `f` come from Miller's recurrence.

    INPUT:

    - ``degree`` -- largest degree wanted

    EXAMPLES::

        >>> taylor_coefficients(2)
        {2: Fraction(1, 2)}

    TESTS::

        >>> taylor_coefficients(10)[10] == Fraction(299, 2 * 3**8 * 5**2 * 7)
        True
    """
    terms = degree // 2 + 1
    a = _coth_series(terms)
    # a(s)^2
    square = [sum(a[i] * a[j - i] for i in range(j + 1)) for j in range(terms)]
    f = [Fraction(1)] + [-square[j] for j in range(terms - 1)]
    out: dict[int, Fraction] = {}
    for n in range(1, degree, 2):
        order = (n - 1) // 2
        power = -n
        g = [Fraction(1)]
        for k in range(1, order + 1):
            total = sum((power * i - (k - i)) * f[i] * g[k - i] for i in range(1, k + 1))
            g.append(total / k)
        out[n + 1] = g[order] / n / (n + 1)
    return out


def parametrisation(t, mp):
    r"""
    Return `(V(t),L(t))`, using the arithmetic of the module ``mp`` (``mpmath`` or ``mpmath.iv``).

    EXAMPLES::

        >>> import mpmath
        >>> mpmath.mp.dps = 30
        >>> v, l = parametrisation(mpmath.mpf(1), mpmath)
        >>> round(float(v), 6), round(float(l), 6)
        (0.902009, 0.427534)
    """
    coth = mp.coth(t)
    sinh = mp.sinh(t)
    u = coth - 1 / t
    v = t * (1 - u * u)
    return v, mp.log(t / sinh) + t * coth - t * t / (sinh * sinh)


def remainder(t, coefficients: dict[int, Fraction], degree: int, mp):
    r"""
    Return `R_\nu(V(t))=L(V(t))-T_\nu(V(t))` for ``degree`` `=\nu`.

    EXAMPLES::

        >>> import mpmath
        >>> mpmath.mp.dps = 60
        >>> c = taylor_coefficients(8)
        >>> remainder(mpmath.mpf(1), c, 8, mpmath) > 0
        True
    """
    v, value = parametrisation(t, mp)
    power = v * v
    square = power
    for nu in range(2, degree + 1, 2):
        value = value - mp.mpf(coefficients[nu].numerator) / mp.mpf(coefficients[nu].denominator) * power
        power = power * square
    return value
