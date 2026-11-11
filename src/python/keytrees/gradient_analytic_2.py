r"""
Hypergeometric toolkit of ``articles/gradient-bound/analytic/02-hypergeometric.tex``.

`X \sim \mathrm{Hyp}(s,k,a)` counts the ones in a uniform `a`-subset of `s`
positions of which `k` hold a one. Probabilities here are exact fractions.
Transcendental quantities in the bounds are enclosed by :class:`Iv`, an interval
with rational end points, so a comparison made with it is exact arithmetic.

AUTHORS:

- Written for the article on the gradient key (2026).

EXAMPLES::

    >>> from fractions import Fraction as F
    >>> from keytrees.gradient_analytic_2 import pmf, second_moment
    >>> pmf(4, 2, 2)
    {0: Fraction(1, 6), 1: Fraction(2, 3), 2: Fraction(1, 6)}
    >>> second_moment(4, 2, 2)
    Fraction(2, 3)
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from math import comb, isqrt

_BITS = 160
_ONE = 1 << _BITS


def _shift(x: F) -> int:
    """Scale so that a rounded value keeps about 160 significant bits."""
    if x == 0:
        return _BITS
    return _BITS - (abs(x.numerator).bit_length() - x.denominator.bit_length())


def _down(x: F) -> F:
    k = _shift(x)
    if k >= 0:
        return F((x.numerator << k) // x.denominator, 1 << k)
    return F((x.numerator // (x.denominator << -k)) << -k)


def _up(x: F) -> F:
    return -_down(-x)


@dataclass(frozen=True)
class Iv:
    r"""
    A closed interval ``[lo, hi]`` with rational end points, all values positive
    where products and quotients are taken.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_2 import Iv, PI
        >>> PI.lo < 3.1415926536 and PI.hi > 3.1415926535
        True
    """

    lo: F
    hi: F

    @staticmethod
    def of(x) -> "Iv":
        x = F(x)
        return Iv(x, x)

    def __add__(self, o: "Iv") -> "Iv":
        return Iv(_down(self.lo + o.lo), _up(self.hi + o.hi))

    def __neg__(self) -> "Iv":
        return Iv(-self.hi, -self.lo)

    def __sub__(self, o: "Iv") -> "Iv":
        return self + (-o)

    def __mul__(self, o: "Iv") -> "Iv":
        v = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return Iv(_down(min(v)), _up(max(v)))

    def __truediv__(self, o: "Iv") -> "Iv":
        assert o.lo > 0, "division by an interval that is not positive"
        return self * Iv(_down(1 / o.hi), _up(1 / o.lo))


def iv_sqrt(x: Iv) -> Iv:
    r"""
    Enclose the square root of a nonnegative interval.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_2 import Iv, iv_sqrt
        >>> r = iv_sqrt(Iv.of(2))
        >>> r.lo**2 <= 2 <= r.hi**2
        True
    """
    assert x.lo >= 0

    def root_floor(q: F) -> F:
        return F(isqrt((q.numerator << (2 * _BITS)) // q.denominator), _ONE)

    return Iv(root_floor(x.lo), root_floor(x.hi) + F(1, _ONE))


def iv_exp(x: Iv) -> Iv:
    r"""
    Enclose `\exp` of an interval, by halving and a Taylor polynomial with remainder.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_2 import Iv, iv_exp
        >>> r = iv_exp(Iv.of(1))
        >>> r.lo < 2.718281829 and r.hi > 2.718281828
        True
    """

    def point(q: F) -> Iv:
        halvings = 0
        while abs(q) > F(1, 4):
            q /= 2
            halvings += 1
        # Fixed point with floor rounding. Each step errs by at most 2 units, so with
        # |q| <= 1/4 the whole sum errs by far less than 256 units; the truncated
        # tail is below one unit once a term has reached one unit.
        fixed = (q.numerator << _BITS) // q.denominator
        term = total = 1 << _BITS
        for i in range(1, 60):
            term = ((term * fixed) >> _BITS) // i
            total += term
            if abs(term) <= 1:
                break
        r = Iv(F(total - 256, _ONE), F(total + 256, _ONE))
        for _ in range(halvings):
            r = r * r
        return r

    def signed(q: F) -> Iv:
        if q >= 0:
            return point(q)
        r = point(-q)
        return Iv(_down(1 / r.hi), _up(1 / r.lo))

    return Iv(signed(x.lo).lo, signed(x.hi).hi)


# Machin: pi = 16 arctan(1/5) - 4 arctan(1/239), alternating series.
def _pi() -> Iv:
    def atan_inv(n: int, terms: int) -> F:
        return sum(F((-1) ** i, (2 * i + 1) * n ** (2 * i + 1)) for i in range(terms))

    lo = 16 * atan_inv(5, 60) - 4 * atan_inv(239, 61)
    hi = 16 * atan_inv(5, 61) - 4 * atan_inv(239, 60)
    return Iv(_down(lo), _up(hi))


PI = _pi()
SQRT_2_OVER_PI = iv_sqrt(Iv.of(2) / PI)


def pmf(s: int, k: int, a: int) -> dict[int, F]:
    r"""
    The law of `\mathrm{Hyp}(s,k,a)`, on its support.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_2 import pmf
        >>> sum(pmf(9, 4, 5).values())
        Fraction(1, 1)
    """
    total = comb(s, a)
    return {x: F(comb(k, x) * comb(s - k, a - x), total)
            for x in range(max(0, a + k - s), min(a, k) + 1)}


def second_moment(s: int, k: int, a: int) -> F:
    r"""
    `E[4(X-a/2)^2/a]`, by summation over the law.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_2 import second_moment, second_moment_formula
        >>> second_moment(11, 3, 5) == second_moment_formula(11, 3, 5)
        True
    """
    return sum(p * 4 * (x - F(a, 2)) ** 2 / a for x, p in pmf(s, k, a).items())


def second_moment_formula(s: int, k: int, a: int) -> F:
    r"""
    The right side of part (a): `(a/s) v + ((s-a)/(s-1)) (1 - v/s)`, `v = 4e^2/s`.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_2 import second_moment_formula
        >>> second_moment_formula(4, 2, 2)
        Fraction(2, 3)
    """
    e = k - F(s, 2)
    v = 4 * e * e / s
    return F(a, s) * v + F(s - a, s - 1) * (1 - v / s)


def window_probability(s: int, k: int, a: int, b: F) -> F:
    r"""
    `P(|X-a/2| \le b\sqrt a)`, exactly. The test `(x-a/2)^2 \le b^2 a` is rational.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_2 import window_probability
        >>> window_probability(4, 2, 2, F(1, 4))
        Fraction(2, 3)
    """
    return sum((p for x, p in pmf(s, k, a).items() if (x - F(a, 2)) ** 2 <= b * b * a), F(0))


def variance_proxy(s: int, a: int) -> F:
    r"""
    `V(s,a) = (s-a)^2 \sum_{i=s-a}^{s-1} i^{-2}`, for `1 \le a \le s-1`.

    EXAMPLES::

        >>> from keytrees.gradient_analytic_2 import variance_proxy
        >>> variance_proxy(2, 1)
        Fraction(1, 1)
    """
    return (s - a) ** 2 * sum(F(1, i * i) for i in range(s - a, s))


def central_upper(n: int, z: F) -> Iv:
    r"""
    The upper bound of Lemma 1: `\sqrt{2/(\pi n)} \exp(-2z^2/(n+1))`.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_2 import central_upper
        >>> central_upper(2, F(0)).lo > F(1, 2)
        True
    """
    return SQRT_2_OVER_PI / iv_sqrt(Iv.of(n)) * iv_exp(Iv.of(-2 * z * z / (n + 1)))


def central_lower(n: int, z: F) -> Iv:
    r"""
    The lower bound of Lemma 1:
    `\sqrt{2/(\pi(n+1))} \exp(-(2z^2/(n+1))/(1-w^2))`, `w = (2|z|-1)_+/(n+1)`.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_2 import central_lower
        >>> central_lower(1, F(1, 2)).hi < F(1, 2)
        True
    """
    w = max(2 * abs(z) - 1, F(0)) / (n + 1)
    expo = -(2 * z * z / (n + 1)) / (1 - w * w)
    return SQRT_2_OVER_PI / iv_sqrt(Iv.of(n + 1)) * iv_exp(Iv.of(expo))


def sqrt_upper(x: F) -> F:
    r"""
    A rational upper bound for `\sqrt x`, within `2^{-160}`.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_2 import sqrt_upper
        >>> sqrt_upper(F(4)) >= 2
        True
    """
    return iv_sqrt(Iv.of(x)).hi


def amax(T: F, s_star: int, theta: F = F(3, 5)) -> Iv:
    r"""
    `A(T,s_*)` of part (b); requires `s_* > 4T^2`.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_2 import amax
        >>> 1.6286 < float(amax(F(0), 10**9).lo) < 1.6288
        True
    """
    assert s_star > 4 * T * T
    pref = iv_sqrt(Iv.of(2) / (PI * Iv.of(theta * (1 - theta))) * Iv.of(1 + F(1, s_star)))
    return pref * iv_exp(Iv.of(2 * T * T / s_star + 8 * T**4 / (s_star - 4 * T * T)))


def g_alpha(alpha: F, b: F, T: F) -> Iv:
    r"""
    `g_\alpha(b,T) = (1-\alpha)^{-1/2} \exp(-(2\alpha T^2 + 2b^2/3)/(1-\alpha))`.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_2 import g_alpha
        >>> 1.41 < float(g_alpha(F(1, 2), F(0), F(0)).lo) < 1.42
        True
    """
    return iv_exp(Iv.of(-(2 * alpha * T * T + F(2, 3) * b * b) / (1 - alpha))) / iv_sqrt(Iv.of(1 - alpha))


def g_min(b: F, T: F, theta: F = F(3, 5)) -> Iv:
    r"""
    `g(b,T)`, the smaller of `g_\alpha` at the two ends `\alpha = 1-\theta, \theta`.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_2 import g_min
        >>> 1.29 < float(g_min(F(0), F(0)).lo) < 1.30
        True
    """
    p, q = g_alpha(1 - theta, b, T), g_alpha(theta, b, T)
    return Iv(min(p.lo, q.lo), min(p.hi, q.hi))


def _r_upper(b: F, T: F, theta: F) -> Iv:
    return Iv.of(T) + Iv.of(b) * iv_sqrt(Iv.of(theta))


def correction(b: F, T: F, s: int, theta: F = F(3, 5)) -> Iv:
    r"""
    `\Lambda(b,T,s)` of part (c). Requires `b\sqrt{(1-\theta)s} \ge 1/2` and `\rho < 1`.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_2 import correction
        >>> 0.99 < float(correction(F(1, 4), F(1), 10**9).lo) < 1
        True
    """
    one = Iv.of(1)
    gap = Iv.of(1 - theta)
    R = _r_upper(b, T, theta)
    rho2 = Iv.of(4) * R * R / (gap * gap * Iv.of(s))
    assert rho2.hi < 1, "rho must be below one"
    u = Iv.of(b) * iv_sqrt(gap * Iv.of(s))
    assert u.lo >= F(1, 2) or (b * b * (1 - theta) * s >= F(1, 4)), "window too narrow"
    w = one - one / (Iv.of(2) * u)
    nu = Iv(max(F(1, 2), w.lo), max(F(1, 2), w.hi))
    size = one / (one + one / (gap * Iv.of(s)))
    t3 = Iv.of(F(2, 3) * b) / (gap * iv_sqrt(gap * Iv.of(s)))
    err = (Iv.of(2 * b * b) + Iv.of(2) * R * R / gap) * rho2 / (one - rho2)
    return nu * size * iv_exp(-(t3 + err))


def c_bound(b: F, T: F, s: int, theta: F = F(3, 5)) -> Iv:
    r"""
    `c(b,T;s) = 2b\sqrt{2/\pi}\, g(b,T)\, \Lambda(b,T,s)`.

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_2 import c_bound
        >>> 0 < float(c_bound(F(1, 4), F(1, 2), 2000).lo) < 1
        True
    """
    return Iv.of(2 * b) * SQRT_2_OVER_PI * g_min(b, T, theta) * correction(b, T, s, theta)


def master_lower(s: int, k: int, a: int, b: F, T: F, theta: F = F(3, 5)) -> Iv:
    r"""
    The right side of the master inequality of part (c).

    EXAMPLES::

        >>> from fractions import Fraction as F
        >>> from keytrees.gradient_analytic_2 import master_lower, window_probability
        >>> master_lower(200, 103, 100, F(1, 4), F(1, 4)).hi <= window_probability(200, 103, 100, F(1, 4))
        True
    """
    e = k - F(s, 2)
    ap, bp = a + 1, s - a + 1
    # Lattice count: y in a/2 + Z with y^2 <= b^2 a; and u(u+1)/3 with u = b sqrt(a).
    count = sum(1 for x in range(0, a + 1) if (x - F(a, 2)) ** 2 <= b * b * a)
    u = Iv.of(b) * iv_sqrt(Iv.of(a))
    kappa = Iv.of(F(2 * (s + 2), ap * bp))
    centre = Iv.of(F(2 * ap, bp * (s + 2)) * e * e)
    gap = Iv.of(1 - theta)
    R = _r_upper(b, T, theta)
    rho2 = Iv.of(4) * R * R / (gap * gap * Iv.of(s))
    assert rho2.hi < 1
    err = (Iv.of(2 * b * b) + Iv.of(2) * R * R / gap) * rho2 / (Iv.of(1) - rho2)
    expo = centre + kappa * u * (u + Iv.of(1)) / Iv.of(3) + err
    return Iv.of(count) * SQRT_2_OVER_PI * iv_sqrt(Iv.of(F(s, ap * bp))) * iv_exp(-expo)
