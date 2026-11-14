r"""
The gradient key on random tables: bounds of ``articles/gradient-bound``.

A table is an `n \times m` binary matrix with distinct columns, uniform over
the set `\mathcal A_{n,m}`. The gradient key `G` tests, at each fork, the first
row of largest quality `\min(k, s-k)`, where `k` of the `s` columns at the fork
have a one in that row. The cost of a tree is its mean leaf depth.

The functions here fall into three groups.

- Exact values for small tables: :func:`gradient_cost`, :func:`exact_average`,
  :func:`complete_cost`, :func:`distinct_probability`, :func:`quality_le1_probability`.
- Exact or interval-rigorous constants used in the proofs:
  :func:`balanced_probability`, :func:`fail_probability`,
  :func:`theorem1_excess_bound`, :func:`chain_upper`, :func:`tail_mu`,
  :func:`kappa_column`, :func:`rho_upper`, :func:`large_failure_log2`.
- Plain floating-point versions for plots and checks: :func:`chain`, :func:`rho`.

Rigorous constants use the interval arithmetic of ``mpmath.iv``; they return
upper bounds as ``mpmath.mpf`` values. Floating-point functions are not proofs.

AUTHORS:

- Written for the article on the gradient key (2026).

EXAMPLES::

    >>> from keytrees.gradient_bound import exact_average, complete_cost
    >>> exact_average(3, 4)
    Fraction(71, 35)
    >>> complete_cost(4)
    Fraction(2, 1)
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import itertools
import math
from fractions import Fraction
from math import comb

import mpmath
from mpmath import iv

__all__ = [
    "gradient_cost",
    "exact_average",
    "path_formula",
    "path_formula_terms",
    "complete_cost",
    "complete_redundancy_sup",
    "distinct_probability",
    "quality_le1_probability",
    "balanced_probability",
    "fail_probability",
    "theorem1_excess_bound",
    "chain",
    "chain_upper",
    "tail_mu",
    "best_tail_alpha",
    "kappa_column",
    "rho",
    "rho_upper",
    "cylinder_tail",
    "large_failure_log2",
]


# ---------------------------------------------------------------- exact small values


def gradient_cost(columns, n):
    r"""
    Return the cost (mean leaf depth) of the gradient key on a set of columns.

    INPUT:

    - ``columns`` -- distinct integers in `[0, 2^n)`; bit `r` is row `r`
    - ``n`` -- number of rows

    OUTPUT: a ``Fraction``

    EXAMPLES:

    The words 000, 010, 110, 111 of the random-words chapter, written with the
    first position as bit 0 (so 110 is 3)::

        >>> gradient_cost([0b000, 0b010, 0b011, 0b111], 3)
        Fraction(2, 1)
        >>> gradient_cost([1, 2, 4], 3)
        Fraction(5, 3)
    """
    m = len(columns)

    def depth_sum(cols):
        s = len(cols)
        if s <= 1:
            return 0
        best, bestq = None, 0
        for r in range(n):
            k = sum((c >> r) & 1 for c in cols)
            q = min(k, s - k)
            if q > bestq:
                best, bestq = r, q
        if best is None:
            raise ValueError("columns are not distinct")
        one = [c for c in cols if (c >> best) & 1]
        zero = [c for c in cols if not (c >> best) & 1]
        return s + depth_sum(one) + depth_sum(zero)

    return Fraction(depth_sum(list(columns)), m)


def exact_average(n, m):
    r"""
    Return `t_{n,m}(G)`, the average cost of the gradient key over `\mathcal A_{n,m}`.

    The cost does not depend on the order of the columns, so the average is
    taken over `m`-subsets of `\{0,1\}^n`. Brute force; the C++ engine
    ``gradient_bound exact`` does the same for larger cases.

    EXAMPLES:

    Agreement with the monograph's `t_{n,4}(G) = 2 + p_{n,4}/4`::

        >>> all(exact_average(n, 4) == 2 + quality_le1_probability(n, 4) / 4 for n in (2, 3, 4))
        True
        >>> exact_average(3, 5) == Fraction(12, 5) + Fraction(2, 5) * quality_le1_probability(3, 5)
        True
    """
    words = range(2**n)
    total = Fraction(0)
    count = 0
    for cols in itertools.combinations(words, m):
        total += gradient_cost(cols, n)
        count += 1
    return total / count


def _integer_partitions(a, top=None):
    top = a if top is None else top
    if a == 0:
        yield []
        return
    for v in range(min(a, top), 0, -1):
        for rest in _integer_partitions(a - v, v):
            yield [v] + rest


def _partition_weight(lam):
    a = sum(lam)
    c = math.factorial(a)
    for v in lam:
        c //= v
    for v in set(lam):
        c //= math.factorial(lam.count(v))
    return -c if (a - len(lam)) % 2 else c


def path_formula(n, m):
    r"""
    Return `t_{n,m}(G)` from the path formula of the article (exact, slow reference version).

    The sum runs over chains `m = s_0 > \dots > s_d \ge 2` and one integer
    partition of each cell size; see ``src/cpp/gradient_bound.cpp`` for the fast
    version. No chain is pruned here. The terms are :func:`path_formula_terms`.

    EXAMPLES::

        >>> all(path_formula(n, m) == exact_average(n, m) for n in (2, 3, 4) for m in range(2, min(2**n, 6) + 1))
        True
        >>> path_formula(20, 3)
        Fraction(5, 3)
    """

    tables = math.prod(2**n - i for i in range(m))
    total = sum(mult * coef * fn for mult, coef, fn in path_formula_terms(n, m).values())
    return Fraction(total, tables)


def path_formula_terms(n, m):
    r"""
    Return the terms of the path formula: a dictionary from (chain, partitions of
    the cells) to the triple (`\prod_j\binom{s_j-1}{s_{j+1}-1}`, `\prod_j c(\lambda_j)`, `F_n`).

    No chain is pruned, so pruned chains appear with `F_n = 0`.

    EXAMPLES::

        >>> terms = path_formula_terms(3, 4)
        >>> sorted({c: sum(a * b * f for (cc, _), (a, b, f) in terms.items() if cc == c) for c, _ in terms}.items())
        [((4,), 1680), ((4, 2), 1488), ((4, 3), 144), ((4, 3, 2), 96)]
    """

    def chains(s):
        yield [s]
        for nxt in range(2, s[-1] if isinstance(s, list) else s):
            for rest in chains(nxt):
                yield [s] + rest

    def count(cells, q, strict, inner):
        """Assignments with the given polynomial on the innermost set, then outward."""
        poly = inner
        for lam, qj, st in zip(reversed(cells), reversed(q), reversed(strict)):
            for v in lam:
                new = [0] * (len(poly) + v)
                for k, c in enumerate(poly):
                    new[k] += c
                    new[k + v] += c
                poly = new
            size = len(poly) - 1
            poly = [c if min(k, size - k) <= qj - st else 0 for k, c in enumerate(poly)]
        return sum(poly)

    terms = {}
    for s in chains(m):
        d = len(s) - 1
        a = [s[j] - s[j + 1] for j in range(d)]
        q = [min(a[j], s[j + 1]) for j in range(d)]
        mult = math.prod(comb(s[j] - 1, s[j + 1] - 1) for j in range(d))
        for lams in itertools.product(*[list(_integer_partitions(x)) for x in a + [s[-1]]]):
            coef = math.prod(_partition_weight(lam) for lam in lams)
            base = [1]
            for v in lams[d]:
                new = [0] * (len(base) + v)
                for k, c in enumerate(base):
                    new[k] += c
                    new[k + v] += c
                base = new
            f = {0: 1}
            for _ in range(n):
                g = {}
                for mask, val in f.items():
                    strict = [0 if mask >> j & 1 else 1 for j in range(d)]
                    w = count(lams[:d], q, strict, base)
                    g[mask] = g.get(mask, 0) + val * w
                    for j in range(d):
                        if mask >> j & 1:
                            continue
                        own = [0] * (s[j] + 1)
                        own[s[j + 1]] += 1
                        own[a[j]] += 1
                        w = count(lams[:j], q[:j], strict[:j], own)
                        g[mask | 1 << j] = g.get(mask | 1 << j, 0) + val * w
                f = g
            terms[tuple(s), tuple(map(tuple, lams))] = (mult, coef, f.get((1 << d) - 1, 0))
    return terms


def complete_cost(m):
    r"""
    Return the cost `l + 2 - 2^{l+1}/m` of the complete tree on `m` leaves, `l = \lfloor\log m\rfloor`.

    EXAMPLES::

        >>> [complete_cost(m) for m in (1, 2, 3, 5, 8)]
        [Fraction(0, 1), Fraction(1, 1), Fraction(5, 3), Fraction(12, 5), Fraction(3, 1)]
    """
    if m == 1:
        return Fraction(0)
    l = m.bit_length() - 1
    return Fraction(l + 2) - Fraction(2 ** (l + 1), m)


def complete_redundancy_sup():
    r"""
    Return `\gamma = 2 - 1/\ln 2 - \log(2\ln 2)`, the supremum of `C(m) - \log m`.

    EXAMPLES::

        >>> round(complete_redundancy_sup(), 6)
        0.086071
    """
    return 2 - 1 / math.log(2) - math.log2(2 * math.log(2))


def distinct_probability(n, m):
    r"""
    Return the probability that `m` independent uniform words of length `n` are distinct.

    EXAMPLES::

        >>> distinct_probability(2, 2)
        Fraction(3, 4)
        >>> distinct_probability(3, 9)
        Fraction(0, 1)
    """
    p = Fraction(1)
    for i in range(m):
        p *= Fraction(2**n - i, 2**n)
    return p


def quality_le1_probability(n, m):
    r"""
    Return `p_{n,m}`, the probability that a matrix of `\mathcal A_{n,m}` has
    quality at most 1 (the monograph's lemma on quality, `m \ge 3`).

    EXAMPLES::

        >>> quality_le1_probability(3, 4)
        Fraction(4, 35)
    """
    num = (m + 1) ** n - sum((-1) ** i * (i - 1) * comb(m, i) * (m - i + 1) ** n for i in range(2, m + 1))
    den = 1
    for i in range(m):
        den *= 2**n - i
    return Fraction(2**n * num, den)


# ---------------------------------------------------------------- one row on s columns


def balanced_probability(s):
    r"""
    Return `\beta_s`, the probability that a uniform row on `s` columns has
    quality `\lfloor s/2\rfloor`.

    EXAMPLES::

        >>> [balanced_probability(s) for s in (2, 3, 4, 5, 6)]
        [Fraction(1, 2), Fraction(3, 4), Fraction(3, 8), Fraction(5, 8), Fraction(5, 16)]
        >>> all(balanced_probability(s) >= 1 / math.sqrt(2 * s) for s in range(2, 200))
        True
    """
    return 1 - fail_probability(s, s // 2)


def fail_probability(s, k):
    r"""
    Return `f_k(s)`, the probability that a uniform row on `s` columns has quality
    less than `k`, for `1 \le k \le \lfloor s/2\rfloor`.

    EXAMPLES::

        >>> fail_probability(4, 2), fail_probability(4, 1)
        (Fraction(5, 8), Fraction(1, 8))

    The second value is the probability that the row is constant::

        >>> fail_probability(6, 3) == 1 - balanced_probability(6)
        True
    """
    if not 1 <= k <= s // 2:
        raise ValueError("need 1 <= k <= s//2")
    return Fraction(2 * sum(comb(s, j) for j in range(k)), 2**s)


def theorem1_excess_bound(n, m):
    r"""
    Return the bound of Theorem 1 on `t_{n,m}(G) - C(m)`:

    .. MATH::

        \Bigl(\frac{m+1}2 - \frac1m - C(m)\Bigr)\frac{\sum_{s=4}^m \binom ms (1-\beta_s)^n}{P(D)} .

    EXAMPLES:

    For `m = 4` the bound is close to the exact excess `p_{n,4}/4`::

        >>> n = 10
        >>> exact = quality_le1_probability(n, 4) / 4
        >>> bound = theorem1_excess_bound(n, 4)
        >>> exact <= bound, float(bound / exact) < 1.04
        (True, True)
        >>> theorem1_excess_bound(10, 3)
        Fraction(0, 1)
    """
    cmax = Fraction(m + 1, 2) - Fraction(1, m)
    fail = sum(comb(m, s) * (1 - balanced_probability(s)) ** n for s in range(4, m + 1))
    return (cmax - complete_cost(m)) * fail / distinct_probability(n, m)


# ---------------------------------------------------------------- blocked key: the size chain


def _h(x):
    if x <= 0 or x >= 1:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def chain(b, smax, skip=True):
    r"""
    Return `[r_b(0), r_b(1), \dots, r_b(s_{\max})]` in floating point.

    `r_b(s)` is the expected redundancy of the key that, at every fork of size
    `s`, tests the best of `b` fresh uniform rows. With ``skip`` a row block of
    quality 0 is passed over at no cost (the key `G_b` of the article); without
    it such a test is charged.

    EXAMPLES:

    `b = 1` with skipping is the PATRICIA recursion, whose limit redundancy is
    about `0.3327`; `b = 1` without skipping is the trie::

        >>> r = chain(1, 64)
        >>> r[2], round(r[3], 6)
        (0.0, 0.081704)
        >>> round(r[64], 3), round(chain(1, 64, skip=False)[64], 3)
        (0.321, 1.321)
    """
    r = [0.0] * (smax + 1)
    for s in range(2, smax + 1):
        pm = [comb(s, k) / 2**s for k in range(s + 1)]
        half = s // 2
        c = [0.0] * (half + 2)  # c[q] = P(one row has quality < q)
        acc = 0.0
        for q in range(1, half + 1):
            acc += pm[q - 1]
            c[q] = 2 * acc
        c[half + 1] = 1.0
        p0 = c[1] ** b
        tot = p0 if not skip else 0.0
        for q in range(1, half + 1):
            pq = c[q + 1] ** b - c[q] ** b
            x = q / s
            tot += pq * (1 - _h(x) + x * r[q] + (1 - x) * r[s - q])
        r[s] = tot / (1 - p0)
    return r


def chain_upper(b, s0, prec=64):
    r"""
    Return rigorous upper bounds for `r_b(1), \dots, r_b(s_0)` (skipping version).

    Uses interval arithmetic; the result is a list of ``mpf`` upper bounds,
    index `s`.

    EXAMPLES::

        >>> u = chain_upper(2, 40)
        >>> f = chain(2, 40)
        >>> all(f[s] <= float(u[s]) + 1e-12 and float(u[s]) - f[s] < 1e-9 for s in range(1, 41))
        True
    """
    iv.prec = prec
    one = iv.mpf(1)
    ln2 = iv.log(2)
    lo = [iv.mpf(0)] * (s0 + 1)
    for s in range(2, s0 + 1):
        scale = iv.mpf(2) ** (-s)
        half = s // 2
        c = [iv.mpf(0)] * (half + 2)
        acc = 0
        for q in range(1, half + 1):
            acc += comb(s, q - 1)
            c[q] = 2 * iv.mpf(acc) * scale
        c[half + 1] = one
        cb = [x**b for x in c]
        tot = iv.mpf(0)
        for q in range(1, half + 1):
            pq = cb[q + 1] - cb[q]
            x = iv.mpf(q) / s
            y = one - x
            hq = -(x * iv.log(x) + y * iv.log(y)) / ln2
            tot += pq * (one - hq + x * lo[q] + y * lo[s - q])
        lo[s] = tot / (one - cb[1])
    return [mpmath.mpf(v.b) for v in lo]


def tail_mu(s0, alpha):
    r"""
    Return a rigorous upper bound for the constant `\mu` of the tail lemma:

    .. MATH::

        \mu = \frac{(s_0+1)^{\alpha-1}}{\phi_\alpha(1/4)\,(1 - 2e^{-(s_0+1)/8})},
        \qquad \phi_\alpha(x) = x^{1-\alpha} + (1-x)^{1-\alpha} - 1 .

    EXAMPLES::

        >>> float(tail_mu(1000, 0.156)) < 0.031
        True
    """
    iv.prec = 64
    a = iv.mpf(alpha)
    t = iv.mpf(s0 + 1)
    phi = iv.mpf(0.25) ** (1 - a) + iv.mpf(0.75) ** (1 - a) - 1
    val = t ** (a - 1) / (phi * (1 - 2 * iv.exp(-t / 8)))
    return mpmath.mpf(val.b)


def best_tail_alpha(s0):
    r"""
    Return the `\alpha` in `\{0.001, \dots, 0.999\}` minimising :func:`tail_mu`, in floating point.

    EXAMPLES::

        >>> best_tail_alpha(1000)
        0.156
    """
    def f(a):
        return (s0 + 1) ** (a - 1) / (0.25 ** (1 - a) + 0.75 ** (1 - a) - 1)

    return min((f(i / 1000), i / 1000) for i in range(1, 1000))[1]


# ---------------------------------------------------------------- the gradient key itself


def kappa_column(K, smax, margin=3):
    r"""
    Return the quality requirement `\kappa_K(s)` for `2 \le s \le s_{\max}` (index `s`).

    `\kappa_K(s)` is the largest `k \le \lfloor s/2 \rfloor` with
    `f_k(s)^K \le 2^{-(s+\text{margin})}`, or 1 if there is none. Then
    `\binom ms f_{\kappa(s)}(s)^n \le m^{-\text{margin}}/s!` whenever `n \ge K\log m`.
    The test is done in exact integer arithmetic.

    EXAMPLES::

        >>> kappa_column(30, 10)[2:]
        [1, 1, 2, 2, 3, 3, 4, 4, 4]
        >>> k = kappa_column(30, 60)
        >>> all(k[s] == 1 or fail_probability(s, k[s]) ** 30 <= Fraction(1, 2 ** (s + 3)) for s in range(2, 61))
        True
    """
    kap = [0, 0]
    for s in range(2, smax + 1):
        best = 1
        acc = 0
        lim = K * (s - 1) - s - margin  # need acc^K <= 2^lim, acc = sum_{j<k} C(s, j)
        for k in range(1, s // 2 + 1):
            acc += comb(s, k - 1)
            if k == 1:
                continue
            if lim >= 0 and acc**K <= 1 << lim:
                best = k
            else:
                break
        kap.append(best)
    return kap


def rho(kappa, smax):
    r"""
    Return the largest redundancy of a `\kappa`-admissible tree, for `s \le s_{\max}`, in floating point.

    A tree is `\kappa`-admissible when every fork of size `s` has a part of
    size at least `\kappa(s)`.

    EXAMPLES::

        >>> r = rho([0, 0] + [s // 2 for s in range(2, 20)], 19)
        >>> all(abs(r[s] - (float(complete_cost(s)) - math.log2(s))) < 1e-12 for s in range(1, 20))
        True
    """
    r = [0.0] * (smax + 1)
    for s in range(2, smax + 1):
        r[s] = max(1 - _h(k / s) + k / s * r[k] + (s - k) / s * r[s - k] for k in range(kappa[s], s // 2 + 1))
    return r


def rho_upper(kappa, smax, prec=64):
    r"""
    Return rigorous upper bounds for :func:`rho`, as a list of ``mpf``.

    EXAMPLES::

        >>> k = kappa_column(30, 50)
        >>> u, f = rho_upper(k, 50), rho(k, 50)
        >>> all(f[s] <= float(u[s]) + 1e-12 and float(u[s]) - f[s] < 1e-9 for s in range(1, 51))
        True
    """
    iv.prec = prec
    one = iv.mpf(1)
    ln2 = iv.log(2)
    up = [mpmath.mpf(0)] * (smax + 1)
    for s in range(2, smax + 1):
        best = mpmath.mpf(0)
        for k in range(kappa[s], s // 2 + 1):
            x = iv.mpf(k) / s
            y = one - x
            hk = -(x * iv.log(x) + y * iv.log(y)) / ln2
            v = one - hk + x * up[k] + y * up[s - k]
            best = max(best, mpmath.mpf(v.b))
        up[s] = best
    return up


def cylinder_tail(s1, x):
    r"""
    Return `4x/(s_1(1-\theta))` with `\theta = 1/2 + \sqrt{x/s_1}` (rigorous upper bound).

    It bounds the part of the redundancy that comes from forks of size more
    than `s_1` in a tree whose forks of size `s > s_1` have imbalance at most `\sqrt{xs}`.

    EXAMPLES::

        >>> round(float(cylinder_tail(500, 1)), 4)
        0.0176
    """
    iv.prec = 64
    th = iv.mpf(0.5) + iv.sqrt(iv.mpf(x) / s1)
    return mpmath.mpf((4 * iv.mpf(x) / (s1 * (1 - th))).b)


def large_failure_log2(n, m, s1, x):
    r"""
    Return an upper bound for `\log_2` of `\frac m2 (d^*+1)(2n)^{d^*}g^{n-d^*}`,
    with `g = 2e^{-2x}` and `d^* = \lfloor \log(m/s_1)/\log(1/\theta)\rfloor`.
    Returns ``-inf`` when `m \le s_1`.

    EXAMPLES::

        >>> large_failure_log2(1000, 2**20, 500, 1) < -10
        True
        >>> large_failure_log2(100, 400, 500, 1)
        -inf
    """
    if m <= s1:
        return -math.inf
    theta = 0.5 + math.sqrt(x / s1)
    dstar = math.floor(math.log2(m / s1) / -math.log2(theta) + 1e-12)
    if n <= dstar:
        return math.inf
    log2g = 1 - 2 * x / math.log(2)
    val = math.log2(m / 2) + math.log2(dstar + 1) + dstar * math.log2(2 * n) + (n - dstar) * log2g
    return val + 1e-9
