r"""
Average cost of the random algorithm `R_s` on random sets of binary words

Exact arithmetic behind the article in ``articles/random-algorithm-bound/``.
A set of `m` distinct binary words of length `n` is drawn uniformly; `t_{n,m}`
is the average cost of the tree built by `R_s` (monograph, chapter on random
words). For `m \ge 4` the algorithm `R_3` obeys

.. MATH::

    t_{n,m} = 1 + \sum_k c_{n,m,k}\,\varphi(k), \qquad
    \varphi(k) = \frac{k\,t_{n-1,k} + (m-k)\,t_{n-1,m-k}}{m},

where `c_{n,m,k}` is hypergeometric. This module computes `t_{n,m}` exactly,
checks the convexity facts used to prove that `t_{n,m}` never decreases with
`n`, and verifies the base range of the uniform bound
`t_m \le \log_2 m + c^*` with rigorous rational enclosures.

EXAMPLES::

    >>> t = finite_costs(4)
    >>> t[3][4], t[4][4]
    (Fraction(15, 7), Fraction(1006, 455))
    >>> all(t[n][m] >= t[n - 1][m] for n in range(2, 5) for m in range(2, 2**(n - 1) + 1))
    True
    >>> mellin_offset(3)
    Fraction(-5, 8)
"""
from __future__ import annotations

from fractions import Fraction
from math import comb

INF = None  # threshold s = infinity (PATRICIA)

# ---------------------------------------------------------------------------
# Finite n


def split_weights(n: int, m: int) -> list[Fraction]:
    r"""
    Return `c_{n,m,k}`, `k = 0..m`: the probability that a tested position has
    `k` ones, for a uniform set of `m` distinct words of length `n`.

    EXAMPLES::

        >>> split_weights(2, 3)
        [Fraction(0, 1), Fraction(1, 2), Fraction(1, 2), Fraction(0, 1)]
        >>> sum(split_weights(5, 7))
        Fraction(1, 1)
    """
    h = 2 ** (n - 1)
    d = comb(2 * h, m)
    return [Fraction(comb(h, k) * comb(h, m - k), d) for k in range(m + 1)]


def finite_costs(n_max: int, s: int | None = 3) -> dict[int, list[Fraction]]:
    r"""
    Return ``t`` with ``t[n][m]`` `= t_{n,m}(R_s)` for `1 \le n \le n_{max}`,
    `0 \le m \le 2^n`, exactly. ``s=None`` is `R_\infty` (PATRICIA).

    EXAMPLES::

        >>> t = finite_costs(3)
        >>> t[3][5], t[3][8]
        (Fraction(17, 7), Fraction(3, 1))
        >>> finite_costs(3, s=None)[3][4]
        Fraction(74, 35)
        >>> finite_costs(2, s=1)[2][2]
        Fraction(4, 3)
    """
    t: dict[int, list[Fraction]] = {0: [Fraction(0), Fraction(0)]}
    for n in range(1, n_max + 1):
        h = 2 ** (n - 1)
        row = [Fraction(0)] * (2**n + 1)
        prev = t[n - 1]
        for m in range(2, 2**n + 1):
            d = comb(2 * h, m)
            lo, hi = max(0, m - h), min(m, h)
            charged = s is not None and m > s
            num = 0  # numerator over d * m
            for k in range(lo, hi + 1):
                w = comb(h, k) * comb(h, m - k)
                if k in (0, m):
                    # a test that does not split: charged only if m > s
                    num += w * (m * prev[m] + (m if charged else 0))
                else:
                    num += w * (m + k * prev[k] + (m - k) * prev[m - k])
            row[m] = Fraction(num, d * m)
        t[n] = row
    return t


def patricia_closed_form(n: int, m: int) -> Fraction:
    r"""
    The exact solution of Prodinger and Szpankowski for `R_\infty`:
    `t_{n,m} = n - \binom{2^n-1}{m-1}^{-1}\sum_{l=0}^{n-1}\binom{2^n-2^l-1}{m-1}`.

    EXAMPLES::

        >>> t = finite_costs(5, s=None)
        >>> all(patricia_closed_form(n, m) == t[n][m] for n in range(1, 6) for m in range(1, 2**n + 1))
        True
    """
    big = 2**n - 1
    return n - Fraction(sum(comb(big - 2**l, m - 1) for l in range(n)), comb(big, m - 1))


def second_differences(row: list[Fraction]) -> list[Fraction | None]:
    r"""
    Return `\Delta^2 u(j) = u(j-1) - 2u(j) + u(j+1)` for `u(k) = k\,t_{n,k}`,
    indexed by `j` (entries `0` and `2^n` are ``None``).

    EXAMPLES::

        >>> d = second_differences(finite_costs(4)[4])
        >>> d[1], d[2], d[4] < 0, d[15]
        (Fraction(2, 1), Fraction(1, 1), True, Fraction(0, 1))
    """
    u = [k * v for k, v in enumerate(row)]
    top = len(u) - 1
    return [None] + [u[j - 1] - 2 * u[j] + u[j + 1] for j in range(1, top)] + [None]


def invariants(row: list[Fraction]) -> dict[str, object]:
    r"""
    The facts about `u_n(k) = k\,t_{n,k}` used in the monotonicity proof:
    the indices `j` with `\Delta^2u(j) < 0`, and the values of
    `F_1 = \Delta^2u(3) + 2\Delta^2u(4)` and `F_2 = \Delta^2u(4) + \Delta^2u(5)`.

    EXAMPLES::

        >>> inv = invariants(finite_costs(6)[6])
        >>> inv["negative"], inv["F1"] > 0, inv["F2"] > 0
        ([4], True, True)
    """
    d = second_differences(row)
    neg = [j for j in range(1, len(d) - 1) if d[j] < 0]
    out: dict[str, object] = {"negative": neg}
    if len(d) > 5:
        out["F1"] = d[3] + 2 * d[4]
    if len(d) > 6:
        out["F2"] = d[4] + d[5]
    return out


def convex_order_gaps(n: int, m: int) -> list[Fraction]:
    r"""
    Return `D(j) = E(K_{n+1}-j)^+ - E(K_n-j)^+`, `j = 0..m`, where `K_n` is the
    number of ones in a row of a uniform `n \times m` matrix with distinct
    columns. `D \ge 0` is the convex order `K_n \le_{cx} K_{n+1}`.

    EXAMPLES::

        >>> D = convex_order_gaps(4, 8)
        >>> D[0], D[8], all(x >= 0 for x in D), D[3] == D[5]
        (Fraction(0, 1), Fraction(0, 1), True, True)
    """
    p, q = split_weights(n, m), split_weights(n + 1, m)
    return [sum((q[k] - p[k]) * max(k - j, 0) for k in range(m + 1)) for j in range(m + 1)]


# ---------------------------------------------------------------------------
# Polynomials with rational coefficients, lowest degree first


def _pmul(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _padd(a: list[Fraction], b: list[Fraction], sb: int = 1) -> list[Fraction]:
    n = max(len(a), len(b))
    a = a + [Fraction(0)] * (n - len(a))
    b = b + [Fraction(0)] * (n - len(b))
    return [x + sb * y for x, y in zip(a, b)]


def _binom_poly(scale: int, k: int, shift: int = 0) -> list[Fraction]:
    r"""`\binom{scale\cdot H + shift}{k}` as a polynomial in `H`."""
    out = [Fraction(1)]
    for i in range(k):
        out = _pmul(out, [Fraction(shift - i, i + 1), Fraction(scale, i + 1)])
    return out


def _taylor_shift(a: list[Fraction], h0: int) -> list[Fraction]:
    r"""Coefficients of `a(h_0 + x)` in `x`."""
    out: list[Fraction] = [Fraction(0)]
    for c in reversed(a):  # Horner: out = out * (h0 + x) + c
        out = _padd(_pmul(out, [Fraction(h0), Fraction(1)]), [c])
    return out


def gap_certificate(m: int = 8, j: int = 3, i: int = 4, kappa: Fraction = Fraction(1, 2),
                    h0: int = 128) -> list[Fraction]:
    r"""
    Certificate that `D_n(j) \ge \kappa D_n(i)` for all `n` with `2^{n-1} \ge h_0`.

    Writing `H = 2^{n-1}`, the quantity `(D(j) - \kappa D(i))\binom{2H}{m}\binom{4H}{m}`
    is a polynomial in `H`; the function returns its coefficients in `x = H - h_0`.
    All of them nonnegative proves the claim, since both binomials are positive.

    EXAMPLES::

        >>> cert = gap_certificate()
        >>> all(c >= 0 for c in cert), any(c > 0 for c in cert)
        (True, True)
        >>> D = convex_order_gaps(8, 8)
        >>> D[3] >= D[4] / 2
        True
    """
    def moment_numerator(scale: int, level: int) -> list[Fraction]:
        # sum_k C(sH,k) C(sH,m-k) (k-level)^+
        acc: list[Fraction] = [Fraction(0)]
        for k in range(m + 1):
            if k > level:
                term = _pmul(_binom_poly(scale, k), _binom_poly(scale, m - k))
                acc = _padd(acc, [c * (k - level) for c in term])
        return acc

    den1, den2 = _binom_poly(2, m), _binom_poly(4, m)
    def gap(level: int) -> list[Fraction]:
        # D(level) * den1 * den2 = A_2(H) den1 - A_1(H) den2
        return _padd(_pmul(moment_numerator(2, level), den1), _pmul(moment_numerator(1, level), den2), -1)
    poly = _padd(gap(j), [kappa * c for c in gap(i)], -1)
    return _taylor_shift(poly, h0)


def step_certificate(m0: int = 16) -> list[Fraction]:
    r"""
    Certificate that `f(m) \le \tfrac12 + \tfrac1{2m}` for every real `m \ge m_0`, where
    `f` is the function of the inductive step (see :func:`step_function`). The
    returned list holds the coefficients, in `x = m - m_0`, of
    `(\tfrac12 + \tfrac1{2m} - f(m))\cdot 12 m^2 (m+1)^4`.

    EXAMPLES::

        >>> all(c >= 0 for c in step_certificate())
        True
        >>> step_function(1025) < Fraction(1, 2) + Fraction(1, 2050)
        True
    """
    one = Fraction(1)
    mp1 = [one, one]  # m + 1
    mp1_2 = _pmul(mp1, mp1)
    mp1_4 = _pmul(mp1_2, mp1_2)
    m1 = [-one, one]  # m - 1 = N
    # 12 m^2 (m+1)^4 * f(m), term by term, with f(m) = m*(1/m - 1/(2m^2) + 1/(3m^3)
    #   - N/(2(m+1)^2) - N(3N-2)/(4(m+1)^4))
    t1 = _pmul([Fraction(0), Fraction(0), Fraction(12)], mp1_4)            # 12 m^2 (m+1)^4 * 1
    t2 = _pmul([Fraction(0), Fraction(-6)], mp1_4)                         # * (-1/(2m))
    t3 = _pmul([Fraction(4)], mp1_4)                                       # * 1/(3m^2)
    t4 = _pmul(_pmul([Fraction(0), Fraction(0), Fraction(0), Fraction(-6)], m1), mp1_2)  # -6 m^3 N (m+1)^2
    t5 = _pmul([Fraction(0), Fraction(0), Fraction(0), Fraction(-3)], _pmul(m1, [Fraction(-5), Fraction(3)]))
    f = t1
    for t in (t2, t3, t4, t5):
        f = _padd(f, t)
    # 12 m^2 (m+1)^4 * (1/2 + 1/(2m)) = (6 m^2 + 6 m)(m+1)^4
    g = _pmul([Fraction(0), Fraction(6), Fraction(6)], mp1_4)
    return _taylor_shift(_padd(g, f, -1), m0)


def step_function(m: int) -> Fraction:
    r"""
    `f(m) = m\bigl(\frac1m - \frac1{2m^2} + \frac1{3m^3} - \frac{E y^2}{2} - \frac{E y^4}{4}\bigr)`
    with `y = (1+K-\mu)/\mu`, `K \sim \mathrm{Bin}(m-1, 1/2)`, `\mu = (m+1)/2`:
    `E y^2 = (m-1)/(m+1)^2` and `E y^4 = (m-1)(3m-5)/(m+1)^4`.

    EXAMPLES::

        >>> float(step_function(10**6) - Fraction(1, 2)) * 10**6
        0.25000...
    """
    mm = Fraction(m)
    n = mm - 1
    ey2 = n / (mm + 1) ** 2
    ey4 = n * (3 * n - 2) / (mm + 1) ** 4
    return mm * (1 / mm - 1 / (2 * mm**2) + 1 / (3 * mm**3) - ey2 / 2 - ey4 / 4)


# ---------------------------------------------------------------------------
# The limit n -> infinity, with rigorous enclosures


def limit_costs(m_max: int, s: int | None = 3) -> list[Fraction]:
    r"""
    Exact limits `t_m = \lim_n t_{n,m}(R_s)`, `0 \le m \le m_{max}` (small `m` only:
    the denominators grow like `2^{m^2/2}`).

    EXAMPLES::

        >>> limit_costs(5)[4:]
        [Fraction(16, 7), Fraction(274, 105)]
        >>> limit_costs(4, s=None)[4]
        Fraction(15, 7)
    """
    t = [Fraction(0)] * (m_max + 1)
    for m in range(2, m_max + 1):
        q = Fraction(1, 2 ** (m - 1))
        inner = sum(comb(m - 1, k - 1) * t[k] for k in range(2, m)) * q
        if s is None or m <= s:
            t[m] = 1 + inner / (1 - q)
        else:
            t[m] = (1 + inner) / (1 - q)
    return t


def limit_enclosures(m_max: int, s: int | None = 3, bits: int = 96) -> tuple[list[int], list[int], int]:
    r"""
    Return integer lists ``lo``, ``hi`` and ``bits`` with
    `lo_m 2^{-bits} \le t_m \le hi_m 2^{-bits}` for `m \le m_{max}`.

    The recurrence has positive coefficients, so rounding every step outward
    keeps the bounds rigorous.

    EXAMPLES::

        >>> lo, hi, b = limit_enclosures(30)
        >>> exact = limit_costs(30)
        >>> all(Fraction(lo[m], 2**b) <= exact[m] <= Fraction(hi[m], 2**b) for m in range(31))
        True
        >>> (hi[30] - lo[30]) < 2**(b - 80)
        True
    """
    one = 1 << bits
    lo = [0] * (m_max + 1)
    hi = [0] * (m_max + 1)
    for m in range(2, m_max + 1):
        den = 2 ** (m - 1) - 1  # t_m (2^{m-1}-1) = 2^{m-1} [or 2^{m-1}-1] + sum C(m-1,k-1) t_k
        toll = 2 ** (m - 1) if (s is not None and m > s) else den
        slo = shi = 0
        c = 1  # C(m-1, k-1), starting at k = 1
        for k in range(1, m):
            if k >= 2:
                slo += c * lo[k]
                shi += c * hi[k]
            c = c * (m - k) // k
        lo[m] = (toll * one + slo) // den
        hi[m] = -((-(toll * one + shi)) // den)
    return lo, hi, bits


def ln2_bounds(bits: int = 128) -> tuple[Fraction, Fraction]:
    r"""
    Rational bounds for `\ln 2 = 2\operatorname{artanh}(1/3)`.

    EXAMPLES::

        >>> a, b = ln2_bounds()
        >>> float(a), float(b), 0 < b - a < Fraction(1, 2**100)
        (0.6931471805599453, 0.6931471805599453, True)
    """
    return _ln_bounds(Fraction(2), bits)


def _ln_bounds(x: Fraction, bits: int) -> tuple[Fraction, Fraction]:
    r"""Bounds for `\ln x`, `1 \le x \le 2`, by the artanh series with a geometric tail."""
    z = (x - 1) / (x + 1)  # 0 <= z <= 1/3
    total = Fraction(0)
    power = z
    k = 0
    while True:
        term = power / (2 * k + 1)
        total += term
        power *= z * z
        tail = power / (2 * k + 3) / (1 - z * z)
        if 2 * tail < Fraction(1, 2**bits):
            return 2 * total, 2 * (total + tail)
        k += 1


def log2_bounds(m: int, bits: int = 128) -> tuple[Fraction, Fraction]:
    r"""
    Rational bounds `lo \le \log_2 m \le hi`.

    EXAMPLES::

        >>> lo, hi = log2_bounds(3)
        >>> float(lo), hi - lo < Fraction(1, 2**100)
        (1.584962500721156, True)
        >>> log2_bounds(1024)
        (Fraction(10, 1), Fraction(10, 1))
    """
    e = m.bit_length() - 1
    x = Fraction(m, 2**e)
    if x == 1:
        return Fraction(e), Fraction(e)
    l2lo, l2hi = ln2_bounds(bits)
    lxlo, lxhi = _ln_bounds(x, bits)
    return e + lxlo / l2hi, e + lxhi / l2lo


def verify_base(c: Fraction, alpha: Fraction, k0: int, m_max: int, s: int | None = 3,
                bits: int = 96) -> dict[str, object]:
    r"""
    Check `t_m \le \log_2 m + c - \alpha/m` for `k_0 \le m \le m_{max}` and
    `t_m \le \log_2 m + c` for `2 \le m < k_0`, with rigorous enclosures.

    Returns the smallest margin found, where it occurs, and the largest lower
    bound of `t_m - \log_2 m` seen (a rigorous lower bound for the supremum).

    EXAMPLES::

        >>> r = verify_base(Fraction(4311, 10000), Fraction(723, 1000), 8, 200)
        >>> r["ok"], r["argmin"]
        (True, 92)
        >>> float(r["sup_lower"]) > 0.4274
        True
    """
    lo, hi, b = limit_enclosures(m_max, s, bits)
    scale = Fraction(1, 2**b)
    margin, argmin, sup_lower, arg_sup = None, None, Fraction(0), None
    ok = True
    for m in range(2, m_max + 1):
        l2lo, l2hi = log2_bounds(m, bits + 32)
        slack = alpha / m if m >= k0 else Fraction(0)
        mg = l2lo + c - slack - hi[m] * scale
        if mg < 0:
            ok = False
        if m >= k0 and (margin is None or mg < margin):
            margin, argmin = mg, m
        low = lo[m] * scale - l2hi
        if low > sup_lower:
            sup_lower, arg_sup = low, m
    return {"ok": ok, "margin": margin, "argmin": argmin, "sup_lower": sup_lower, "arg_sup": arg_sup}


def step_holds(m: int, alpha: Fraction, k0: int = 8, bits: int = 64) -> bool:
    r"""
    Check the inductive step at one `m` with rational arithmetic:
    `\log_2 e\, f(m) + \alpha\, m\, P(\mathrm{Bin}(m-1,1/2) \le k_0-2) \le \alpha(1-2^{1-m})`.

    EXAMPLES::

        >>> step_holds(1025, Fraction(723, 1000))
        True
        >>> step_holds(100, Fraction(723, 1000))
        False
    """
    l2lo, _ = ln2_bounds(bits)
    log2e_hi = 1 / l2lo
    tail = Fraction(sum(comb(m - 1, i) for i in range(k0 - 1)), 2 ** (m - 1))
    lhs = log2e_hi * step_function(m) + alpha * m * tail
    return lhs <= alpha * (1 - Fraction(1, 2 ** (m - 1)))


# ---------------------------------------------------------------------------
# Asymptotics


def toll_defects(s: int | None) -> dict[int, Fraction]:
    r"""
    The defects `a_k = \tau(k) - k`, `2 \le k \le s`, of the limit recurrence of `R_s`
    written as a trie recurrence `x_m = \tau(m) + 2^{1-m}\sum_k\binom mk x_k` for the
    total cost `x_m = m t_m`.

    EXAMPLES::

        >>> toll_defects(1), toll_defects(2), toll_defects(3)
        ({}, {2: Fraction(-1, 1)}, {2: Fraction(-1, 1), 3: Fraction(-3, 4)})
    """
    if s is None:
        raise ValueError("PATRICIA has a defect at every m; see the article")
    x = [Fraction(k) * v for k, v in enumerate(limit_costs(max(s, 1), s))]
    out = {}
    for m in range(2, s + 1):
        tau = x[m] - sum(comb(m, k) * x[k] for k in range(m + 1)) / 2 ** (m - 1)
        out[m] = tau - m
    return out


def mellin_offset(s: int) -> Fraction:
    r"""
    The rational `a_s` with `\lim (t_m - \log_2 m - P_s(\log_2 m)) = \gamma/\ln 2 + 1/2 + a_s/\ln 2`:
    `a_s = \sum_{k=2}^s a_k/(k(k-1))`.

    EXAMPLES::

        >>> mellin_offset(1), mellin_offset(2), mellin_offset(3)
        (Fraction(0, 1), Fraction(-1, 2), Fraction(-5, 8))
    """
    return sum((a / (k * (k - 1)) for k, a in toll_defects(s).items()), Fraction(0))


def fourier_coefficients(s: int, count: int = 4, dps: int = 30) -> list[complex]:
    r"""
    Coefficients `c_\ell`, `\ell = 1..count`, of the oscillation
    `P_s(x) = 2\,\mathrm{Re}\sum_{\ell\ge1} c_\ell e^{-2\pi i \ell x}` with
    `c_\ell = T^*(-1+\chi_\ell)/\ln 2`, `\chi_\ell = 2\pi i\ell/\ln 2` and
    `T^*(s) = -\Gamma(s+1) + \sum_k a_k\Gamma(s+k)/k!`. Needs ``mpmath``.

    EXAMPLES::

        >>> c = fourier_coefficients(3, 2)
        >>> round(abs(c[0]) * 1e6, 3)
        8.546
    """
    import mpmath

    mpmath.mp.dps = dps
    ln2 = mpmath.log(2)
    defects = toll_defects(s)
    out = []
    for ell in range(1, count + 1):
        z = -1 + 2j * mpmath.pi * ell / ln2
        v = -mpmath.gamma(z + 1)
        for k, a in defects.items():
            v += mpmath.mpf(a.numerator) / a.denominator * mpmath.gamma(z + k) / mpmath.factorial(k)
        out.append(complex(v / ln2))
    return out


# ---------------------------------------------------------------------------
# Data files for the article


ALPHA = Fraction(723, 1000)
C_STAR = Fraction(4311, 10000)
C_PATRICIA = Fraction(33276, 100000)
BASE_RANGE = 1024


def write_data(directory: str) -> None:
    r"""
    Write ``base_check.csv`` and ``constants.csv`` into ``directory``
    (``data/random-algorithm-bound`` in the repository).
    """
    import csv
    import math
    import os

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "base_check.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "c", "alpha", "k0", "m_max", "ok", "min_margin", "argmin",
                    "sup_lower", "arg_sup_lower"])
        for name, s, c in (("s3", 3, C_STAR), ("patricia", None, C_PATRICIA)):
            r = verify_base(c, ALPHA, 8, BASE_RANGE, s)
            w.writerow([name, str(c), str(ALPHA), 8, BASE_RANGE, r["ok"], f"{float(r['margin']):.6e}",
                        r["argmin"], f"{float(r['sup_lower']):.10f}", r["arg_sup"]])
    gamma = 0.57721566490153286061
    ln2 = math.log(2)
    with open(os.path.join(directory, "constants.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "offset_a", "kappa", "abs_c1", "max_P"])
        for s in (1, 2, 3):
            a = mellin_offset(s)
            kappa = gamma / ln2 + 0.5 + float(a) / ln2
            coef = fourier_coefficients(s, 6)
            grid = [i / 4000 for i in range(4000)]
            top = max(2 * sum((c * complex(math.cos(2 * math.pi * (k + 1) * x),
                                           -math.sin(2 * math.pi * (k + 1) * x))).real
                              for k, c in enumerate(coef)) for x in grid)
            w.writerow([f"s{s}", str(a), f"{kappa:.12f}", f"{abs(coef[0]):.6e}", f"{top:.6e}"])


if __name__ == "__main__":  # pragma: no cover
    import sys

    write_data(sys.argv[1] if len(sys.argv) > 1 else "data/random-algorithm-bound")
