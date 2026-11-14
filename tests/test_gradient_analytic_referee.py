# SPDX-License-Identifier: MIT
"""Referee probes for the analytic gradient-key steps.

See articles/gradient-bound/analytic/REFEREE.md. Independent re-implementations
(no helpers from the step authors), exact rational arithmetic where it matters.
These are finite checks of the statements refereed, not proofs.
"""

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations, product
from math import comb, erf, exp, log, log2, pi, sqrt

import pytest


# ---------------------------------------------------------------- G itself

def bit(col, r):
    return (col >> r) & 1


def quality(cols, S, r):
    k = sum(bit(cols[c], r) for c in S)
    return min(k, len(S) - k)


def fork_row(cols, S, n):
    """First row of largest quality on S, or None if no row splits S."""
    if len(S) < 2:
        return None
    best, arg = 0, None
    for r in range(n):
        q = quality(cols, S, r)
        if q > best:
            best, arg = q, r
    return arg


def path(cols, n, j=0):
    """Forks on the path of column j: list of (S_l, r_l, pattern of r_l on S_l)."""
    S = tuple(range(len(cols)))
    out = []
    while True:
        r = fork_row(cols, S, n)
        if r is None:
            return out
        pat = tuple(bit(cols[c], r) for c in S)
        out.append((S, r, pat))
        S = tuple(c for c in S if bit(cols[c], r) == bit(cols[j], r))


def sets_of(h, m, j=0):
    """S_0..S_i determined by a history h = ((rho_l, pi_l))_l."""
    S = tuple(range(m))
    out = [S]
    for _, pat in h:
        side = pat[S.index(j)]
        S = tuple(c for c, b in zip(S, pat) if b == side)
        out.append(S)
    return out


def in_A(pattern, r, h, m):
    """pattern (tuple of m bits) satisfies A_{r,i}(h)."""
    Ss = sets_of(h, m)
    for l, (rho, pat) in enumerate(h):
        S = Ss[l]
        k = sum(pattern[c] for c in S)
        q = min(k, len(S) - k)
        kl = sum(pat)
        ql = min(kl, len(S) - kl)
        if r > rho and not q <= ql:
            return False
        if r < rho and not q < ql:
            return False
    return True


# ------------------------------------------- Step 7: Gibbs exposure under U

@pytest.mark.parametrize("n,m", [(3, 3), (3, 4), (3, 5), (4, 4)])
def test_gibbs_exposure_exact(n, m):
    """Step 7, thm:gibbs and lem:nest, re-derived independently.

    Under the uniform measure on tables with distinct columns, given N_i,
    H_i = h and all rows but r (r unchosen), row r is uniform on Pi(h, alpha):
    the patterns in A_{r,i}(h) that differ on every direction-r pair.
    """
    groups = defaultdict(Counter)
    for cols in permutations(range(2 ** n), m):
        forks = path(cols, n)
        for i in range(len(forks) + 1):
            h = tuple((r, pat) for _, r, pat in forks[:i])
            used = {r for r, _ in h}
            for r in range(n):
                if r in used:
                    continue
                mask = ~(1 << r)
                alpha = tuple(c & mask for c in cols)
                row = tuple(bit(c, r) for c in cols)
                groups[(i, h, r, alpha)][row] += 1
    checked = 0
    for (i, h, r, alpha), cnt in groups.items():
        pairs = [(a, b) for a in range(m) for b in range(a + 1, m)
                 if alpha[a] == alpha[b]]
        # matching property
        flat = [c for p in pairs for c in p]
        assert len(flat) == len(set(flat))
        Pi = {pi_ for pi_ in product((0, 1), repeat=m)
              if all(pi_[a] != pi_[b] for a, b in pairs) and in_A(pi_, r, h, m)}
        assert set(cnt) == Pi
        assert len(set(cnt.values())) == 1  # uniform
        # nesting: pairs never straddle S_l, so e on S_l equals e on S_l'
        for S in sets_of(h, m):
            for a, b in pairs:
                assert (a in S) == (b in S)
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("n,m", [(3, 4), (4, 4)])
def test_rows_not_independent_under_U(n, m):
    """Why Step 1 cannot be used under U: given H_1 = h the unchosen rows are
    not independent (the joint law is not the product of its marginals)."""
    joint = defaultdict(Counter)
    for cols in permutations(range(2 ** n), m):
        forks = path(cols, n)
        if not forks:
            continue
        _, r0, pat = forks[0]
        h = ((r0, pat),)
        rest = [r for r in range(n) if r != r0]
        key = tuple(tuple(bit(c, r) for c in cols) for r in rest[:2])
        joint[h][key] += 1
    found = False
    for h, cnt in joint.items():
        tot = sum(cnt.values())
        m1, m2 = Counter(), Counter()
        for (x, y), v in cnt.items():
            m1[x] += v
            m2[y] += v
        for x in m1:
            for y in m2:
                if Fraction(cnt[(x, y)], tot) != Fraction(m1[x] * m2[y], tot * tot):
                    found = True
    assert found


# -------------------------------- Step 3 / Step 7: the subset form of the filter

def test_subset_form_of_filter_fails():
    """Step 7, first draft of lem:filter: C'_0 = {+-s/2} lies in {|e| >= d}
    and forces E[4 e^2/s | A_1] = s, which exceeds 2.5."""
    s = 10
    num = Fraction(0)
    den = Fraction(0)
    for k in (0, s):
        w = Fraction(comb(s, k), 2 ** s)
        num += w * Fraction(4 * (k - Fraction(s, 2)) ** 2, s)
        den += w
    assert num / den == s


# ----------------------------------------------- Step 3: closed-form constants

def c1_limit(b):
    eta = 4 * b / sqrt(pi)
    M = 1 / (1 - 2 * eta)
    return 4 * b / sqrt(pi) * exp(-M / 2 - 4 * b * b / 3)


def test_c1_limit_values():
    assert abs(c1_limit(0.1) - 0.0895) < 5e-4
    assert abs(c1_limit(0.05) - 0.059) < 1e-3
    best = max((c1_limit(b / 1000), b / 1000) for b in range(1, 221))
    assert abs(best[0] - 0.0905) < 5e-4 and abs(best[1] - 0.11) < 0.01


def test_hole_sup_is_erf_2b():
    """Hole mass for k = s/2, a = s/2 approaches erf(2b), above the binomial 1.6 b."""
    b, s = 0.1, 4000
    a, k = s // 2, s // 2
    # exact hypergeometric window mass
    tot = Fraction(0)
    lo, hi = a / 2 - b * sqrt(a), a / 2 + b * sqrt(a)
    den = comb(s, a)
    for x in range(int(lo) - 1, int(hi) + 2):
        if lo <= x <= hi:
            tot += Fraction(comb(k, x) * comb(s - k, a - x), den)
    assert abs(float(tot) - erf(2 * b)) < 0.03
    assert float(tot) > 1.6 * b


def variant_constants_limit(b, T, theta=0.6):
    """Prop. filter-variant in the limit sigma_1 -> infinity (Lambda_1 -> 1)."""
    ap, am = theta, 1 - theta
    bh = b / sqrt(0.99)
    A = sqrt(2 / (pi * theta * (1 - theta)))
    eta = max(2 * bh * sqrt(ap) * A, exp(-2 * max(sqrt(am) * T - b, 0) ** 2))
    M = am / (1 - eta - ap)
    ghat = lambda y: ap ** -0.5 * exp(-(2 * ap * y + 2 * b * b / 3) / am)
    c1 = 2 * b * sqrt(2 / pi) * (ghat(M / 4) - ghat(T * T) * M / (4 * T * T))
    return eta, M, c1


def test_step7_constants_from_step3():
    eta, M, c1 = variant_constants_limit(0.068, 2.5)
    assert eta < 0.4 and M < 2.5
    assert abs(c1 - 0.0372) < 1e-3
    # the closure condition eta < 0.4 holds up to b_h = 0.1585
    A = sqrt(2 / (pi * 0.24))
    assert abs(0.4 / (2 * sqrt(0.6) * A) - 0.1585) < 1e-3


# ----------------------------------------------------- Step 4: constant threshold

def sigma_needed(theta, K, beta):
    hK = (1 / K) * log2(K) + (1 - 1 / K) * log2(K / (K - 1))
    L = (1 + beta + K * hK) / (K - 1)
    x = (L + 1) * log(2) / (2 * (theta - 0.5) ** 2)
    return -int(-x // 1)


def test_cat_sigma_values():
    assert [sigma_needed(0.75, K, 2) for K in (2, 1.5, 1.25, 1.1, 1.05)] == \
        [34, 55, 93, 199, 371]


def test_cat_union_exact_small():
    """Step 4, thm:cat-union: Q(Cat) <= sum_d binom(n,d) W_d g_*^{n-d}, exactly,
    for n = 4, m = 4, theta = 0.6, sigma = 2 (W_d, g_* computed exactly)."""
    n, m, theta, sigma = 4, 4, 0.6, 2
    cat = 0
    total = 0
    for cols in product(range(2 ** n), repeat=m):
        total += 1
        if len(set(cols)) < m:
            continue
        stack = [tuple(range(m))]
        bad = False
        while stack:
            S = stack.pop()
            r = fork_row(cols, S, n)
            if r is None:
                continue
            q = quality(cols, S, r)
            if len(S) >= sigma and q < (1 - theta) * len(S):
                bad = True
            stack.append(tuple(c for c in S if bit(cols[c], r) == 0))
            stack.append(tuple(c for c in S if bit(cols[c], r) == 1))
        cat += bad
    QCat = Fraction(cat, total)

    def f(s):
        return Fraction(sum(comb(s, k) for k in range(s + 1)
                            if min(k, s - k) < (1 - theta) * s), 2 ** s)
    gstar = max(f(s) for s in range(sigma, m + 1))
    bound = Fraction(0)
    for d in range(n):
        p = Fraction(1, 2 ** d)
        tail = sum(comb(m, t) * p ** t * (1 - p) ** (m - t) for t in range(sigma, m + 1))
        bound += comb(n, d) * 2 ** d * tail * gstar ** (n - d)
    assert QCat <= bound


# ---------------------------------------------- Step 5: bias under success

def test_step5_bias_counterexample():
    """Step 5, rem:bias: m = 5, n = 3, P(|S_1| = 3 | every fork perfect) = 201/385."""
    n, m = 3, 5
    good = hit = 0
    for cols in product(range(2 ** n), repeat=m):
        forks = path(cols, n)
        S_last = forks[-1][0] if forks else None
        # path must end at a singleton with every fork perfect
        ok = bool(forks)
        for S, r, pat in forks:
            k = sum(pat)
            if min(k, len(S) - k) != len(S) // 2:
                ok = False
        if ok:
            S, r, pat = forks[-1]
            side = pat[S.index(0)]
            if sum(1 for b in pat if b == side) != 1:
                ok = False
        if ok:
            good += 1
            S0, r0, pat0 = forks[0]
            side = pat0[0]
            hit += sum(1 for b in pat0 if b == side) == 3
    assert Fraction(hit, good) == Fraction(201, 385)


# ------------------------------------------ Step 2: closed form near its edge

def _hyp_logpmf(s, k, a, x):
    from math import lgamma
    lc = lambda N, K: lgamma(N + 1) - lgamma(K + 1) - lgamma(N - K + 1)
    return lc(k, x) + lc(s - k, a - x) - lc(s, a)


@pytest.mark.parametrize("b,T", [(0.05, 1.0), (0.1, 0.5), (0.2, 1.5)])
def test_window_closed_form_at_threshold(b, T):
    """Step 2, cor:hg-window-simple, at s = s_*(b, T), worst a and e
    (floating point via lgamma; a probe, not a certificate)."""
    s = int(max(100 / b ** 2, 2000 * (1 + T) ** 4)) + 1
    for a in (int(0.4 * s) + 1, s // 2, int(0.6 * s)):
        alpha = a / s
        g = (1 - alpha) ** -0.5 * exp(-(2 * alpha * T * T + 2 * b * b / 3) / (1 - alpha))
        for k in (s // 2, s // 2 + int(T * sqrt(s)), s // 2 - int(T * sqrt(s))):
            lo, hi = a / 2 - b * sqrt(a), a / 2 + b * sqrt(a)
            mass = sum(exp(_hyp_logpmf(s, k, a, x))
                       for x in range(int(lo), int(hi) + 2) if lo <= x <= hi)
            assert mass >= 1.27 * b * g


# ------------------------------------------- Step 6: constants of Theorem C

def _h2(p):
    return 0.0 if p in (0, 1) else -p * log2(p) - (1 - p) * log2(1 - p)


def _Lambda(b, T, s, theta):
    """Step 2's Lambda(b, T, s) for the angle theta."""
    R = T + b * sqrt(theta)
    rho = 2 * R / ((1 - theta) * sqrt(s))
    u0 = b * sqrt((1 - theta) * s)
    nu = max(0.5, 1 - 1 / (2 * u0))
    return (nu / (1 + 1 / ((1 - theta) * s))
            * exp(-2 * b / (3 * (1 - theta) ** 1.5 * sqrt(s))
                  - (2 * b * b + 2 * R * R / (1 - theta)) * rho * rho / (1 - rho * rho)))


def step6_constants(b=0.05, T=1.25, s1=210):
    """Independent float evaluation of Step 3's closed forms at Step 6's choice."""
    th = 0.5 + b / sqrt(s1)
    ap, am = th, 1 - th
    A = sqrt(2 * (1 + 1 / s1) / (pi * th * (1 - th))) * exp(2 * T * T / s1 + 8 * T ** 4 / (s1 - 4 * T * T))
    eta = max((2 * b * sqrt(ap) + 1 / sqrt(s1)) * A, exp(-2 * max(sqrt(am) * T - b, 0) ** 2))
    M = am / (1 - eta - ap)
    ghat = lambda y: ap ** -0.5 * exp(-(2 * ap * y + 2 * b * b / 3) / am)
    c1 = 2 * b * sqrt(2 / pi) * _Lambda(b, T, s1, th) * (ghat(M / 4) - ghat(T * T) * M / (4 * T * T))
    # (P): Step 2's range at s1
    assert 4 * b * b * (1 - th) * s1 >= 1
    assert 2 * (T + b * sqrt(th)) < (1 - th) * sqrt(s1)
    return eta, M, c1


def test_step6_theorem_C_constants():
    eta, M, c1 = step6_constants()
    assert abs(eta - 0.25144) < 1e-4 and eta < 1 - (0.5 + 0.05 / sqrt(210))
    assert abs(M - 2.02583) < 1e-3
    assert c1 >= 0.00716 and c1 < 0.0073


def test_step6_budget_line():
    """B'(m) <= u + v log m with u <= 17.14, v <= 0.4548 (theta_4 = 3/4, sigma = 34)."""
    th, sig = 0.75, 34
    v = (1 - _h2(th)) / log2(1 / th)
    small = sum(1 - _h2(1 / s) for s in range(3, sig))
    u = small + (1 - _h2(th)) * (1 - log2(sig) / log2(1 / th))
    assert v <= 0.4548 and u <= 17.14
    for lam in range(6, 200):
        dstar = (lam - log2(sig)) / log2(1 / th)
        Bp = (1 - _h2(th)) * (int(dstar) + 1) + small
        assert Bp <= u + v * lam + 1e-9


def test_step6_p_minimum_float():
    """Step 5's exact minimum p at Step 6's choice (b = 1/20, T_5^2 <= 1.012913, s_1 = 210),
    in floating point: at least 0.010129."""
    from math import lgamma
    b, s1, T5 = 0.05, 210, sqrt(1.012913)
    lc = lambda N, K: lgamma(N + 1) - lgamma(K + 1) - lgamma(N - K + 1)
    best = 1.0
    N = s1 + 1
    while N / 2 - b * sqrt(N) <= s1:
        kmin = int(N / 2 - T5 * sqrt(N)) - 1
        for k in range(max(kmin, 0), N + 1):
            if abs(k - N / 2) > T5 * sqrt(N):
                if k > N / 2:
                    break
                continue
            for u in range(2, int(min(s1, N / 2 + b * sqrt(N))) + 1):
                xs = {u // 2, (u + 1) // 2}
                pr = 0.0
                for x in xs:
                    if 0 <= x <= k and 0 <= u - x <= N - k:
                        pr += exp(lc(k, x) + lc(N - k, u - x) - lc(N, u))
                best = min(best, pr)
        N += 1
    assert best >= 0.010129 and best < 0.0102
