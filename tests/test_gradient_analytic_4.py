"""Checks of articles/gradient-bound/analytic/04-catastrophe.tex.

Exact enumeration on tiny tables, and grids for the inequalities. A grid is a
check of the algebra, not a proof; the proofs are in the step file.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction as F

import pytest

LOG2E = 1 / math.log(2)


def h(x):
    return 0.0 if x in (0, 1) else -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def f_exact(theta, s):
    """Probability that a uniform row has quality < (1 - theta) s on s columns."""
    return F(sum(math.comb(s, k) for k in range(s + 1)
                 if min(k, s - k) < (1 - theta) * s), 2**s)


def w_exact(m, d, sigma):
    """Exact 2^d P(Bin(m, 2^-d) >= sigma)."""
    p = F(1, 2**d)
    return 2**d * sum(math.comb(m, j) * p**j * (1 - p)**(m - j) for j in range(sigma, m + 1))


def d_top(m, theta, sigma):
    """Largest d with theta^d m >= sigma (sigma <= m)."""
    d = 0
    while theta ** (d + 1) * m >= sigma:
        d += 1
    return d


def forks(cols, n):
    """Yield (path forks) of G: for each column the list of (s, k); and all forks."""
    per_col = {c: [] for c in cols}
    every = []

    def go(S, depth):
        if len(S) == 1:
            return
        best, row = -1, None
        for r in range(n):
            k = sum((c >> r) & 1 for c in S)
            q = min(k, len(S) - k)
            if q > best:
                best, row = q, r
        ones = [c for c in S if (c >> row) & 1]
        zeros = [c for c in S if not (c >> row) & 1]
        assert ones and zeros
        every.append((len(S), best, depth))
        for c in S:
            per_col[c].append((len(S), len(ones)))
        go(ones, depth + 1)
        go(zeros, depth + 1)

    go(list(cols), 0)
    return per_col, every


def budget_sharp(m, theta, sigma):
    th = float(theta)
    big = d_top(m, theta, sigma) + 1 if sigma <= m else 0
    return (1 - h(th)) * big + sum(1 - h(1 / s) for s in range(3, min(sigma - 1, m) + 1))


def budget(m, theta, s_c):
    th = float(theta)
    d = max(0.0, math.log2(m / s_c) / math.log2(1 / th))
    return d + s_c


CELLS = [(3, m) for m in range(2, 9)] + [(4, m) for m in range(2, 11)] + [(5, m) for m in range(2, 6)]
THETAS = [F(3, 5), F(3, 4), F(9, 10)]


@pytest.mark.parametrize("n,m", CELLS)
def test_exact_enumeration(n, m):
    """Q(Cat) against the union bound, and the per-path budget, on every table."""
    tables = list(itertools.combinations(range(2**n), m))
    trees = [forks(t, n) for t in tables]
    weight = F(math.factorial(m), 2 ** (n * m))
    for theta in THETAS:
        for sigma in range(2, m + 2):
            cat = 0
            for per_col, every in trees:
                is_cat = any(s >= sigma and q < (1 - theta) * s for s, q, _ in every)
                cat += is_cat
                assert all(d < n for s, _, d in every if s >= 2)
                if is_cat:
                    continue
                assert all(d <= d_top(m, theta, sigma) for s, _, d in every if s >= sigma)
                bound = min(m - 1, budget_sharp(m, theta, sigma))
                assert bound <= max(m - 1, budget(m, theta, sigma)) + 1e-12
                for path in per_col.values():
                    total = sum(1 - h(k / s) for s, k in path)
                    assert total <= bound + 1e-9
                    assert total <= budget(m, theta, sigma) + 1e-9
            q_cat = cat * weight
            if sigma > m:
                assert cat == 0
                continue
            g = max(f_exact(theta, s) for s in range(sigma, m + 1))
            top = min(d_top(m, theta, sigma), n - 1)
            union = sum(math.comb(n, d) * w_exact(m, d, sigma) * g ** (n - d) for d in range(top + 1))
            assert q_cat <= union


def test_enumeration_is_not_vacuous():
    """At least one cell has a catastrophe and a union bound below 1."""
    n, m, theta, sigma = 4, 8, F(3, 5), 8
    tables = itertools.combinations(range(2**n), m)
    cat = sum(any(s >= sigma and q < (1 - theta) * s for s, q, _ in forks(t, n)[1]) for t in tables)
    union = sum(math.comb(n, d) * w_exact(m, d, sigma) * f_exact(theta, 8) ** (n - d)
                for d in range(min(d_top(m, theta, sigma), n - 1) + 1))
    assert cat > 0
    assert cat * F(math.factorial(m), 2 ** (n * m)) <= union < 1


@pytest.mark.parametrize("theta", [F(51, 100), F(11, 20), F(3, 5), F(2, 3), F(3, 4), F(9, 10), F(99, 100)])
def test_hoeffding(theta):
    t = float(theta) - 0.5
    for s in range(1, 500):
        assert float(f_exact(theta, s)) <= 2 * math.exp(-2 * t * t * s) * (1 + 1e-12)


def test_w_bounds():
    for m in range(2, 40):
        for sigma in range(2, m + 1):
            for d in range(0, 14):
                w = w_exact(m, d, sigma)
                assert w <= 2**d
                assert w * sigma <= m
                assert w * math.factorial(sigma) * 2 ** (d * (sigma - 1)) <= m**sigma


# Tables of log2 factorials for the sums in logs.
LF = [0.0]
for i in range(1, 40001):
    LF.append(LF[-1] + math.log2(i))


def lchoose(n, d):
    return LF[n] - LF[d] - LF[n - d]


def lsum(xs):
    top = max(xs)
    return top + math.log2(sum(2 ** (x - top) for x in xs))


def lgfact(s):
    return math.lgamma(s + 1) * LOG2E


def closed_log(lam, n, L, sigma, u):
    return lam - math.log2(sigma) - (n - lam) * L - lam * math.log2(u) + n * math.log2(1 + u)


def union_log(lam, n, L, sigma):
    """log2 of sum_{d<n} C(n,d) min(2^d, m/sigma, m^sigma 2^{-d(sigma-1)}/sigma!) g^{n-d}."""
    terms = []
    for d in range(n):
        w = min(d, lam - math.log2(sigma), sigma * lam - d * (sigma - 1) - lgfact(sigma))
        terms.append(lchoose(n, d) + w - (n - d) * L)
    return lsum(terms)


def sigma_of(theta, K, beta):
    L = (1 + beta + K * h(1 / K)) / (K - 1)
    return math.ceil((L + 1) * math.log(2) / (2 * (theta - 0.5) ** 2)), L


@pytest.mark.parametrize("K", [1.05, 1.1, 1.25, 1.5, 1.75, 2.0])
@pytest.mark.parametrize("beta", [1.0, 2.0, 5.0])
@pytest.mark.parametrize("theta", [0.55, 0.6, 0.65, 0.75, 0.9, 0.99])
def test_constant_threshold(K, beta, theta):
    sigma, LK = sigma_of(theta, K, beta)
    Lg = 2 * (theta - 0.5) ** 2 * sigma * LOG2E - 1
    assert Lg >= LK - 1e-9
    u = 1 / (K - 1)
    assert 2 ** (1 - sigma) <= 2 ** (-Lg) * u <= 1
    assert Lg > math.log2(1 + u)
    for lam in [math.log2(sigma), 8.0, 13.7, 20.0, 40.0, 100.0, 400.0]:
        n0 = math.ceil(K * lam)
        for n in {n0, n0 + 1, 2 * n0, 5 * n0, 20 * n0, 10000}:
            if n < K * lam or n > 40000:
                continue
            c = closed_log(lam, n, Lg, sigma, u)
            assert c <= -beta * lam - math.log2(sigma) + 1e-6
            assert union_log(lam, n, Lg, sigma) <= c + 1e-6


def test_constants():
    assert 1 / math.log2(1 / 0.6) < 1.357
    assert 1 / math.log2(1 / 0.6) > 1.3569
    a = 0.02 * LOG2E
    assert 82 * a >= 2.357
    assert 47 * a < 1.3569
    assert 1 / math.sqrt(2) == pytest.approx(2 ** (-1 / 2))
    assert sigma_of(0.75, 2, 2) == (34, 5.0)
    assert sigma_of(0.6, 2, 2) == (208, 5.0)


@pytest.mark.parametrize("eps", [0.01, 0.1, 0.3, 0.643, 1.0])
def test_contract_form(eps):
    C = 82 / eps
    for lam in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0, 1000.0, 5000.0]:
        n0 = max(4, math.ceil((1.357 + eps) * lam))
        for n in {n0, n0 + 1, 3 * n0, 100 * n0}:
            s_c = C * math.log2(2 * n)
            if math.log2(s_c) > lam:
                continue
            d_star = (lam - math.log2(s_c)) / math.log2(1 / 0.6)
            L = 2 * 0.01 * s_c * LOG2E - 1
            assert L > 0
            top = math.floor(d_star)
            assert top < n
            xs = [d * math.log2(2 * n) - (n - d) * L for d in range(top + 1)]
            assert lsum(xs) <= 1 + xs[-1] + 1e-9
            assert lsum(xs) <= 1 - 2 * lam + 1e-9


@pytest.mark.parametrize("eps,C", [(0.1, 400.0), (0.3, 150.0), (0.643, 70.0)])
def test_contract_needs_c_of_order_one_over_eps(eps, C):
    """Below C (eps + 1e-4) = 47 the displayed sum has a huge top term."""
    assert C * (eps + 1e-4) < 47
    lam = 1e9
    n = math.ceil((1.357 + eps) * lam)
    Lam = math.log2(2 * n)
    s_c = C * Lam
    d = math.floor((lam - math.log2(s_c)) / math.log2(1 / 0.6))
    log_term = d * Lam - (n - d) * (2 * 0.01 * s_c * LOG2E - 1)
    assert log_term > 0


def limit_c(theta, K=2.0):
    delta = 1 / math.log2(1 / theta)
    return delta / (2 * (theta - 0.5) ** 2 * LOG2E * (K - delta))


def test_theta_in_contract_form():
    grid = [t / 1000 for t in range(510, 700)]
    best = min(grid, key=limit_c)
    assert best == pytest.approx(0.636, abs=0.002)
    assert 61.2 < limit_c(best) < 61.4
    assert 73.0 < limit_c(0.6) < 73.2
    assert 1 / math.log2(1 / 0.7071) < 2 < 1 / math.log2(1 / 0.7072)


TABLE = {  # theta: (sigma, B at 2^10, 2^20, 2^40, sharp B' at 2^10, 2^20, 2^40)
    0.55: (832, 832.3, 843.9, 867.1, 788.9, 789.0, 789.2),
    0.60: (208, 211.1, 224.7, 251.8, 179.1, 179.5, 180.3),
    0.65: (93, 98.6, 114.7, 146.8, 71.3, 72.4, 74.5),
    0.70: (52, 60.4, 79.8, 118.7, 35.4, 37.7, 42.3),
    0.75: (34, 45.8, 69.9, 118.1, 21.5, 26.1, 35.3),
    0.80: (24, 40.8, 71.9, 134.0, 16.2, 24.8, 42.3),
    0.90: (13, 54.4, 120.2, 251.8, 26.2, 61.3, 130.8),
}


def sharp_float(lam, theta, sigma):
    d = (lam - math.log2(sigma)) / math.log2(1 / theta)
    return (1 - h(theta)) * (math.floor(d) + 1) + sum(1 - h(1 / s) for s in range(3, sigma))


@pytest.mark.parametrize("theta", sorted(TABLE))
def test_theta_table(theta):
    sigma, *vals = TABLE[theta]
    assert sigma_of(theta, 2, 2)[0] == sigma
    for lam, v in zip((10, 20, 40), vals[:3]):
        assert round((lam - math.log2(sigma)) / math.log2(1 / theta) + sigma, 1) == pytest.approx(v)
    for lam, v in zip((10, 20, 40), vals[3:]):
        assert round(sharp_float(lam, theta, sigma), 1) == pytest.approx(v)


@pytest.mark.parametrize("K,sig60,sig75", [(1.05, 2316, 371), (1.1, 1242, 199), (1.25, 576, 93),
                                           (1.5, 339, 55), (2.0, 208, 34)])
def test_sigma_against_k(K, sig60, sig75):
    assert sigma_of(0.6, K, 2)[0] == sig60
    assert sigma_of(0.75, K, 2)[0] == sig75


def test_two_power_exceeds_z():
    """2^{-1/y} <= y on (0, 1], used for g u <= 1."""
    for i in range(1, 2001):
        y = i / 2000
        assert 2 ** (-1 / y) <= y


def test_transfer_at_two_log_m():
    """Q(D) >= 1/2 at n = ceil(2 log m), exactly, and below 2 log m it collapses."""
    for m in [2, 3, 5, 16, 100, 1000, 4096]:
        n = math.ceil(2 * math.log2(m))
        q = F(1)
        for i in range(m):
            q *= 1 - F(i, 2**n)
        assert q >= F(1, 2)
