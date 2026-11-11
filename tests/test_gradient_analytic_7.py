# SPDX-License-Identifier: GPL-2.0-or-later
"""Checks for Step 7 of the analytic gradient-key plan (n below 2 log m).

See articles/gradient-bound/analytic/07-below-2log.tex. Exact enumeration over
all n x m matrices with distinct columns (the uniform measure U) for tiny n, m;
these are finite checks of the statements, not part of their proofs.
"""

from collections import defaultdict
from fractions import Fraction
from itertools import permutations, product
from math import comb, exp, floor, log2, lgamma, log, pi, sqrt

import pytest

SIZES = [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4)]


def bit(col, r):
    return (col >> r) & 1


def quality(cols, S, r):
    k = sum(bit(cols[c], r) for c in S)
    return min(k, len(S) - k)


def fork_row(cols, S, n):
    """Row chosen by G at S (first of largest quality), or None."""
    if len(S) < 2:
        return None
    best, arg = 0, None
    for r in range(n):
        q = quality(cols, S, r)
        if q > best:
            best, arg = q, r
    return arg


def path(cols, n, j=0):
    """Forks on the path of column j: list of (S_l, r_l, pattern on S_l)."""
    S = tuple(range(len(cols)))
    out = []
    while True:
        r = fork_row(cols, S, n)
        if r is None:
            return out, S
        pat = tuple(bit(cols[c], r) for c in S)
        out.append((S, r, pat))
        S = tuple(c for c in S if bit(cols[c], r) == bit(cols[j], r))


def tree_forks(cols, n):
    """All forks of G with their depth: list of (depth, S)."""
    res, stack = [], [(0, tuple(range(len(cols))))]
    while stack:
        d, S = stack.pop()
        r = fork_row(cols, S, n)
        if r is None:
            res.append((d, S))  # leaf, kept for the level partition
            continue
        res.append((d, S))
        for b in (0, 1):
            stack.append((d + 1, tuple(c for c in S if bit(cols[c], r) == b)))
    return res


def tables(n, m):
    return permutations(range(1 << n), m)


def in_A(rowbits, hist, r):
    """A_{r,i}(h): row r (tuple of bits by column) against history hist."""
    for S, rl, pat in hist:
        k = sum(rowbits[c] for c in S)
        q = min(k, len(S) - k)
        ql = min(sum(pat), len(S) - sum(pat))
        if r > rl and not q <= ql:
            return False
        if r < rl and not q < ql:
            return False
    return True


def mask(cols, rows):
    m = 0
    for r in rows:
        m |= 1 << r
    return tuple(c & ~m for c in cols)


def pairs_direction(cols, r):
    classes = defaultdict(list)
    for c, v in enumerate(mask(cols, [r])):
        classes[v].append(c)
    return [tuple(v) for v in classes.values() if len(v) >= 2]


@pytest.mark.parametrize("n,m", SIZES)
def test_gibbs_exposure_and_nesting(n, m):
    """Lemma 2.1 and 2.2: given H_i=h and all rows but r, row r is uniform on the
    patterns that are antisymmetric on the direction-r pairs and satisfy A_{r,i}(h);
    pairs are never separated by a set on the path."""
    groups = defaultdict(set)
    meta = {}
    for cols in tables(n, m):
        forks, _ = path(cols, n)
        for i in range(len(forks) + 1):
            hist = forks[:i]
            chosen = {rl for _, rl, _ in hist}
            h = tuple((rl, S, pat) for S, rl, pat in hist)
            for r in range(n):
                if r in chosen:
                    continue
                key = (i, h, r, mask(cols, [r]))
                groups[key].add(tuple(bit(c, r) for c in cols))
                prs = pairs_direction(cols, r)
                assert all(len(p) == 2 for p in prs)
                for p in prs:  # nesting: both or neither in each S_l
                    for S, _, _ in hist:
                        assert (p[0] in S) == (p[1] in S)
                meta[key] = (hist, prs)
    for key, seen in groups.items():
        i, h, r, _ = key
        hist, prs = meta[key]
        want = set()
        for bits in product((0, 1), repeat=m):
            if any(bits[a] == bits[b] for a, b in prs):
                continue
            if in_A(bits, hist, r):
                want.add(bits)
        assert seen == want


@pytest.mark.parametrize("n,m", SIZES)
def test_gibbs_product_bound(n, m):
    """Lemma 2.4: P(all rows of R' fail | h, rows outside R') is at most the product of
    the largest conditional failure probabilities given everything but one row.
    Failure: no perfect split of S_i (quality < floor(s_i/2))."""
    # atoms keyed by (i, h); inside, by the full table
    by_h = defaultdict(list)
    for cols in tables(n, m):
        forks, _ = path(cols, n)
        for i in range(len(forks) + 1):
            hist = forks[:i]
            S = forks[i][0] if i < len(forks) else None
            if S is None:
                continue
            h = tuple((rl, SS, pat) for SS, rl, pat in hist)
            by_h[(i, h, S)].append(cols)
    checked = 0
    for (i, h, S), tabs in by_h.items():
        chosen = {rl for rl, _, _ in h}
        free = [r for r in range(n) if r not in chosen]
        kappa = len(S) // 2
        fail = {r: (lambda cols, r=r: quality(cols, S, r) < kappa) for r in free}
        for size in (1, 2, len(free)):
            if size > len(free) or size == 0:
                continue
            Rp = free[-size:]
            # conditional failure of one row given all others
            single = {}
            for r in Rp:
                cnt = defaultdict(lambda: [0, 0])
                for cols in tabs:
                    c = cnt[mask(cols, [r])]
                    c[0] += fail[r](cols)
                    c[1] += 1
                single[r] = cnt
            atoms = defaultdict(list)
            for cols in tabs:
                atoms[mask(cols, Rp)].append(cols)
            for key, ts in atoms.items():
                both = sum(all(fail[r](c) for r in Rp) for c in ts)
                bound = Fraction(1)
                for r in Rp:
                    worst = max(Fraction(single[r][mask(c, [r])][0],
                                         single[r][mask(c, [r])][1]) for c in ts)
                    bound *= worst
                assert Fraction(both, len(ts)) <= bound
                checked += 1
    assert checked > 0


def hamming_bad(cols, w):
    bad = set()
    for a in range(len(cols)):
        for b in range(len(cols)):
            if a != b and bin(cols[a] ^ cols[b]).count("1") <= w:
                bad.add(a)
    return bad


@pytest.mark.parametrize("n,m", SIZES)
def test_pollution_averaging(n, m):
    """Lemma 3.1: at each depth, polluted forks hold at most |Bad|/eta columns; and
    U(column 0 is bad) <= (m-1)(V-1)/(2^n-1)."""
    for w in (1, 2):
        if w >= n:
            continue
        V = sum(comb(n, t) for t in range(w + 1))
        nbad, total = 0, 0
        for cols in tables(n, m):
            bad = hamming_bad(cols, w)
            nbad += 0 in bad
            total += 1
            for eta in (Fraction(1, 2), Fraction(1, 3)):
                lev = defaultdict(int)
                for d, S in tree_forks(cols, n):
                    if len(bad.intersection(S)) > eta * len(S):
                        lev[d] += len(S)
                assert all(v <= len(bad) / eta for v in lev.values())
        assert Fraction(nbad, total) <= Fraction((m - 1) * (V - 1), (1 << n) - 1)


def test_ball_volume():
    """Corollary 4.3: V(n, floor(Delta/(2 log(e n)))) <= 2^(Delta/2)."""
    for n in range(2, 400):
        for delta in range(1, n + 1):
            w = floor(delta / (2 * log2(exp(1) * n)))
            V = sum(comb(n, t) for t in range(w + 1))
            assert V <= 2 ** (delta / 2)


def filter_run(s0, b, levels):
    """Exact forward filter of Lemma 3.2 (floats): worst thresholds 2b sqrt(s'),
    non-strict constraint |e| >= delta; returns max N_l, min P(good at last)."""
    from math import comb as C

    def hyp(s, k, a):
        tot = C(s, a)
        return [C(k, x) * C(s - k, a - x) / tot for x in range(a + 1)]

    s = s0
    mu = [C(s, k) / 2 ** s for k in range(s + 1)]
    maxN = 0.0
    for _ in range(levels):
        N = sum(p * 4 * (k - s / 2) ** 2 / s for k, p in enumerate(mu))
        maxN = max(maxN, N)
        delta = 2 * b * sqrt(s)
        nu = [p if abs(k - s / 2) >= delta else 0.0 for k, p in enumerate(mu)]
        z = sum(nu)
        nu = [p / z for p in nu]
        a = s // 2
        new = [0.0] * (a + 1)
        for k, p in enumerate(nu):
            if p:
                for x, q in enumerate(hyp(s, k, a)):
                    new[x] += p * q
        mu, s = new, a
    N = sum(p * 4 * (k - s / 2) ** 2 / s for k, p in enumerate(mu))
    good = sum(p for k, p in enumerate(mu) if abs(k - s / 2) <= b * sqrt(s))
    return max(maxN, N), good


def test_filter_numeric():
    """A finite numeric instance of Lemma 3.2 (not a proof): s'_0 = 512, halving."""
    maxN, good = filter_run(512, 0.04, 3)
    assert maxN <= 2
    assert good > 0.02


def test_poisson_loss():
    """Remark 5.2: U(|U| = m) for Bernoulli(m/N) cells is about (2 pi m (1-p))^(-1/2)."""
    for n, m in [(20, 2 ** 12), (24, 2 ** 14), (30, 2 ** 16)]:
        N = 2 ** n
        p = m / N
        lp = (lgamma(N + 1) - lgamma(m + 1) - lgamma(N - m + 1)
              + m * log(p) + (N - m) * log(1 - p))
        ratio = exp(lp) * sqrt(2 * pi * m * (1 - p))
        assert abs(ratio - 1) < 0.01


def test_filter_variant_constants():
    """Section 3.2: with b = 0.068, T = 2, sigma_1 = 0.99 s_1, b_h = b/sqrt(0.99), the
    constants of Step 3, Prop. filter-variant, give M* <= 2.5 and c_1 > 0.03 for
    10^5 <= s_1 <= 10^12 (floating point, not interval-certified)."""
    import keytrees.gradient_analytic_3 as g3
    b, T = 0.068, 2.0
    bh = b / sqrt(0.99)
    assert bh < 0.1585
    s1 = 1e5
    while s1 <= 1e12:
        sig = 0.99 * s1
        assert 4 * b * b * 0.4 * sig >= 1 and sqrt(sig) > 2 * (T + b * sqrt(0.6)) / 0.4
        assert T <= sqrt(sig / 8)
        M = g3.m_star_theta(bh, sig, T, 0.6)
        assert M <= 2.5
        assert g3.c1_bound_theta(b, sig, T, M, 0.6) > 0.03
        s1 *= 1.5
