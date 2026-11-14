"""Exact enumeration checks of articles/gradient-bound/analytic/01-exposure.tex.

All matrices (including those with equal columns) are enumerated, in rational
arithmetic, so nothing here is sampled. A matrix is a tuple of n row masks; bit c
of a mask is the entry in column c. Rows are indexed from 0.
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
from math import comb

import pytest

import keytrees.gradient_bound as gb

CASES = [(2, 3), (2, 4), (3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (5, 3)]


def weight(mask):
    return mask.bit_count()


def quality(row, S):
    k = weight(row & S)
    return min(k, weight(S) - k)


def path(rows, m, j):
    """Forks on the path of column j: list of (S, chosen row, pattern), and the final set."""
    S = (1 << m) - 1
    forks = []
    while True:
        if weight(S) < 2:
            return forks, S
        qs = [quality(r, S) for r in rows]
        best = max(qs)
        if best == 0:
            return forks, S
        rho = qs.index(best)
        pat = rows[rho] & S
        forks.append((S, rho, pat))
        S = pat if pat >> j & 1 else S & ~pat


def distinct(rows, m):
    cols = {tuple(r >> c & 1 for r in rows) for c in range(m)}
    return len(cols) == m


@lru_cache(None)
def histories(n, m, j):
    """Map each history h (tuple of (rho, pattern)) to the list of matrices with H_i = h."""
    table = defaultdict(list)
    for rows in itertools.product(range(1 << m), repeat=n):
        forks, _ = path(rows, m, j)
        h = tuple((rho, pat) for _, rho, pat in forks)
        for i in range(len(h) + 1):
            table[h[:i]].append(rows)
    return dict(table)


def sets_of(h, m, j):
    S = (1 << m) - 1
    out = [S]
    for _, pat in h:
        S = pat if pat >> j & 1 else S & ~pat
        out.append(S)
    return out


def in_A(row, r, h, sets, t):
    """Whether a row with index r satisfies A_{r,t}(h)."""
    for l in range(t):
        rho, pat = h[l]
        S = sets[l]
        q = min(weight(pat), weight(S) - weight(pat))
        if r > rho and not quality(row, S) <= q:
            return False
        if r < rho and not quality(row, S) < q:
            return False
    return True


def row_event(r, h, sets, m):
    """The set of row masks allowed by the product form (eq:product) for row index r."""
    chosen = {rho: l for l, (rho, _) in enumerate(h)}
    allowed = []
    for row in range(1 << m):
        if r in chosen:
            l = chosen[r]
            if row & sets[l] == h[l][1] and in_A(row, r, h, sets, l):
                allowed.append(row)
        elif in_A(row, r, h, sets, len(h)):
            allowed.append(row)
    return allowed


def hyp(pop, marked, draws, x):
    return Fraction(comb(marked, x) * comb(pop - marked, draws - x), comb(pop, draws))


@pytest.mark.parametrize("n,m", CASES)
@pytest.mark.parametrize("j", [0, 1])
def test_product_form_and_independence(n, m, j):
    """eq:product: the matrices with H_i = h are exactly the product of the row events."""
    for h, mats in histories(n, m, j).items():
        sets = sets_of(h, m, j)
        events = [row_event(r, h, sets, m) for r in range(n)]
        assert sorted(mats) == sorted(itertools.product(*events))


@pytest.mark.parametrize("n,m", CASES)
def test_inadmissible_histories_absent(n, m):
    for h in histories(n, m, 0):
        rhos = [rho for rho, _ in h]
        assert len(set(rhos)) == len(rhos) and len(h) <= n
        sets = sets_of(h, m, 0)
        for l, (_, pat) in enumerate(h):
            assert 0 < weight(pat) < weight(sets[l])


@pytest.mark.parametrize("n,m", CASES)
def test_failure_product(n, m):
    """cor:product, including stops at non-singletons (kappa = 1)."""
    for h, mats in histories(n, m, 0).items():
        sets = sets_of(h, m, 0)
        S, i = sets[-1], len(h)
        if weight(S) < 2:
            continue
        unchosen = [r for r in range(n) if r not in {rho for rho, _ in h}]
        assert len(unchosen) == n - i
        for kappa in range(1, weight(S) // 2 + 1):
            direct = Fraction(sum(all(quality(row, S) < kappa for row in rows) for rows in mats), len(mats))
            prod = Fraction(1)
            for r in unchosen:
                ev = row_event(r, h, sets, m)
                prod *= Fraction(sum(quality(row, S) < kappa for row in ev), len(ev))
            assert direct == prod
        stops = sum(path(rows, m, 0)[1] == S and len(path(rows, m, 0)[0]) == i for rows in mats)
        const = sum(all(quality(row, S) == 0 for row in rows) for rows in mats)
        assert stops == const


@pytest.mark.parametrize("m", [3, 4, 5, 6])
def test_markov_chain_on_fixed_sets(m):
    """lem:markov: the joint law of the counts on every nested chain of three sets."""
    full = (1 << m) - 1
    subsets = list(range(1, full + 1))
    for T1 in subsets:
        if T1 == full:
            continue
        for T2 in subsets:
            if T2 & ~T1 or T2 == T1:
                continue
            law = Counter((weight(y), weight(y & T1), weight(y & T2)) for y in range(1 << m))
            t1, t2 = weight(T1), weight(T2)
            for (a0, a1, a2), c in law.items():
                model = Fraction(comb(m, a0), 1 << m) * hyp(m, a0, t1, a1) * hyp(t1, a1, t2, a2)
                assert Fraction(c, 1 << m) == model


@pytest.mark.parametrize("n,m", CASES)
def test_forward_filter(n, m):
    """cor:filter: the recursion, the uniform pattern given the count, and the density bound."""
    for h in histories(n, m, 0):
        sets = sets_of(h, m, 0)
        i = len(h)
        chosen = {rho for rho, _ in h}
        for r in set(range(n)) - chosen:
            mu = {k: Fraction(comb(m, k), 1 << m) for k in range(m + 1)}
            for l in range(i):
                rho, pat = h[l]
                s, s2 = weight(sets[l]), weight(sets[l + 1])
                q = min(weight(pat), s - weight(pat))
                C = [k for k in range(s + 1) if (min(k, s - k) <= q if r > rho else min(k, s - k) < q)]
                mass = sum(mu.get(k, 0) for k in C)
                assert mass > 0
                new = defaultdict(Fraction)
                for k in C:
                    for x in range(max(0, s2 - (s - k)), min(k, s2) + 1):
                        new[x] += mu.get(k, 0) * hyp(s, k, s2, x) / mass
                mu = dict(new)
            ev = [row for row in range(1 << m) if in_A(row, r, h, sets, i)]
            S = sets[-1]
            direct = Counter(weight(row & S) for row in ev)
            assert {k: v for k, v in mu.items() if v} == {k: Fraction(c, len(ev)) for k, c in direct.items()}
            patterns = Counter(row & S for row in ev)
            for pat, c in patterns.items():
                assert Fraction(c, len(ev)) == mu[weight(pat)] / comb(weight(S), weight(pat))
            assert len(patterns) == sum(comb(weight(S), k) for k, v in mu.items() if v)


@pytest.mark.parametrize("n,m", CASES)
@pytest.mark.parametrize("j", [0, 2])
def test_side_of_column(n, m, j):
    """thm:side: eq:choice, the uniform pattern, k/s, and the law of the next size."""
    for h, mats in histories(n, m, j).items():
        sets = sets_of(h, m, j)
        S, i = sets[-1], len(h)
        s = weight(S)
        if s < 2:
            continue
        unchosen = [r for r in range(n) if r not in {rho for rho, _ in h}]
        events = {r: row_event(r, h, sets, m) for r in unchosen}
        outcome = defaultdict(list)
        byq = defaultdict(Counter)
        for rows in mats:
            forks, _ = path(rows, m, j)
            if len(forks) > i:
                _, rho, pat = forks[i]
                k = weight(pat)
                outcome[rho, k].append(pat)
                side = k if pat >> j & 1 else s - k
                byq[rho, min(k, s - k)][side] += 1
        for rho in unchosen:
            for k in range(1, s):
                q = min(k, s - k)
                model = Fraction(sum(weight(y & S) == k for y in events[rho]), len(events[rho]))
                for r in unchosen:
                    if r != rho:
                        good = sum(quality(y, S) < q if r < rho else quality(y, S) <= q for y in events[r])
                        model *= Fraction(good, len(events[r]))
                assert Fraction(len(outcome.get((rho, k), [])), len(mats)) == model
        for (rho, k), pats in [(key, v) for key, v in outcome.items() if v]:
            count = Counter(pats)
            assert len(count) == comb(s, k) and len(set(count.values())) == 1
            assert Fraction(sum(p >> j & 1 for p in pats), len(pats)) == Fraction(k, s)
        for (rho, q), sides in byq.items():
            total = sum(sides.values())
            if 2 * q == s:
                assert set(sides) == {q}
            else:
                assert Fraction(sides[q], total) == Fraction(q, s)
                assert Fraction(sides[s - q], total) == Fraction(s - q, s)


@pytest.mark.parametrize("n,m", [c for c in CASES if c[0] <= 4 and c[1] <= 2**c[0]])
def test_reduction_to_one_column(n, m):
    """lem:transfer in depth form: the mean depth of a fixed column on D is t_{n,m}(G)."""
    exact = gb.exact_average(n, m)
    for j in range(m):
        total, count = 0, 0
        for rows in itertools.product(range(1 << m), repeat=n):
            if distinct(rows, m):
                forks, last = path(rows, m, j)
                assert weight(last) == 1
                total += len(forks)
                count += 1
        assert Fraction(total, count) == exact


@pytest.mark.parametrize("n,m", [(3, 4), (4, 4), (3, 5)])
def test_transfer_entropy_form(n, m):
    """lem:transfer: t - log m = E[Sigma_j 1_D]/Q(D), in floating point for the entropy terms."""
    total, count = 0.0, 0
    for rows in itertools.product(range(1 << m), repeat=n):
        if distinct(rows, m):
            forks, _ = path(rows, m, 0)
            total += sum(1 - gb._h(weight(p) / weight(S)) for S, _, p in forks)
            count += 1
    assert total / count == pytest.approx(float(gb.exact_average(n, m)) - math.log2(m), abs=1e-10)
