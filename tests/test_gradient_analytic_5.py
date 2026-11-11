"""Exact checks of articles/gradient-bound/analytic/05-small-forks.tex.

Everything here is brute force over all n x m bit matrices or exact arithmetic
on hypergeometric weights. Expectations of sum T_i are compared as vectors of
rational coefficients of the symbols T(s) = 1 - h(floor(s/2)/s), so equalities
are exact and do not depend on floating point.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from fractions import Fraction
from functools import cache

import pytest

GAMMA = 2 - 1 / math.log(2) - math.log2(2 * math.log(2))
POP = [bin(x).count("1") for x in range(1 << 8)]


def quality(row, S):
    k = POP[row & S]
    return min(k, POP[S] - k)


def perfect_on(row, U):
    return quality(row, U) == POP[U] // 2


def walk(rows, m):
    """Forks (r, S, pattern) of G on the path of column 0; last one may be dead."""
    S = (1 << m) - 1
    forks = []
    while POP[S] > 1:
        best, r = -1, -1
        for index, row in enumerate(rows):
            q = quality(row, S)
            if q > best:
                best, r = q, index
        forks.append((r, S, rows[r] & S, best))
        if best == 0:
            break
        S = S & rows[r] if rows[r] & 1 else S & ~rows[r]
    return forks, S


def is_perfect(fork):
    return fork[3] == POP[fork[1]] // 2


@cache
def f_vector(s):
    """f(s) as coefficients of T(.), by the recursion of the walk."""
    if s == 1:
        return {}
    lo, hi = s // 2, s - s // 2
    out = Counter({s: Fraction(1)})
    for size in (lo, hi):
        for key, value in f_vector(size).items():
            out[key] += Fraction(size, s) * value
    return dict(out)


def add(target, vector, scale=1):
    for key, value in vector.items():
        target[key] += scale * value


def subsets(S):
    """All submasks of S with at least two elements."""
    sub = S
    while sub:
        if POP[sub] >= 2:
            yield sub
        sub = (sub - 1) & S


def in_A(pattern, r, prefix):
    """A_{r,i}: prefix is a list of forks (r_l, S_l, _, q_l)."""
    for r_l, S_l, _, q_l in prefix:
        q = quality(pattern, S_l)
        if (r > r_l and q > q_l) or (r < r_l and q >= q_l):
            return False
    return True


def blocks_of(R, count):
    """Consecutive blocks of near-equal size, each of size >= floor(|R|/count)."""
    size, extra = divmod(len(R), count)
    out, start = [], 0
    for g in range(count):
        stop = start + size + (1 if g < extra else 0)
        out.append(R[start:stop])
        start = stop
    return out


@cache
def small_stage(n, m, s1, b):
    """Group all matrices by the history at the first small fork, good histories only."""
    groups = defaultdict(list)
    for rows in itertools.product(range(1 << m), repeat=n):
        forks, last = walk(rows, m)
        i0, good = 0, True
        for fork in forks:
            s = POP[fork[1]]
            if s <= s1:
                break
            if s / 2 - fork[3] > b * math.sqrt(s):
                good = False
                break
            i0 += 1
        if not good or (i0 == len(forks) and POP[last] > s1):
            continue
        key = tuple((f[0], f[1], f[2], f[3]) for f in forks[:i0])
        groups[key].append((rows, forks[i0:], last))
    return groups


def stage_set(rows_forks):
    """The set S_{i0}: first small fork, or the leaf if the stage is empty."""
    _, forks, last = rows_forks
    return forks[0][1] if forks else last


CASES = [(4, 4, 4), (5, 3, 5), (3, 4, 3), (5, 4, 3), (6, 3, 3)]  # (m, n, s1)


@pytest.fixture(scope="module", params=CASES, ids=str)
def stage(request):
    m, n, s1 = request.param
    return m, n, s1, small_stage(n, m, s1, 0.4)


# ------------------------------------------------------- the walk identity


def f_float(s):
    a = s.bit_length() - 1
    return a + 2 - 2 ** (a + 1) / s - math.log2(s)


def T_float(s):
    x = (s // 2) / s
    return 1 + x * math.log2(x) + (1 - x) * math.log2(1 - x)


def test_f_equals_complete_tree_excess_and_is_at_most_gamma():
    assert abs(GAMMA - 0.0860713) < 1e-6
    for s in range(1, 1 << 14):
        value = sum(float(c) * T_float(t) for t, c in f_vector(s).items())
        assert abs(value - f_float(s)) < 1e-10
        assert -1e-12 <= f_float(s) <= GAMMA + 1e-12


def test_stopped_walk_identity_is_exact(stage):
    """E[sum_{i<tau} T_i | H] = f(s) - E[f(s_tau); tau < inf | H], per history."""
    m, n, s1, groups = stage
    assert groups
    for key, items in groups.items():
        lhs = Counter()
        size = POP[stage_set(items[0])]
        for rows, forks, last in items:
            assert POP[forks[0][1] if forks else last] == size
            for fork in forks:
                if not is_perfect(fork):
                    add(lhs, f_vector(POP[fork[1]]))
                    break
                lhs[POP[fork[1]]] += 1
        rhs = Counter()
        add(rhs, f_vector(size), len(items))
        assert {k: v for k, v in lhs.items() if v} == {k: v for k, v in rhs.items() if v}


def test_conditioning_on_success_biases_the_walk():
    """m = 5, n = 3: given that every fork is perfect, j joins the 3-side with
    probability 201/385, not 3/5, and the conditioned mean of sum T_i is below f(5)."""
    n, m = 3, 5
    total = hits = 0
    for rows in itertools.product(range(1 << m), repeat=n):
        forks, _ = walk(rows, m)
        if all(is_perfect(f) for f in forks):
            total += 1
            hits += len(forks) >= 2 and POP[forks[1][1]] == 3
    weight = Fraction(hits, total)
    assert weight == Fraction(201, 385)
    biased = T_float(5) + float(weight) * T_float(3)
    assert abs(biased - 0.071705) < 1e-6
    assert abs(f_float(5) - 0.078072) < 1e-6


# --------------------------------------------- failure bounds, per history


def row_probabilities(key, S, R, m):
    """pi_r = min over U in S, |U| >= 2, of P(r perfect on U | A_{r,i0})."""
    out = {}
    for r in R:
        support = [x for x in range(1 << m) if in_A(x, r, key)]
        assert support
        out[r] = min(
            Fraction(sum(perfect_on(x, U) for x in support), len(support))
            for U in subsets(S)
        )
    return out


def test_group_and_union_bounds_hold_per_history(stage):
    m, n, s1, groups = stage
    for key, items in groups.items():
        S = stage_set(items[0])
        s = POP[S]
        if s < 2:
            continue
        R = [r for r in range(n) if r not in {f[0] for f in key}]
        levels = math.ceil(math.log2(s))
        pi = row_probabilities(key, S, R, m)
        fail = Fraction(
            sum(1 for _, forks, _ in items if not all(is_perfect(f) for f in forks)),
            len(items),
        )
        grouped = sum(
            math.prod((1 - pi[r] for r in block), start=Fraction(1))
            for block in blocks_of(R, levels)
        )
        union = (2 ** (s - 1) - 1) * math.prod(
            (1 - pi[r] for r in R), start=Fraction(1)
        )
        assert fail <= grouped
        assert fail <= union


def test_levels_of_perfect_splits():
    """A perfect path from size s has at most ceil(log s) forks, and one has exactly that."""
    for s in range(2, 2000):
        depth, x = 0, s
        while x > 1:
            x, depth = x - x // 2, depth + 1
        assert depth == math.ceil(math.log2(s)) == (s - 1).bit_length()


def test_block_rows_keep_only_the_large_stage_conditioning():
    """Given H_{i0+1} on G_1, the rows of block 1 are uniform on prod_r A_{r,i0}."""
    m, n, s1 = 5, 4, 3
    groups = small_stage(n, m, s1, 0.4)
    checked = 0
    for key, items in groups.items():
        S = stage_set(items[0])
        if POP[S] != 3:
            continue
        R = [r for r in range(n) if r not in {f[0] for f in key}]
        b0, b1 = blocks_of(R, 2)
        joint = defaultdict(Counter)
        for rows, forks, _ in items:
            first = forks[0]
            if not is_perfect(first) or first[0] > b0[-1] or len(forks) < 2:
                continue
            history = (first[0], first[2], forks[1][1])
            joint[history][tuple(rows[r] for r in b1)] += 1
        for history, counts in joint.items():
            supports = [
                [x for x in range(1 << m) if in_A(x, r, key)] for r in b1
            ]
            expected = set(itertools.product(*supports))
            assert set(counts) == expected
            assert len(set(counts.values())) == 1
            checked += 1
    assert checked > 0


def test_earlier_rows_are_dead_only_at_sizes_two_and_three():
    """A row imperfect on a fork of size 2 or 3 is constant there; from size 4 it need not be."""
    for s in (2, 3):
        S = (1 << s) - 1
        for x in range(1 << s):
            if not perfect_on(x, S):
                assert x in (0, S)
    for s in range(4, 8):
        S = (1 << s) - 1
        alive = [
            x
            for x in range(1 << s)
            if not perfect_on(x, S)
            and any(
                perfect_on(x, U) and POP[U] in (s // 2, s - s // 2) for U in subsets(S)
            )
        ]
        assert alive


# --------------------------------------------- the lower bound on p


def hyp_perfect(N, k, u):
    """Exact P(X in {floor(u/2), ceil(u/2)}), X ~ Hyp(N, k, u)."""
    xs = {u // 2, u - u // 2}
    return Fraction(
        sum(math.comb(k, x) * math.comb(N - k, u - x) for x in xs), math.comb(N, u)
    )


def p_exact(s1, T, b):
    """Minimum of hyp_perfect over the finite moderate range; also the closed form."""
    best, N = None, s1 + 1
    while N / 2 - b * math.sqrt(N) <= s1:
        smax = min(s1, math.floor(N / 2 + b * math.sqrt(N)))
        klo = math.ceil(N / 2 - T * math.sqrt(N))
        khi = math.floor(N / 2 + T * math.sqrt(N))
        for u in range(2, smax + 1):
            for k in range(klo, khi + 1):
                value = hyp_perfect(N, k, u)
                closed = Fraction(1, 4**u * (u + 1))
                assert value >= closed
                if best is None or value < best:
                    best = value
        N += 1
    return best


@pytest.mark.parametrize(
    "s1, T, b, low, high",
    [
        (47, 0.75, 0.1, 0.05568, 0.05569),
        (78, 1.0, 0.1, 0.01689, 0.01690),
    ],
)
def test_closed_form_for_p(s1, T, b, low, high):
    assert s1 >= 64 * (T + b) ** 2
    value = p_exact(s1, T, b)
    assert low < value < high
    assert value >= Fraction(1, 4**s1 * (s1 + 1))
