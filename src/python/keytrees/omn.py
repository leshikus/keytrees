r"""
(o, m, n) trees: brute force from the definition and published closed forms

An `(o, m, n)` tree, `m \le n`, asks free binary questions (any subset of the
`o` objects) and names the object whenever at most `m` answers are wrong; when
at most `n` are wrong it names the object or reports errors, never a wrong
object. The C++ engine ``omn`` computes the least worst-case depth `W(o,m,n)`
and the least error-free total depth `A(o,m,n)` on class-size vectors. This
module is the independent reference: it works on *labelled* objects, asks
every subset, and decides whether a leaf is allowed by listing the scenarios
(true object, number of wrong answers) directly, so it uses neither the
symmetry reduction nor the class-vector child formula. It is exponential and
meant for `o \le 5`.

EXAMPLES::

    >>> brute_worst(3, 1, 1), pelc_one_lie(3)
    (5, 5)
    >>> brute_total(3, 0, 0), complete_tree_total(3)
    (5, 5)
    >>> brute_worst(4, 0, 2), log2_ceil(4) + 2
    (4, 4)
"""
from __future__ import annotations

from functools import cache

ERROR = -1


def log2_ceil(o: int) -> int:
    r"""
    Return `\lceil \log_2 o \rceil` for `o \ge 1`.

    EXAMPLES::

        >>> [log2_ceil(o) for o in range(1, 10)]
        [0, 1, 2, 2, 3, 3, 3, 3, 4]
    """
    return (o - 1).bit_length()


def complete_tree_total(o: int) -> int:
    r"""
    Return the least sum of leaf depths of a binary tree with `o` leaves,
    `o(l+2) - 2^{l+1}` with `l = \lfloor \log_2 o \rfloor`.

    EXAMPLES::

        >>> [complete_tree_total(o) for o in range(1, 9)]
        [0, 2, 5, 8, 12, 16, 20, 24]
    """
    level = o.bit_length() - 1
    return o * (level + 2) - 2 ** (level + 1)


def ball(q: int, k: int) -> int:
    r"""
    Return `V(q, k) = \sum_{j \le k} \binom{q}{j}`.

    EXAMPLES::

        >>> ball(5, 1), ball(5, 2), ball(3, 5)
        (6, 16, 8)
    """
    from math import comb

    return sum(comb(q, j) for j in range(k + 1))


def character(o: int, e: int) -> int:
    r"""
    Return the Berlekamp character `\min\{q : o V(q, e) \le 2^q\}` of the
    initial state with `o` objects and `e` lies (the Hamming bound).

    EXAMPLES::

        >>> character(3, 1), character(2, 3)
        (4, 7)
    """
    q = 0
    while o * ball(q, e) > 2**q:
        q += 1
    return q


def pelc_one_lie(o: int) -> int:
    r"""
    Return Pelc's (1987) least number of questions with one lie: the least `q`
    with `o(q+1) \le 2^q` for even `o`, and with `o(q+1) + (q-1) \le 2^q` for
    odd `o`. The statement is taken from Pelc's 2002 survey, Section 3.1.1,
    not from the 1987 paper.

    EXAMPLES::

        >>> [pelc_one_lie(o) for o in range(1, 11)]
        [0, 3, 5, 5, 6, 6, 6, 6, 7, 7]
    """
    q = 0
    while o * (q + 1) + (q - 1 if o % 2 else 0) > 2**q:
        q += 1
    return q


# Deppe, preprint of 15 March 1999 (SFB 343, Bielefeld), Theorem 40, case N < 266:
# ch(N) + 2 questions for N in M2, ch(N) for N in M0, ch(N) + 1 otherwise.
DEPPE_M2 = frozenset({3, 5})
DEPPE_M0 = frozenset(
    {1, 2}
    | set(range(14, 17)) | set(range(22, 29)) | set(range(35, 51))
    | set(range(57, 89)) | set(range(95, 155)) | set(range(158, 265))
)


def deppe_three_lies(o: int) -> int:
    r"""
    Return the least number of questions with three lies for `1 \le o \le 265`,
    as stated in Theorem 40 of Deppe's 1999 preprint.

    EXAMPLES::

        >>> [deppe_three_lies(o) for o in (1, 2, 3, 4, 5, 14)]
        [0, 7, 11, 11, 12, 13]
    """
    if not 1 <= o <= 265:
        raise ValueError("the small case of the theorem covers 1 <= o <= 265")
    k = character(o, 3)
    return k + (2 if o in DEPPE_M2 else 0 if o in DEPPE_M0 else 1)


def _leaf_allowed(counts: tuple[int, ...], m: int, n: int) -> bool:
    # Scenarios still possible: (object y, c wrong answers) with c <= n.
    scen = [(y, c) for y, c in enumerate(counts) if c <= n]
    for label in [ERROR, *range(len(counts))]:
        if all(label == y if c <= m else label in (y, ERROR) for y, c in scen):
            return True
    return False


def _answer(counts: tuple[int, ...], mask: int, yes: bool, n: int) -> tuple[int, ...]:
    return tuple(
        min(n + 1, c + (bool(mask >> y & 1) != yes)) for y, c in enumerate(counts)
    )


def brute_worst(o: int, m: int, n: int) -> int:
    r"""
    Return `W(o, m, n)` by iterative deepening over all subsets as questions.

    EXAMPLES::

        >>> [brute_worst(2, m, n) for m, n in ((0, 0), (0, 1), (1, 1), (1, 2))]
        [1, 2, 3, 4]
    """

    @cache
    def ok(counts: tuple[int, ...], q: int) -> bool:
        if _leaf_allowed(counts, m, n):
            return True
        if q == 0:
            return False
        return any(
            ok(_answer(counts, s, True, n), q - 1) and ok(_answer(counts, s, False, n), q - 1)
            for s in range(2**o)
        )

    q = 0
    while not ok((0,) * o, q):
        q += 1
    return q


def brute_total(o: int, m: int, n: int) -> int:
    r"""
    Return `A(o, m, n)`, the least sum over the objects of the number of
    questions asked when every answer is right. Questions with every possible
    object on one side are skipped; they can be removed from any tree without
    raising its cost.

    EXAMPLES::

        >>> [brute_total(3, 0, n) for n in range(3)]
        [5, 8, 11]
    """

    @cache
    def best(counts: tuple[int, ...]) -> int:
        if _leaf_allowed(counts, m, n):
            return 0
        live = [y for y, c in enumerate(counts) if c <= n]
        here = sum(c == 0 for c in counts)
        value = None
        for s in range(2**o):
            side = [bool(s >> y & 1) for y in live]
            if all(side) or not any(side):
                continue
            v = here + best(_answer(counts, s, True, n)) + best(_answer(counts, s, False, n))
            value = v if value is None else min(value, v)
        assert value is not None
        return value

    return best((0,) * o)


def brute_bounded_total(o: int, m: int, n: int, q: int | None = None) -> int | None:
    r"""
    Return the least error-free total among the `(o, m, n)` trees of depth at
    most `q` (by default `q = W(o, m, n)`), or ``None`` if there is none.

    EXAMPLES::

        >>> brute_bounded_total(3, 1, 1), brute_bounded_total(3, 1, 1, 4)
        (8, None)
    """
    if q is None:
        q = brute_worst(o, m, n)

    @cache
    def best(counts: tuple[int, ...], r: int) -> int | None:
        if _leaf_allowed(counts, m, n):
            return 0
        if r == 0:
            return None
        here = sum(c == 0 for c in counts)
        value = None
        for s in range(2**o):
            vy = best(_answer(counts, s, True, n), r - 1)
            vz = best(_answer(counts, s, False, n), r - 1)
            if vy is None or vz is None:
                continue
            v = here + vy + vz
            value = v if value is None else min(value, v)
        return value

    return best((0,) * o, q)
