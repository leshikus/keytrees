r"""
Identification keys that tolerate wrong test answers

Two constructions for species at pairwise Hamming distance at least `2t+1`:

- the *sphere key* of the 1998 program ``AUGMENT``, a tree on every word within distance
  `t` of a species;
- the *elimination key*, added in 2026, which discards a species after `t+1`
  disagreements and stops when one species remains.

The C++ program ``ecc`` runs the large experiments; this module is the readable
reference version and is what the tests compare against.

EXAMPLES::

    >>> code = ["00000", "11111"]
    >>> minimum_distance(code)
    5
    >>> identify(code, "01100", t=2)
    (0, 5)
"""
from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations


def hamming(a: str, b: str) -> int:
    r"""
    Return the Hamming distance between two words of equal length.

    EXAMPLES::

        >>> hamming("0011", "0101")
        2
    """
    return sum(x != y for x, y in zip(a, b, strict=True))


def minimum_distance(code: Sequence[str]) -> int:
    r"""
    Return the smallest distance between two distinct words of ``code``.

    EXAMPLES::

        >>> minimum_distance(["000", "011", "101"])
        2
    """
    return min(hamming(a, b) for a, b in combinations(code, 2))


def ball(word: str, t: int) -> list[str]:
    r"""
    Return all words within Hamming distance ``t`` of ``word``.

    This is the set of leaves that ``AUGMENT`` added around a species.

    EXAMPLES::

        >>> sorted(ball("00", 1))
        ['00', '01', '10']
    """
    words = []
    for r in range(t + 1):
        for positions in combinations(range(len(word)), r):
            flipped = list(word)
            for p in positions:
                flipped[p] = "1" if flipped[p] == "0" else "0"
            words.append("".join(flipped))
    return words


def identify(code: Sequence[str], answers: str, t: int) -> tuple[int, int]:
    r"""
    Return the identified species and the number of tests used by the elimination key.

    Tests are taken in order. A species is discarded after `t+1` disagreements, and the
    walk stops when one species remains.

    If the minimum distance of ``code`` is at least `2t+1` and ``answers`` differs from
    the true word in at most `t` places, the result is always the true species: the true
    species is never discarded, and two survivors at the end would have to be within
    distance `2t` of each other.

    INPUT:

    - ``code`` -- words of the species
    - ``answers`` -- the specimen's answers, possibly with errors
    - ``t`` -- the number of errors to tolerate

    EXAMPLES::

        >>> identify(["0000000", "1110000", "0001111"], "0100000", t=1)
        (0, 4)

    TESTS::

        >>> identify(["000", "001"], "000", t=1)
        Traceback (most recent call last):
        ...
        ValueError: minimum distance 1 is below 2t+1 = 3
    """
    distance = minimum_distance(code)
    if distance < 2 * t + 1:
        raise ValueError(f"minimum distance {distance} is below 2t+1 = {2 * t + 1}")
    wrong = [0] * len(code)
    alive = set(range(len(code)))
    tests = 0
    for position, answer in enumerate(answers):
        if len(alive) == 1:
            break
        tests += 1
        for i in list(alive):
            if code[i][position] != answer:
                wrong[i] += 1
                if wrong[i] > t:
                    alive.discard(i)
    if len(alive) != 1:
        raise ValueError("more than t errors in the answers")
    return alive.pop(), tests


def volume_bound(m: int, t: int) -> int:
    r"""
    Return the least `q` with `2^q \ge m \sum_{i \le t} \binom{q}{i}`.

    Any adaptive search among `m` objects with at most `t` lies needs at least this many
    questions in the worst case when questions are free. It is the volume bound used in
    the Ulam--Renyi game.

    EXAMPLES::

        >>> volume_bound(2 ** 20, 0)
        20
        >>> volume_bound(2 ** 20, 1)
        25
    """
    from math import comb

    q = 1
    while 2**q < m * sum(comb(q, i) for i in range(t + 1)):
        q += 1
    return q
