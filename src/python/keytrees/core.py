r"""
Identification trees over binary tests

A :class:`Table` lists objects and tests. A test answers ``1``, ``0`` or ``None``;
``None`` means the test is undefined for that object, and such a test may not be used
while that object is still a candidate, as in the 1998 program ``RTA``.

EXAMPLES::

    >>> t = Table.from_rows(["a", "b", "c", "d"], {"x": [1, 1, 0, 0], "y": [1, 0, 1, 0]})
    >>> sorted(greedy_tree(t).depths().items())
    [('a', 2), ('b', 2), ('c', 2), ('d', 2)]
    >>> huffman_cost([0.25] * 4)
    2.0
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Table:
    r"""
    Objects, test names and answers.

    INPUT:

    - ``objects`` -- tuple of object names
    - ``tests`` -- tuple of test names
    - ``answers`` -- tuple of rows, one for each test, each with one value per object

    EXAMPLES::

        >>> t = Table.from_rows(["a", "b"], {"x": [1, 0]})
        >>> t.objects, t.tests
        (('a', 'b'), ('x',))
    """

    objects: tuple[str, ...]
    tests: tuple[str, ...]
    answers: tuple[tuple[int | None, ...], ...]

    @classmethod
    def from_rows(cls, objects: Sequence[str], rows: Mapping[str, Sequence[int | None]]) -> Table:
        r"""
        Return a table from a mapping of test name to answers.

        EXAMPLES::

            >>> Table.from_rows(["a"], {"x": [1]}).answers
            ((1,),)

        TESTS::

            >>> Table.from_rows(["a", "b"], {"x": [1]})
            Traceback (most recent call last):
            ...
            ValueError: test 'x' has 1 answers for 2 objects
        """
        for name, row in rows.items():
            if len(row) != len(objects):
                raise ValueError(f"test {name!r} has {len(row)} answers for {len(objects)} objects")
        return cls(tuple(objects), tuple(rows), tuple(tuple(r) for r in rows.values()))

    def uniform(self) -> dict[str, float]:
        r"""
        Return the uniform distribution on the objects.

        EXAMPLES::

            >>> Table.from_rows(["a", "b"], {"x": [1, 0]}).uniform()
            {'a': 0.5, 'b': 0.5}
        """
        return {name: 1.0 / len(self.objects) for name in self.objects}

    def split(self, test: int, candidates: frozenset[int]) -> tuple[frozenset[int], frozenset[int]] | None:
        r"""
        Return the objects answering 0 and 1, or ``None`` if the test cannot be used.

        A test cannot be used if it is undefined on a candidate or does not separate
        the candidates.

        EXAMPLES::

            >>> t = Table.from_rows(["a", "b", "c"], {"x": [1, 0, None]})
            >>> t.split(0, frozenset({0, 1}))
            (frozenset({1}), frozenset({0}))
            >>> t.split(0, frozenset({0, 2})) is None
            True
        """
        row = self.answers[test]
        if any(row[i] is None for i in candidates):
            return None
        zero = frozenset(i for i in candidates if row[i] == 0)
        one = candidates - zero
        return (zero, one) if zero and one else None


@dataclass(frozen=True)
class Node:
    r"""
    A tree node: a leaf with ``name``, or a fork with ``test``, ``no`` and ``yes``.

    EXAMPLES::

        >>> Node(test="x", no=Node(name="a"), yes=Node(name="b")).depths()
        {'a': 1, 'b': 1}
    """

    name: str | None = None
    test: str | None = None
    no: Node | None = None
    yes: Node | None = None

    def depths(self, depth: int = 0) -> dict[str, int]:
        r"""
        Return the mapping from leaf name to its depth.

        EXAMPLES::

            >>> Node(name="a").depths()
            {'a': 0}
        """
        if self.name is not None:
            return {self.name: depth}
        assert self.no is not None and self.yes is not None
        return {**self.no.depths(depth + 1), **self.yes.depths(depth + 1)}

    def cost(self, p: Mapping[str, float]) -> float:
        r"""
        Return the average number of tests under the distribution ``p``.

        EXAMPLES::

            >>> tree = Node(test="x", no=Node(name="a"), yes=Node(name="b"))
            >>> tree.cost({"a": 0.5, "b": 0.5})
            1.0
        """
        return sum(p[name] * d for name, d in self.depths().items())


def balance_score(table: Table, test: int, candidates: frozenset[int], p: Sequence[float]) -> float | None:
    r"""
    Return the 1998 "Greedy Up" score of a test; smaller is better.

    The score is `n_0 P_0 + n_1 P_1`, where `n_i` and `P_i` are the size and probability
    of each part. Under the uniform distribution it is smallest for the most balanced
    test, which is the gradient rule of the 1999 paper. ``None`` means the test cannot
    be used.

    EXAMPLES::

        >>> t = Table.from_rows("abcd", {"x": [1, 0, 0, 0], "y": [1, 1, 0, 0]})
        >>> u = [0.25] * 4
        >>> balance_score(t, 0, frozenset(range(4)), u), balance_score(t, 1, frozenset(range(4)), u)
        (2.5, 2.0)
    """
    parts = table.split(test, candidates)
    if parts is None:
        return None
    zero, one = parts
    return len(zero) * sum(p[i] for i in zero) + len(one) * sum(p[i] for i in one)


def greedy_tree(table: Table, p: Mapping[str, float] | None = None) -> Node:
    r"""
    Return the tree of the gradient algorithm, taking the first best test at each node.

    INPUT:

    - ``table`` -- a :class:`Table`
    - ``p`` -- optional distribution on the objects; the default is uniform

    EXAMPLES::

        >>> t = Table.from_rows("abc", {"x": [1, 0, 0], "y": [0, 1, 0]})
        >>> sorted(greedy_tree(t).depths().values())
        [1, 2, 2]

    TESTS::

        >>> greedy_tree(Table.from_rows("ab", {"x": [1, 1]}))
        Traceback (most recent call last):
        ...
        ValueError: no usable test separates ['a', 'b']
    """
    dist = [(p or table.uniform())[name] for name in table.objects]

    def build(candidates: frozenset[int]) -> Node:
        if len(candidates) == 1:
            return Node(name=table.objects[next(iter(candidates))])
        best, best_score = None, math.inf
        for j in range(len(table.tests)):
            score = balance_score(table, j, candidates, dist)
            if score is not None and score < best_score - 1e-12:
                best, best_score = j, score
        if best is None:
            raise ValueError(f"no usable test separates {sorted(table.objects[i] for i in candidates)}")
        parts = table.split(best, candidates)
        assert parts is not None
        return Node(test=table.tests[best], no=build(parts[0]), yes=build(parts[1]))

    return build(frozenset(range(len(table.objects))))


def optimal_cost(table: Table, p: Mapping[str, float] | None = None) -> float:
    r"""
    Return the minimal average number of tests over all trees for these tests.

    The problem is NP-hard, so this memoised search is for small tables only. The
    original ``RTA`` left this computation as a stub.

    EXAMPLES::

        >>> t = Table.from_rows("abcd", {"x": [1, 1, 0, 0], "y": [1, 0, 1, 0]})
        >>> optimal_cost(t)
        2.0
    """
    dist = [(p or table.uniform())[name] for name in table.objects]

    @lru_cache(maxsize=None)
    def best(candidates: frozenset[int]) -> float:
        if len(candidates) <= 1:
            return 0.0
        here = sum(dist[i] for i in candidates)
        options = [
            here + best(parts[0]) + best(parts[1])
            for j in range(len(table.tests))
            if (parts := table.split(j, candidates)) is not None
        ]
        return min(options, default=math.inf)

    return best(frozenset(range(len(table.objects))))


def entropy(p: Sequence[float]) -> float:
    r"""
    Return the binary entropy of ``p``.

    EXAMPLES::

        >>> entropy([0.5, 0.25, 0.25])
        1.5
    """
    return -sum(x * math.log2(x) for x in p if x > 0)


def shannon_cost(p: Sequence[float]) -> float:
    r"""
    Return the average length of the Shannon code `\lceil -\log_2 p_i \rceil`.

    EXAMPLES::

        >>> shannon_cost([0.5, 0.25, 0.25])
        1.5
    """
    return sum(x * math.ceil(-math.log2(x)) for x in p if x > 0)


def huffman_lengths(p: Sequence[float]) -> list[int]:
    r"""
    Return optimal codeword lengths for the probabilities ``p``.

    SEEALSO:

    Sage's ``sage.coding.source_coding.huffman`` builds Huffman codes for encoding text;
    this function is only the length computation needed here.

    EXAMPLES::

        >>> huffman_lengths([0.5, 0.25, 0.25])
        [1, 2, 2]

    TESTS::

        >>> huffman_lengths([1.0])
        [0]
    """
    heap = [(x, i, (i,)) for i, x in enumerate(p)]
    heapq.heapify(heap)
    lengths = [0] * len(p)
    counter = len(p)
    while len(heap) > 1:
        a, _, members_a = heapq.heappop(heap)
        b, _, members_b = heapq.heappop(heap)
        for i in members_a + members_b:
            lengths[i] += 1
        heapq.heappush(heap, (a + b, counter, members_a + members_b))
        counter += 1
    return lengths


def huffman_cost(p: Sequence[float]) -> float:
    r"""
    Return the average Huffman codeword length.

    EXAMPLES::

        >>> round(huffman_cost([0.4, 0.3, 0.2, 0.1]), 12)
        1.9
    """
    return sum(x * n for x, n in zip(p, huffman_lengths(p)))


def complete_tree_cost(m: int) -> float:
    r"""
    Return the optimal uniform cost `l + 2 - 2^{l+1}/m`, with `l = \lfloor\log_2 m\rfloor`.

    EXAMPLES::

        >>> complete_tree_cost(4), round(complete_tree_cost(3), 12)
        (2.0, 1.666666666667)
    """
    level = m.bit_length() - 1
    return level + 2 - 2 ** (level + 1) / m
