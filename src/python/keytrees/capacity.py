r"""
Minimax identification keys under a partial order on frequencies

Given inequalities `p_i \le p_j`, the set of admissible distributions is a polytope.
By Ryabko's theorem, the smallest worst-case redundancy of a key is within 1 of the
capacity of the channel whose rows are the polytope's vertices. So a good key comes
from the Blahut--Arimoto algorithm followed by Huffman's.

The Maple worksheets ``partord.mws``, ``po2.mws``, ``capacity.mws`` held the 1999
versions of these computations.

EXAMPLES::

    >>> Q = order_vertices(3, [(0, 1), (1, 2)])
    >>> capacity, q = blahut_arimoto(Q)
    >>> round(capacity, 6)
    0.483517
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import product

import numpy as np


def order_vertices(m: int, order: Iterable[tuple[int, int]], reduce: bool = True) -> np.ndarray:
    r"""
    Return the vertices of the polytope `\{p : p_i \le p_j \text{ for } (i,j) \in I\}`.

    Each vertex is uniform on an up-set: a set containing `j` whenever it contains `i`
    and `(i,j) \in I`. The theorem in the 1999 draft lists every such point. Some of them
    are convex combinations of others, and ``reduce`` removes those.

    The enumeration takes `2^m` steps, as the draft notes.

    INPUT:

    - ``m`` -- number of objects
    - ``order`` -- pairs `(i, j)` meaning `p_i \le p_j`
    - ``reduce`` -- whether to keep only true vertices

    EXAMPLES::

        >>> order_vertices(2, [(0, 1)])
        array([[0.5, 0.5],
               [0. , 1. ]])

    TESTS::

        >>> order_vertices(2, [(0, 1)], reduce=False).shape
        (2, 2)
    """
    pairs = list(order)
    sets = []
    for bits in product((0, 1), repeat=m):
        if sum(bits) and all(bits[i] <= bits[j] for i, j in pairs):
            sets.append(frozenset(i for i in range(m) if bits[i]))
    if reduce:
        # A uniform point on S is a mixture of those on smaller up-sets exactly when S is
        # a disjoint union of nonempty up-sets.
        def decomposes(s: frozenset[int]) -> bool:
            smaller = [t for t in sets if t < s]
            return any(s - t in smaller for t in smaller)

        sets = [s for s in sets if not decomposes(s)]
    sets.sort(key=lambda s: (-len(s), sorted(s)))
    rows = np.zeros((len(sets), m))
    for r, s in enumerate(sets):
        rows[r, list(s)] = 1.0 / len(s)
    return rows


def blahut_arimoto(Q: np.ndarray, eps: float = 1e-9, max_iter: int = 2_000_000) -> tuple[float, np.ndarray]:
    r"""
    Return the capacity in bits and a maximising input distribution of the channel ``Q``.

    ALGORITHM:

    Iterate `q_i \leftarrow q_i c_i / \sum_k q_k c_k`, where
    `\log_2 c_i = \sum_j Q_{ij} \log_2 (Q_{ij} / \sum_k q_k Q_{kj})`. Stop when
    `\log_2 \max_i c_i - \log_2 \sum_i q_i c_i < \varepsilon`; these two numbers bound the
    capacity from above and below.

    INPUT:

    - ``Q`` -- row-stochastic matrix
    - ``eps`` -- gap between the bounds at which to stop
    - ``max_iter`` -- limit on the number of iterations

    EXAMPLES::

        >>> blahut_arimoto(np.eye(4))[0]
        2.0

    TESTS::

        >>> blahut_arimoto(np.array([[0.5, 0.6]]))
        Traceback (most recent call last):
        ...
        ValueError: rows of Q must sum to 1
    """
    Q = np.asarray(Q, dtype=float)
    if not np.allclose(Q.sum(axis=1), 1.0):
        raise ValueError("rows of Q must sum to 1")
    n = Q.shape[0]
    q = np.full(n, 1.0 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(max_iter):
            out = q @ Q
            ratio = np.where(Q > 0, Q * np.log2(Q / out), 0.0)
            c = np.exp2(ratio.sum(axis=1))
            lower = float(np.log2(q @ c))
            upper = float(np.log2(c.max()))
            if upper - lower < eps:
                return lower, q
            q = q * c / (q @ c)
    raise RuntimeError(f"no convergence within {max_iter} iterations; gap {upper - lower}")


def channel_output(Q: np.ndarray, q: np.ndarray) -> np.ndarray:
    r"""
    Return the output distribution of the channel for input ``q``.

    EXAMPLES::

        >>> channel_output(np.eye(2), np.array([0.25, 0.75]))
        array([0.25, 0.75])
    """
    return np.asarray(q) @ np.asarray(Q)


def worst_case(lengths: Sequence[float], Q: np.ndarray) -> tuple[float, float]:
    r"""
    Return the largest average cost and largest redundancy of a key over the polytope.

    The cost is linear and the redundancy convex in the distribution, so both reach their
    maximum at a vertex.

    INPUT:

    - ``lengths`` -- path lengths of the key, one per object
    - ``Q`` -- vertices of the polytope, as rows

    EXAMPLES::

        >>> worst_case([1, 1], np.array([[0.5, 0.5], [0.0, 1.0]]))
        (1.0, 1.0)
    """
    L = np.asarray(lengths, dtype=float)
    costs = Q @ L
    with np.errstate(divide="ignore", invalid="ignore"):
        entropies = -np.where(Q > 0, Q * np.log2(Q), 0.0).sum(axis=1)
    return float(costs.max()), float((costs - entropies).max())
