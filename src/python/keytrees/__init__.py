r"""
Identification keys and search trees

Tools extracted and modernised from the 1998--1999 work on identification keys:
key construction (:mod:`keytrees.core`), minimax keys for ordered frequencies
(:mod:`keytrees.capacity`), keys that tolerate wrong test answers
(:mod:`keytrees.errors`), and the real keys used in the text
(:mod:`keytrees.flora`).

EXAMPLES::

    >>> from keytrees.core import Table, greedy_tree, optimal_cost
    >>> t = Table.from_rows(["AB", "AC", "BC"], {"t1": [1, 1, 0], "t2": [1, 0, 1]})
    >>> greedy_tree(t).cost(t.uniform())
    1.6666666666666665
    >>> optimal_cost(t, t.uniform())
    1.6666666666666665
"""

__all__ = ["capacity", "core", "errors", "flora"]
