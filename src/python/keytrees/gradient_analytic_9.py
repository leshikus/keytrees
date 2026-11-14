r"""
Step 9 of the analytic estimate for the gradient key: exact values for small `m` and `n`.

See ``articles/gradient-bound/analytic/09-exact.tex`` in the ``codes`` repository and
``PLAN.md`` there for the notation. Everything here is exact rational arithmetic, except
where a value involves `\log` or `\sqrt\pi`; such values are evaluated in ``mpmath`` with
40 digits, for display and comparison only.

Under `\mathbf P` (iid fair bits on `n \times m` matrices, columns not necessarily
distinct), `G` is run with the rule of Step 1: at a set `S`, `|S| \ge 2`, the first row of
largest quality is tested if that quality is at least `1`; otherwise the construction
stops. Along the path `S_0 = [m] \supset S_1 \supset \dots` of column `1` (index `0`
here) a *class* is a key ``(sizes, rows)``: the sizes `s_0, \dots, s_i` and the rows
`r_0, \dots, r_{i-1}` chosen on the path. For each class with `s_i \ge 2` we keep

- ``W``: the number of matrices in the class (out of `2^{nm}`);
- ``qh[q]``: how many of them have largest quality `q` at `S_i` (`q = 0`: a stop);
- ``kh[r][q]``, ``kp[r][q]``: for each row `r` not chosen on the path, how many have
  `q_r(S_i) = q`, respectively `q_r(S_{i-1}) = q`.

Three independent sources give the same numbers:

- :func:`brute_classes`: every matrix, in pure Python (tiny cases only);
- ``gradient_enum brute|sym`` (``src/cpp/gradient_enum.cpp``), read by :func:`parse_enum`:
  every matrix, or every matrix up to row complements and the order of columns `2..m`;
- :func:`product_classes`: the product formula of Step 1 (``thm:exposure``), with one
  dynamic programme per row over its counts on the nested sets.

The quantities the steps bound are computed from class data by :func:`fork_records`
and summarised by :func:`depth_summary`; :func:`step5_failure` gives the exact
probability that a fork on the path of column `1` is not perfect, and
:func:`step5_assembled` the resulting bound when `m \le s_1`.

EXAMPLES::

    >>> from keytrees.gradient_analytic_9 import brute_classes, product_classes
    >>> brute_classes(2, 3) == product_classes(2, 3)
    True
    >>> c = product_classes(3, 4)
    >>> c[(4,), ()]["W"] == 2**12
    True
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import csv
import itertools
import math
from fractions import Fraction
from functools import lru_cache
from math import comb
from pathlib import Path

import mpmath

from keytrees.gradient_bound import (
    complete_cost,
    distinct_probability,
    rho_upper,
    theorem1_excess_bound,
)

mpmath.mp.dps = 40

# Cells of the full enumeration (gradient_enum sym), and the (b, s_1) pairs studied there.
ENUM_CELLS = [(4, 8), (5, 8), (6, 8), (6, 9), (4, 12), (5, 10), (5, 11), (5, 12), (4, 16), (4, 20), (7, 6), (7, 7), (8, 5), (8, 6), (9, 5)]
ENUM_BS = [(Fraction(1, 5), 5), (Fraction(2, 5), 3)]


def _q(k, s):
    return min(k, s - k)


# ---------------------------------------------------------------- class data, three ways


def brute_classes(n, m):
    r"""
    Return the class data of all `2^{nm}` matrices, by direct enumeration (pure Python).

    The result maps ``(sizes, rows)`` to a dict with keys ``W``, ``qh``, ``kh``, ``kp``
    (``kh`` and ``kp`` map a row index to a list indexed by quality). Feasible for
    `nm \le 16` or so.

    EXAMPLES:

    With one row and two columns the root is a fork exactly when the row splits::

        >>> d = brute_classes(1, 2)
        >>> d[(2,), ()]
        {'W': 4, 'qh': [2, 2], 'kh': {0: [2, 2]}, 'kp': {0: []}}

    Every matrix passes the root::

        >>> sum(v["W"] for (s, r), v in brute_classes(2, 4).items() if len(s) == 1)
        256
    """
    out = {}
    full = (1 << m) - 1
    for cells in itertools.product(range(1 << m), repeat=n):
        rows = list(cells)  # rows[r]: bit c is column c
        S, prev, sizes, chosen = full, 0, [m], []
        while True:
            s = sizes[-1]
            if s < 2:
                break
            ks = [bin(rows[r] & S).count("1") for r in range(n)]
            qs = [_q(k, s) for k in ks]
            best = max(range(n), key=lambda r: (qs[r], -r))
            bq = qs[best]
            rec = out.setdefault((tuple(sizes), tuple(chosen)), {"W": 0, "qh": [0] * (s // 2 + 1), "kh": {}, "kp": {}})
            rec["W"] += 1
            rec["qh"][bq] += 1
            ps = sizes[-2] if len(sizes) > 1 else None
            for r in range(n):
                if r in chosen:
                    continue
                kh = rec["kh"].setdefault(r, [0] * (s // 2 + 1))
                kh[qs[r]] += 1
                kp = rec["kp"].setdefault(r, [0] * (ps // 2 + 1) if ps else [])
                if ps:
                    kp[_q(bin(rows[r] & prev).count("1"), ps)] += 1
            if bq == 0:
                break
            side = rows[best] & S if rows[best] & 1 else S & ~rows[best]
            prev, S = S, side
            sizes.append(bin(S).count("1"))
            chosen.append(best)
    return out


def parse_enum(text):
    r"""
    Read the output of ``gradient_enum`` into the format of :func:`brute_classes`.

    EXAMPLES::

        >>> t = "C 0 2 ; ; 4 ; 2 2\nR 0 ; 2 2 ;\n"
        >>> parse_enum(t) == brute_classes(1, 2)
        True
    """
    out = {}
    rec = None
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.split() for p in line[2:].split(";")]
        if line[0] == "C":
            i = int(parts[0][0])
            sizes = tuple(int(x) for x in parts[0][1:])
            rows = tuple(int(x) for x in parts[1])
            assert len(sizes) == i + 1 and len(rows) == i
            rec = {"W": int(parts[2][0]), "qh": [int(x) for x in parts[3]], "kh": {}, "kp": {}}
            out[sizes, rows] = rec
        else:
            r = int(parts[0][0])
            rec["kh"][r] = [int(x) for x in parts[1]]
            rec["kp"][r] = [int(x) for x in parts[2]]
    return out


@lru_cache(maxsize=None)
def row_table(sizes, cons, targets, fixed=False):
    r"""
    Count the patterns of one row on `S_0` by its qualities on given levels.

    The sets are nested, `|S_l| =` ``sizes[l]``. The row must satisfy, for each
    ``(l, (q, strict))`` in ``enumerate(cons)``, `q_r(S_l) < q` if ``strict`` and
    `q_r(S_l) \le q` otherwise. ``targets`` lists levels in decreasing order; the result
    maps the tuple of the row's qualities on those levels to the number of patterns.

    With ``fixed``, ``sizes = (s_0, ..., s_L, s_{L+1})`` and the row is the one chosen at
    level `L`: its pattern on `S_L` is fixed, with `s_{L+1}` ones, and only the part
    outside `S_L` is free (``targets`` must then be empty).

    EXAMPLES:

    A free row on 4 columns, by quality::

        >>> row_table((4,), (), (0,))
        {(0,): 2, (1,): 8, (2,): 6}

    Rows before the chosen row at a perfect fork of size 3 must be constant there::

        >>> row_table((3, 2), ((1, True),), (1,))
        {(0,): 2}

    The row chosen at level 1 of the chain 6 > 3 > 2, with a perfect fork above it::

        >>> row_table((6, 3, 2), ((3, False),), (), True)
        {(): 8}
    """
    if fixed:
        L = len(sizes) - 2
        states = {((), sizes[L + 1]): 1}
    else:
        L = len(sizes) - 1
        states = {}
        for k in range(sizes[L] + 1):
            tq = (_q(k, sizes[L]),) if L in targets else ()
            states[tq, k] = states.get((tq, k), 0) + comb(sizes[L], k)
    for l in range(L - 1, -1, -1):
        c = sizes[l] - sizes[l + 1]
        new = {}
        for (tq, k), w in states.items():
            for x in range(c + 1):
                kk = k + x
                qq = _q(kk, sizes[l])
                if l < len(cons):
                    q, strict = cons[l]
                    if qq > q - (1 if strict else 0):
                        continue
                t2 = tq + (qq,) if l in targets else tq
                new[t2, kk] = new.get((t2, kk), 0) + w * comb(c, x)
        states = new
    out = {}
    for (tq, _), w in states.items():
        out[tq] = out.get(tq, 0) + w
    return dict(sorted(out.items()))


def _chain_q(sizes):
    return [min(sizes[l + 1], sizes[l] - sizes[l + 1]) for l in range(len(sizes) - 1)]


def _class_from_product(n, sizes, rows):
    """Class data of one class by the product formula of Step 1, or None if it is empty."""
    i = len(rows)
    q = _chain_q(sizes)
    W = 1
    for l in range(i):
        W *= 2 * comb(sizes[l] - 1, sizes[l + 1] - 1)
        cons = tuple((q[t], rows[l] < rows[t]) for t in range(l))
        W *= sum(row_table(tuple(sizes[: l + 2]), cons, (), True).values())
    if W == 0:
        return None
    s = sizes[-1]
    free = [r for r in range(n) if r not in rows]
    targets = (i, i - 1) if i else (0,)
    tabs = {r: row_table(tuple(sizes), tuple((q[l], r < rows[l]) for l in range(i)), targets) for r in free}
    tot = {r: sum(t.values()) for r, t in tabs.items()}
    Wc = W * math.prod(tot.values())
    if Wc == 0:
        return None
    rec = {"W": Wc, "qh": [0] * (s // 2 + 1), "kh": {}, "kp": {}}
    cdf = {}
    for r in free:
        kh = [0] * (s // 2 + 1)
        kp = [0] * (sizes[-2] // 2 + 1) if i else []
        for key, w in tabs[r].items():
            kh[key[0]] += w
            if i:
                kp[key[1]] += w
        rest = Wc // tot[r]
        rec["kh"][r] = [x * rest for x in kh]
        rec["kp"][r] = [x * rest for x in kp]
        cdf[r] = list(itertools.accumulate(kh))
    prev = 0
    for x in range(s // 2 + 1):
        p = W * math.prod(cdf[r][x] for r in free)
        rec["qh"][x] = p - prev
        prev = p
    return rec


def product_classes(n, m, keep=None, max_depth=None):
    r"""
    Return class data by the product formula of Step 1 (``thm:exposure``).

    ``keep(sizes)`` may prune chains: a chain is extended only while it returns true for
    the prefix (called with at least two sizes). ``max_depth`` bounds the depth `i`.
    Without pruning the result equals :func:`brute_classes`.

    EXAMPLES::

        >>> all(product_classes(n, m) == brute_classes(n, m) for n, m in ((1, 3), (2, 4), (3, 3), (4, 2)))
        True
        >>> product_classes(3, 3)[(3, 2), (0,)]["qh"]
        [64, 192]
    """
    out = {}
    depth = n if max_depth is None else min(n, max_depth)

    def go(sizes):
        for rows in itertools.permutations(range(n), len(sizes) - 1):
            rec = _class_from_product(n, sizes, rows)
            if rec is not None:
                out[tuple(sizes), rows] = rec
        if len(sizes) - 1 >= depth:
            return
        for nxt in range(sizes[-1] - 1, 1, -1):
            new = sizes + (nxt,)
            if keep is None or keep(new):
                go(new)

    go((m,))
    return out


# ---------------------------------------------------------------- good forks and the step quantities


def good_fork(s, q, b, s1):
    r"""
    Return whether a fork of size `s` tested with quality `q` is good (``PLAN.md``).

    Good: `s > s_1` and `\delta = s/2 - q \le b\sqrt s`, or `s \le s_1` and `\delta < 1`
    (a perfect split). Decided exactly, `b` rational.

    EXAMPLES::

        >>> good_fork(16, 7, Fraction(2, 5), 3), good_fork(16, 7, Fraction(1, 10), 3)
        (True, False)
        >>> good_fork(5, 2, Fraction(1, 10), 5), good_fork(5, 1, Fraction(1, 10), 5)
        (True, False)
    """
    if q < 1:
        return False
    if s <= s1:
        return q == s // 2
    return (s - 2 * q) ** 2 <= 4 * b * b * s


def min_good_quality(s, b, s1):
    r"""
    Return the least quality `q \ge 1` for which a fork of size `s` is good, or
    `\lfloor s/2\rfloor + 1` if there is none (an odd `s > s_1` with `b\sqrt s < 1/2`).

    EXAMPLES::

        >>> [min_good_quality(s, Fraction(2, 5), 3) for s in (2, 3, 4, 8, 16)]
        [1, 1, 2, 3, 7]
        >>> min_good_quality(5, Fraction(1, 10), 3)
        3
    """
    return min((q for q in range(1, s // 2 + 1) if good_fork(s, q, b, s1)), default=s // 2 + 1)


def _v(q, s):
    return Fraction((s - 2 * q) ** 2, s)


def fork_records(classes, n, b, s1):
    r"""
    Return one record per class with a good prefix (forks `0..i-1` good) and `s_i \ge 2`.

    Each record holds ``i``, ``sizes``, ``rows``, ``W`` and the exact fractions

    - ``bad``: `P(\text{fork } i \text{ not good} \mid H_i)` (a stop counts as not good);
    - ``p``: for each unchosen row, `P(q_r(S_i) \ge \kappa \mid H_i)`, `\kappa` the least
      good quality; for `s_i > s_1` this is the good-window probability
      `P(|e_r(S_i)| \le b\sqrt{s_i} \mid H_i)` that `c_1` bounds below;
    - ``prod``: `\prod_r (1 - p_r)`, which equals ``bad`` by Step 1 (``cor:product``);
    - ``M``: for `i \ge 1`, each unchosen row's `E[v_r(S_{i-1}) \mid H_i]`, the filtered
      second moment `M_{r,i-1}` of Step 3;
    - ``N``: each unchosen row's `E[v_r(S_i) \mid H_i]`, the unfiltered moment `N_{r,i}`.

    By symmetry every history in a class has the same conditional law, so these are the
    pointwise values given `H_i = h` for each `h` in the class.

    EXAMPLES:

    The product identity of Step 1, here on all matrices with `n = 3`, `m = 5`::

        >>> recs = fork_records(brute_classes(3, 5), 3, Fraction(2, 5), 3)
        >>> all(r["bad"] == r["prod"] for r in recs)
        True
        >>> r0 = [r for r in recs if r["i"] == 0][0]
        >>> r0["bad"], r0["p"]
        (Fraction(27, 512), [Fraction(5, 8), Fraction(5, 8), Fraction(5, 8)])
    """
    out = []
    for (sizes, rows), rec in classes.items():
        i = len(rows)
        q = _chain_q(sizes)
        if not all(good_fork(sizes[l], q[l], b, s1) for l in range(i)):
            continue
        s = sizes[-1]
        W = rec["W"]
        kappa = min_good_quality(s, b, s1)
        bad = Fraction(sum(rec["qh"][:kappa]), W)
        p = [Fraction(sum(rec["kh"][r][kappa:]), W) for r in sorted(rec["kh"])]
        prod = Fraction(1)
        for x in p:
            prod *= 1 - x
        M = []
        if i:
            ps = sizes[-2]
            M = [sum(_v(qq, ps) * c for qq, c in enumerate(rec["kp"][r])) / W for r in sorted(rec["kp"])]
        N = [sum(_v(qq, s) * c for qq, c in enumerate(rec["kh"][r])) / W for r in sorted(rec["kh"])]
        out.append({"i": i, "sizes": sizes, "rows": rows, "W": W, "bad": bad, "p": p, "prod": prod, "M": M, "N": N})
    return out


def depth_summary(records, n, m, s1):
    r"""
    Summarise :func:`fork_records` by depth `i` and by fork type (large `s_i > s_1`, or small).

    Returns a list of dicts with keys ``i``, ``large``, ``weight`` (the probability that
    forks `0..i-1` are good and `S_i` has the type given), ``bad_avg``
    (`P(\text{fork } i \text{ not good})` given that event), ``bad_max`` (largest over
    histories), ``pmin`` (least per-row probability over histories and rows),
    ``prod_bound_max`` (largest `(1 - \min_r p_r)^{n-i}` over histories), ``M_max``
    (largest filtered moment `M_{r,i-1}`, only when forks `0..i-1` are all large) and
    ``classes``.

    EXAMPLES::

        >>> recs = fork_records(brute_classes(3, 5), 3, Fraction(2, 5), 3)
        >>> s = depth_summary(recs, 3, 5, 3)
        >>> s[0]["i"], s[0]["large"], s[0]["bad_avg"]
        (0, True, Fraction(27, 512))
    """
    groups = {}
    for r in records:
        groups.setdefault((r["i"], r["sizes"][-1] > s1), []).append(r)
    out = []
    total = 2 ** (n * m)
    for (i, large), rs in sorted(groups.items()):
        w = sum(r["W"] for r in rs)
        ps = [x for r in rs for x in r["p"]]
        Ms = [x for r in rs if all(sz > s1 for sz in r["sizes"][:-1]) for x in r["M"]]
        out.append({
            "i": i, "large": large, "weight": Fraction(w, total),
            "bad_avg": sum(r["bad"] * r["W"] for r in rs) / w,
            "bad_max": max(r["bad"] for r in rs),
            "pmin": min(ps) if ps else None,
            "prod_bound_max": max((1 - min(r["p"])) ** (n - i) if r["p"] else Fraction(1) for r in rs),
            "M_max": max(Ms) if Ms else None,
            "classes": len(rs),
        })
    return out


# ---------------------------------------------------------------- the same, chain by chain


def good_chains(m, n, b, s1):
    r"""
    Yield the chains `m = s_0 > \dots > s_i` with `i \le n` whose forks `0..i-1` are good.

    The last size may be `1` (then the path has ended with good forks only).

    EXAMPLES::

        >>> list(good_chains(5, 3, Fraction(0), 5))
        [(5,), (5, 3), (5, 3, 2), (5, 3, 2, 1), (5, 3, 1), (5, 2), (5, 2, 1)]
    """
    def go(sizes):
        yield sizes
        if len(sizes) - 1 >= n or sizes[-1] < 2:
            return
        s = sizes[-1]
        for t in range(s - 1, 0, -1):
            if good_fork(s, min(t, s - t), b, s1):
                yield from go(sizes + (t,))

    yield from go((m,))


def _cons(sizes, strict):
    q = _chain_q(sizes)
    return tuple((q[l], strict[l]) for l in range(len(strict)))


def chain_aggregate(n, sizes, kappa):
    r"""
    Return ``(W, W_bad)`` for the chain ``sizes`` (`s_i \ge 2`), summed over all row maps
    `\rho`: the number of matrices with this path of column 1, and of those with no row of
    quality `\ge \kappa` at `S_i`.

    The sum over `\rho` is a dynamic programme over the rows in index order, with state the
    set of levels whose row has been seen (a row is strict at level `l` exactly when the row
    of level `l` comes later), as in the article's path formula; by Step 1 the rows
    contribute independent factors.

    EXAMPLES::

        >>> cl = brute_classes(3, 5)
        >>> W = sum(v["W"] for (s, r), v in cl.items() if s == (5, 3))
        >>> chain_aggregate(3, (5, 3), 1)[0] == W
        True
    """
    i = len(sizes) - 1
    s = sizes[-1]
    mult = math.prod(2 * comb(sizes[l] - 1, sizes[l + 1] - 1) for l in range(i))
    full = (1 << i) - 1
    free, freebad, chosen = {}, {}, {}
    for mask in range(full + 1):
        strict = tuple(not (mask >> l) & 1 for l in range(i))
        tab = row_table(tuple(sizes), _cons(sizes, strict), (i,))
        free[mask] = sum(tab.values())
        freebad[mask] = sum(w for (qq,), w in tab.items() if qq < kappa)
        for j in range(i):
            if not (mask >> j) & 1:
                chosen[mask, j] = sum(row_table(tuple(sizes[: j + 2]), _cons(sizes[: j + 2], strict[:j]), (), True).values())
    res = []
    for fw in (free, freebad):
        f = {0: 1}
        for _ in range(n):
            g = {}
            for mask, val in f.items():
                g[mask] = g.get(mask, 0) + val * fw[mask]
                for j in range(i):
                    if not (mask >> j) & 1 and chosen[mask, j]:
                        nm = mask | 1 << j
                        g[nm] = g.get(nm, 0) + val * chosen[mask, j]
            f = g
        res.append(mult * f.get(full, 0))
    return tuple(res)


def chain_pointwise(n, sizes, b, s1):
    r"""
    Return the worst cases over the histories of the chain ``sizes`` (`s_i \ge 2`, `i < n`).

    A history fixes the order of the rows `\rho_0, \dots, \rho_{i-1}`; an unchosen row's
    law depends only on its tie pattern (on which side of each `\rho_l` it lies). Every
    tie pattern compatible with a feasible order is realised with all `n - i` unchosen rows
    in one gap. Returns a dict with ``pmin`` (least `P(q_r(S_i) \ge \kappa \mid H_i)`),
    ``bad_max`` (largest `P(\text{fork } i\text{ not good} \mid H_i) = (1-p_{\min})^{n-i}`,
    by Step 1's product identity), ``M_max`` (largest `M_{r,i-1}`, ``None`` for `i = 0`)
    and ``N_max`` (largest `N_{r,i}`). Returns ``None`` if no order is feasible.

    EXAMPLES::

        >>> recs = fork_records(brute_classes(3, 5), 3, Fraction(2, 5), 3)
        >>> d = chain_pointwise(3, (5, 3), Fraction(2, 5), 3)
        >>> d["pmin"] == min(x for r in recs if r["sizes"] == (5, 3) for x in r["p"])
        True
    """
    i = len(sizes) - 1
    s = sizes[-1]
    kappa = min_good_quality(s, b, s1)
    best = None
    for perm in itertools.permutations(range(i)):
        # perm[l]: rank of rho_l among the chosen rows
        ok = True
        for j in range(i):
            strict = tuple(perm[j] < perm[t] for t in range(j))
            if not sum(row_table(tuple(sizes[: j + 2]), _cons(sizes[: j + 2], strict), (), True).values()):
                ok = False
                break
        if not ok:
            continue
        for g in range(i + 1):
            strict = tuple(not perm[l] < g for l in range(i))
            tab = row_table(tuple(sizes), _cons(sizes, strict), (i, i - 1) if i else (0,))
            tot = sum(tab.values())
            p = Fraction(sum(w for key, w in tab.items() if key[0] >= kappa), tot)
            N = sum(_v(key[0], s) * w for key, w in tab.items()) / tot
            M = sum(_v(key[1], sizes[-2]) * w for key, w in tab.items()) / tot if i else None
            if best is None:
                best = {"pmin": p, "M_max": M, "N_max": N}
            else:
                best["pmin"] = min(best["pmin"], p)
                best["N_max"] = max(best["N_max"], N)
                if M is not None:
                    best["M_max"] = max(best["M_max"], M)
    if best is not None:
        best["bad_max"] = (1 - best["pmin"]) ** (n - i)
    return best


def chain_summary(n, m, b, s1):
    r"""
    The analogue of :func:`depth_summary` computed chain by chain (no enumeration of
    matrices or of row maps), for depths `i < n`.

    EXAMPLES:

    It agrees with the summary of all matrices::

        >>> b, s1 = Fraction(2, 5), 3
        >>> a = depth_summary(fork_records(brute_classes(3, 5), 3, b, s1), 3, 5, s1)
        >>> c = chain_summary(3, 5, b, s1)
        >>> keys = ("i", "large", "weight", "bad_avg", "bad_max", "pmin", "M_max")
        >>> [{k: r[k] for k in keys} for r in a if r["i"] < 3] == [{k: r[k] for k in keys} for r in c]
        True
    """
    groups = {}
    for sizes in good_chains(m, n, b, s1):
        i = len(sizes) - 1
        if sizes[-1] < 2 or i >= n:
            continue
        kappa = min_good_quality(sizes[-1], b, s1)
        W, Wb = chain_aggregate(n, sizes, kappa)
        if W == 0:
            continue
        pw = chain_pointwise(n, sizes, b, s1)
        allbig = all(x > s1 for x in sizes[:-1])
        g = groups.setdefault((i, sizes[-1] > s1), {"W": 0, "Wb": 0, "bad_max": None, "pmin": None, "M_max": None, "chains": 0})
        g["W"] += W
        g["Wb"] += Wb
        g["chains"] += 1
        g["bad_max"] = pw["bad_max"] if g["bad_max"] is None else max(g["bad_max"], pw["bad_max"])
        g["pmin"] = pw["pmin"] if g["pmin"] is None else min(g["pmin"], pw["pmin"])
        if allbig and pw["M_max"] is not None:
            g["M_max"] = pw["M_max"] if g["M_max"] is None else max(g["M_max"], pw["M_max"])
    out = []
    for (i, large), g in sorted(groups.items()):
        out.append({"i": i, "large": large, "weight": Fraction(g["W"], 2 ** (n * m)), "bad_avg": Fraction(g["Wb"], g["W"]),
                    "bad_max": g["bad_max"], "pmin": g["pmin"], "prod_bound_max": g["bad_max"], "M_max": g["M_max"],
                    "chains": g["chains"]})
    return out


# ---------------------------------------------------------------- Step 5 with m <= s_1


def step5_failure(n, m):
    r"""
    Return `\mathbf P(F)` exactly for `s_1 \ge m` (so `i_0 = 0`): the probability that some
    set `S_i` with `s_i \ge 2` on the path of column 1 is not a perfect fork.

    Uses :func:`chain_aggregate` on the perfect chains.

    EXAMPLES:

    For `m = 2, 3` every split is perfect, so only a stop fails::

        >>> step5_failure(3, 2) == Fraction(1, 2**3)
        True
        >>> recs = fork_records(brute_classes(3, 5), 3, 0, 5)
        >>> step5_failure(3, 5) == sum(r["bad"] * r["W"] for r in recs) / 2**15
        True
    """
    tot = Fraction(0)
    for sizes in good_chains(m, n, Fraction(0), m):
        if sizes[-1] >= 2:
            tot += Fraction(chain_aggregate(n, sizes, sizes[-1] // 2)[1], 2 ** (n * m))
    return tot


def perfect_min(m):
    r"""
    Return `\min_{2 \le u \le m}` of the probability that a uniform row is perfect on `u` columns.

    EXAMPLES::

        >>> [perfect_min(m) for m in (2, 3, 4, 8)]
        [Fraction(1, 2), Fraction(1, 2), Fraction(3, 8), Fraction(35, 128)]
    """
    return min(Fraction(2 * comb(u, u // 2) if u % 2 else comb(u, u // 2), 2**u) for u in range(2, m + 1))


def step5_union_bound(n, m, pi=None):
    r"""
    Return the bound `(2^{m-1}-1)(1-\pi)^n` of Step 5 (``thm:fail``, ``eq:union``) on
    `\mathbf P(F)` when `i_0 = 0`. By default `\pi` is :func:`perfect_min`; Step 5's
    Lemma ``lem:p``(c) states `\pi = 1/(m+1)`.

    EXAMPLES::

        >>> step5_union_bound(10, 4) == 7 * Fraction(5, 8) ** 10
        True
        >>> step5_failure(10, 4) <= step5_union_bound(10, 4)
        True
    """
    pi = perfect_min(m) if pi is None else pi
    return (2 ** (m - 1) - 1) * (1 - pi) ** n


def step5_grouped_bound(n, m, pi=None):
    r"""
    Return the grouped bound `\ell(1-\pi)^{\lfloor n/\ell\rfloor}`, `\ell = \lceil\log m\rceil`
    (``eq:grouped`` of Step 5 with `i_0 = 0`).

    EXAMPLES::

        >>> step5_grouped_bound(12, 4) == 2 * Fraction(5, 8) ** 6
        True
    """
    pi = perfect_min(m) if pi is None else pi
    ell = (m - 1).bit_length()
    return ell * (1 - pi) ** (n // ell)


def path_budget(m, n=None):
    r"""
    Return `B_n(m)`, the sum of the `\min(n, m-1)` largest of `1 - h(1/s)`, `2 \le s \le m`,
    as an ``mpf`` (all of them if ``n`` is ``None``). On `D` every fork on a path splits
    (`q \ge 1`, so `T \le 1 - h(1/s)`), the sizes strictly decrease, and a path has at most
    `\min(n, m-1)` forks (Step 1, ``lem:used``); so `\sum_i T_i \le B_n(m)`.

    EXAMPLES::

        >>> float(path_budget(2)), round(float(path_budget(3)), 6)
        (0.0, 0.081704)
        >>> path_budget(10, 3) < path_budget(10)
        True
    """
    terms = []
    for s in range(2, m + 1):
        x = mpmath.mpf(1) / s
        terms.append(1 + x * mpmath.log(x, 2) + (1 - x) * mpmath.log(1 - x, 2))
    terms.sort(reverse=True)
    return mpmath.fsum(terms[: (m - 1) if n is None else min(n, m - 1)])


def _mp(x):
    return mpmath.mpf(x.numerator) / x.denominator


def step5_assembled(n, m, qf):
    r"""
    Return `(f(m) + B(m)\,q_F)/\mathbf P(D)`, a bound on `t_{n,m}(G) - \log m` for `m \le s_1`.

    Here `f(m) = C(m) - \log m`, `B = B_n(m)` is :func:`path_budget` and `q_F \ge \mathbf P(F)`.
    By Step 1 (``lem:transfer``) and Step 5 (``cor:path``, with `i_0 = 0`),
    `E[\Sigma_1 1_D] \le E[\Sigma_1 1_{F^c}] + B\,\mathbf P(F) \le f(m) + B\,\mathbf P(F)`.

    EXAMPLES::

        >>> float(step5_assembled(4, 3, step5_failure(4, 3))) >= 5 / 3 - math.log2(3)
        True
    """
    f = _mp(complete_cost(m)) - mpmath.log(m, 2)
    return (f + path_budget(m, n) * _mp(qf)) / _mp(distinct_probability(n, m))


# ---------------------------------------------------------------- Step 3 limit constants


def m_star_limit(b):
    r"""
    Return `M^*_\infty(b) = 1/(1 - 8b/\sqrt\pi)` of Step 3 (``cor:filter-limit``), or ``inf``.

    EXAMPLES::

        >>> round(float(m_star_limit(0.1)), 4)
        1.8227
    """
    eta = 4 * mpmath.mpf(b) / mpmath.sqrt(mpmath.pi)
    return mpmath.inf if eta >= 0.5 else 1 / (1 - 2 * eta)


def c1_limit(b):
    r"""
    Return `c_1^\infty(b) = (4b/\sqrt\pi)\exp(-M^*_\infty/2 - 4b^2/3)` of Step 3, or 0.

    EXAMPLES:

    The values stated in Step 3::

        >>> round(float(c1_limit(0.1)), 4), round(float(c1_limit(0.05)), 3)
        (0.0895, 0.059)
        >>> c1_limit(0.4)
        mpf('0.0')
    """
    b = mpmath.mpf(b)
    ms = m_star_limit(b)
    if ms == mpmath.inf:
        return mpmath.mpf(0)
    return 4 * b / mpmath.sqrt(mpmath.pi) * mpmath.exp(-ms / 2 - 4 * b * b / 3)


# ---------------------------------------------------------------- the article's bounds


@lru_cache(maxsize=None)
def _rho_j(m, j):
    kap = [0, 0] + [max(1, s // 2 - j) for s in range(2, m + 1)]
    return rho_upper(kap, m)[m]


def prop_union_best(n, m):
    r"""
    Return the best bound from the article's Proposition ``prop:union`` in the form its
    proof gives for `m \le s_1`: `\rho_\kappa(m) + R_{\max}(m)P_{\mathrm{col}}/\mathbf P(D)`
    on `t - \log m`, minimised over `\kappa_j(s) = \max(1, \lfloor s/2\rfloor - j)`.
    Returns ``(bound, j)``; the bound is an upper bound (``rho_upper`` uses interval
    arithmetic; `P_{\mathrm{col}}` and `\mathbf P(D)` are exact).

    The statement of the proposition has in addition `\max_{s\le s_1}` and the term
    `4x/(s_1(1-\theta))`, which its proof does not use when `m \le s_1`.

    EXAMPLES::

        >>> b, j = prop_union_best(10, 4)
        >>> j, float(b) < 0.03
        (0, True)
    """
    qd = _mp(distinct_probability(n, m))
    rmax = mpmath.mpf(m + 1) / 2 - mpmath.mpf(1) / m - mpmath.log(m, 2)
    best = None
    for j in range(max(1, m // 2)):
        kap = [0, 0] + [max(1, s // 2 - j) for s in range(2, m + 1)]
        rho = _rho_j(m, j)
        pcol = Fraction(0)
        for s in range(4, m + 1):
            k = kap[s]
            if k >= 2:
                pcol += comb(m, s) * Fraction(2 * sum(comb(s, x) for x in range(k)), 2**s) ** n
        val = rho + rmax * _mp(pcol) / qd
        if best is None or val < best[0]:
            best = (val, j)
    return best


def load_formula(data_dir):
    r"""
    Return ``{(n, m): t_{n,m}(G)}`` from ``exact_formula.csv`` in ``data_dir``.

    EXAMPLES::

        >>> d = Path(__file__).resolve().parents[3] / "data" / "gradient-bound"
        >>> load_formula(d)[3, 4]
        Fraction(71, 35)
    """
    with open(Path(data_dir) / "exact_formula.csv", newline="") as f:
        return {(int(r["n"]), int(r["m"])): Fraction(int(r["numerator"]), int(r["denominator"])) for r in csv.DictReader(f)}


def nstr(x, digits=6):
    r"""
    Format an exact or ``mpf`` value with ``digits`` significant digits (empty for ``None``).

    EXAMPLES::

        >>> nstr(Fraction(1, 3), 4), nstr(None)
        ('0.3333', '')
    """
    if x is None:
        return ""
    if isinstance(x, (Fraction, int)):
        x = _mp(Fraction(x))
    return mpmath.nstr(x, digits, min_fixed=-30, max_fixed=30)


# ---------------------------------------------------------------- data files


def _frac(x):
    return "" if x is None else f"{Fraction(x).numerator}/{Fraction(x).denominator}"


def exact_rows(data_dir):
    r"""
    Rows of ``exact_vs_bounds.csv``: for every cell of ``exact_formula.csv`` with `m \ge 4`,
    the exact value, the bounds of the article and of Step 5 (with `s_1 = m`), and the
    regime quantities of Theorems A and B. Exact values are kept as ``num/den``.

    EXAMPLES::

        >>> d = Path(__file__).resolve().parents[3] / "data" / "gradient-bound"
        >>> r = [x for x in exact_rows(d) if (x["n"], x["m"]) == (10, 4)][0]
        >>> r["excess"], r["thm_ratio"][:6]
        ('35680/16173661', '1.0367')
    """
    t = load_formula(data_dir)
    rows = []
    for (n, m), v in sorted(t.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if m < 4:
            continue
        c = complete_cost(m)
        ex = v - c
        thm = theorem1_excess_bound(n, m)
        lam = mpmath.log(m, 2)
        qf = step5_failure(n, m)
        bounds = [min(step5_union_bound(n, m, pi), step5_grouped_bound(n, m, pi)) for pi in (None, Fraction(1, m + 1))]
        pu, j = prop_union_best(n, m)
        rows.append({
            "n": n, "m": m, "t": _frac(v), "excess": _frac(ex),
            "t_minus_log": nstr(_mp(v) - lam, 10), "t_minus_C": nstr(ex, 10),
            "thm_bound": nstr(thm, 10), "thm_ratio": nstr(thm / ex, 6) if ex else "",
            "prop_bound": nstr(pu, 10), "prop_j": j,
            "QF": _frac(qf), "QF_dec": nstr(qf, 10),
            "QF_bound_pi_exact": nstr(bounds[0], 10), "QF_bound_pi_lemma": nstr(bounds[1], 10),
            "step5_exact_QF": nstr(step5_assembled(n, m, qf), 10),
            "step5_bound_QF": nstr(step5_assembled(n, m, min(Fraction(1), bounds[0])), 10),
            "n_over_log": nstr(n / lam, 6),
            "thmB_ratio": nstr((n - lam) / mpmath.log(n, 2) ** 2, 6) if n > 1 else "",
            "many_rows": int(n * n >= 2 * m**3),
        })
    return rows


FORK_CELLS = [(n, m) for m in (8, 12, 16, 24, 32) for n in (6, 8, 10, 12, 16, 20, 24) if 2**n >= m]


def fork_rows(cells=None, bs=None):
    r"""
    Rows of ``forks.csv``: :func:`chain_summary` for each cell and each `(b, s_1)`.

    EXAMPLES::

        >>> r = fork_rows([(8, 8)], [(Fraction(1, 5), 5)])
        >>> [(x["i"], x["large"], x["bad_avg"]) for x in r][:1]
        [(0, 1, '5595818096650401/72057594037927936')]
    """
    cells = FORK_CELLS if cells is None else cells
    bs = ENUM_BS if bs is None else bs
    out = []
    for b, s1 in bs:
        c1 = c1_limit(b)
        for n, m in cells:
            for row in chain_summary(n, m, b, s1):
                out.append({
                    "b": str(b), "s1": s1, "n": n, "m": m, "i": row["i"], "large": int(row["large"]),
                    "weight": _frac(row["weight"]), "bad_avg": _frac(row["bad_avg"]), "bad_max": _frac(row["bad_max"]),
                    "pmin": _frac(row["pmin"]), "M_max": _frac(row["M_max"]),
                    "bad_avg_dec": nstr(row["bad_avg"], 6), "bad_max_dec": nstr(row["bad_max"], 6),
                    "pmin_dec": nstr(row["pmin"], 6), "M_max_dec": nstr(row["M_max"], 6),
                    "c1lim_power": nstr((1 - c1) ** (n - row["i"]), 6) if row["large"] else "",
                    "chains": row["chains"],
                })
    return out


def enum_rows(raw_dir, cells=None):
    r"""
    Rows of ``enum_checks.csv`` from the gzipped engine output ``enum_<n>_<m>.txt.gz``.

    For each cell: the totals, the number of classes, and whether the three checks hold:
    the product formula of Step 1 gives the same class data (``product``), the product
    identity `P(\text{not good}\mid H_i) = \prod_r(1-p_r)` holds for every class and every
    `(b, s_1)` of ``ENUM_BS`` and `(0, m)` (``identity``), the chain-by-chain summary
    equals the summary of the enumeration (``chains``), and the exact moment recursion
    `N_{r,i} = 1 + \frac{s_i-1}{s_{i-1}-1}(M_{r,i-1}-1)` of Step 3 holds for every class
    and unchosen row (``recursion``).
    """
    import gzip  # noqa: PLC0415

    cells = ENUM_CELLS if cells is None else cells
    out = []
    keys = ("i", "large", "weight", "bad_avg", "bad_max", "pmin", "M_max")
    for n, m in cells:
        cl = parse_enum(gzip.open(Path(raw_dir) / f"enum_{n}_{m}.txt.gz", "rt").read())
        tot = sum(v["W"] for (s, r), v in cl.items() if len(s) == 1)
        prod_ok = product_classes(n, m) == cl
        ident = chains = True
        for b, s1 in ENUM_BS + [(Fraction(0), m)]:
            recs = fork_records(cl, n, b, s1)
            ident &= all(r["bad"] == r["prod"] for r in recs)
            a = [{k: r[k] for k in keys} for r in depth_summary(recs, n, m, s1) if r["i"] < n]
            c = [{k: r[k] for k in keys} for r in chain_summary(n, m, b, s1)]
            chains &= a == c
        rec_ok = True
        for (sizes, rows), rec in cl.items():
            if not rows:
                continue
            s, ps = sizes[-1], sizes[-2]
            for r in rec["kh"]:
                N = sum(_v(q, s) * c for q, c in enumerate(rec["kh"][r]))
                M = sum(_v(q, ps) * c for q, c in enumerate(rec["kp"][r]))
                W = rec["W"]
                rec_ok &= N * (ps - 1) == W * (ps - 1) + (s - 1) * (M - W)
        out.append({"n": n, "m": m, "matrices_log2": n * m, "total_ok": int(tot == 2 ** (n * m)),
                    "multisets": comb(2**n + m - 2, m - 1), "classes": len(cl), "product": int(prod_ok),
                    "identity": int(ident), "chains": int(chains), "recursion": int(rec_ok)})
    return out


def step5_rows(ms=range(4, 17), ns=(4, 6, 8, 10, 12, 16, 20, 24)):
    r"""
    Rows of ``step5.csv``: the exact `\mathbf P(F)` for `s_1 \ge m` against Step 5's bounds.

    EXAMPLES::

        >>> r = step5_rows([4], [10])[0]
        >>> r["QF"], r["union_exact_pi"]
        ('6367103/536870912', '0.0636646')
    """
    out = []
    for m in ms:
        for n in ns:
            if 2**n < m:
                continue
            qf = step5_failure(n, m)
            pe, pl = perfect_min(m), Fraction(1, m + 1)
            out.append({"m": m, "n": n, "QF": _frac(qf), "QF_dec": nstr(qf, 6), "pi_exact": _frac(pe),
                        "union_exact_pi": nstr(step5_union_bound(n, m, pe), 6),
                        "grouped_exact_pi": nstr(step5_grouped_bound(n, m, pe), 6),
                        "union_lemma_pi": nstr(step5_union_bound(n, m, pl), 6),
                        "grouped_lemma_pi": nstr(step5_grouped_bound(n, m, pl), 6)})
    return out


def write_csv(path, rows):
    """Write a list of dicts as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    """Read a CSV file as a list of dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_data(data_dir, raw_dir):
    r"""Write ``exact_vs_bounds.csv``, ``forks.csv``, ``step5.csv`` and ``enum_checks.csv``
    into ``data_dir/analytic``; ``raw_dir`` holds the gzipped enumerations."""
    out = Path(data_dir) / "analytic"
    write_csv(out / "exact_vs_bounds.csv", exact_rows(data_dir))
    write_csv(out / "forks.csv", fork_rows())
    write_csv(out / "step5.csv", step5_rows())
    write_csv(out / "enum_checks.csv", enum_rows(raw_dir))


# ---------------------------------------------------------------- the TeX fragment


def tex_num(x, digits=3, up=False):
    r"""
    Format a decimal string or number for TeX with ``digits`` significant digits;
    ``up`` rounds upwards (for upper bounds). Small and large values get a power of ten.

    EXAMPLES::

        >>> tex_num("0.0123456"), tex_num("0.0123456", up=True), tex_num("123456.7")
        ('0.0123', '0.0124', '$1.23\\cdot10^{5}$')
        >>> tex_num("0.000012"), tex_num("0"), tex_num("")
        ('$1.20\\cdot10^{-5}$', '0', '')
    """
    if x == "" or x is None:
        return ""
    v = mpmath.mpf(x)
    if v == 0:
        return "0"
    e = int(mpmath.floor(mpmath.log10(abs(v))))
    scale = mpmath.mpf(10) ** (digits - 1 - e)
    r = (mpmath.ceil(v * scale) if up else mpmath.nint(v * scale)) / scale
    e = int(mpmath.floor(mpmath.log10(abs(r))))
    if -3 <= e <= 3:
        return f"{float(r):.{max(0, digits - 1 - e)}f}"
    mant = r / mpmath.mpf(10) ** e
    return f"${float(mant):.{digits - 1}f}\\cdot10^{{{e}}}$"


VALUE_MS = (4, 6, 8, 11, 12, 16, 20, 23, 24, 28, 32)


def _value_table(rows):
    by = {(int(r["n"]), int(r["m"])): r for r in rows}
    lines = [r"\begin{tabular}{rrrlllll}", r"\toprule",
             r"$m$ & $n$ & $\frac n{\log m}$ & $t-\log m$ & $t-C(m)$ & Thm.~4.1 & Prop.~5.2 & Step 5\\",
             r"\midrule"]
    for m in VALUE_MS:
        ns = sorted(n for (n, mm) in by if mm == m)
        lam = math.log2(m)
        pick = sorted({n for n in (math.ceil(lam), math.ceil(2 * lam), math.ceil(3 * lam), ns[-1]) if n in ns})
        for n in pick:
            r = by[n, m]
            lines.append(f"{m} & {n} & {float(r['n_over_log']):.2f} & {tex_num(r['t_minus_log'])} & {tex_num(r['t_minus_C'])} & "
                         f"{tex_num(r['thm_bound'], up=True)} & {tex_num(r['prop_bound'], up=True)} & "
                         f"{tex_num(r['step5_bound_QF'], up=True)}\\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _step5_table(rows):
    lines = [r"\begin{tabular}{rrlllll}", r"\toprule",
             r"$m$ & $n$ & $\mathbf P(F)$ & union, $\pi_m$ & grouped, $\pi_m$ & "
             r"union, $\frac1{m+1}$ & grouped, $\frac1{m+1}$\\", r"\midrule"]
    for r in rows:
        m, n = int(r["m"]), int(r["n"])
        if m in (4, 6, 8, 12, 16) and n in (8, 12, 16, 24):
            lines.append(f"{m} & {n} & {tex_num(r['QF_dec'])} & {tex_num(r['union_exact_pi'], up=True)} & "
                         f"{tex_num(r['grouped_exact_pi'], up=True)} & {tex_num(r['union_lemma_pi'], up=True)} & "
                         f"{tex_num(r['grouped_lemma_pi'], up=True)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _fork_table(rows, b, s1, m, ns):
    lines = [r"\begin{tabular}{rrclllll}", r"\toprule",
             r"$n$ & $i$ & type & weight & not good & worst $h$ & $p_{\min}$ & $M_{\max}$\\", r"\midrule"]
    for n in ns:
        for r in rows:
            if (r["b"], int(r["s1"]), int(r["m"]), int(r["n"])) != (b, s1, m, n):
                continue
            w = Fraction(r["weight"])
            lines.append(f"{n} & {r['i']} & {'L' if r['large'] == '1' else 'S'} & {tex_num(nstr(w, 8))} & "
                         f"{tex_num(r['bad_avg_dec'])} & {tex_num(r['bad_max_dec'])} & {tex_num(r['pmin_dec'])} & "
                         f"{tex_num(r['M_max_dec'])}\\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _enum_table(rows):
    lines = [r"\begin{tabular}{rrrrrc}", r"\toprule",
             r"$n$ & $m$ & $nm$ & matrices up to symmetry & classes & checks\\", r"\midrule"]
    for r in sorted(rows, key=lambda r: (int(r["m"]), int(r["n"]))):
        ok = all(r[k] == "1" for k in ("total_ok", "product", "identity", "chains", "recursion"))
        lines.append(f"{r['n']} & {r['m']} & {r['matrices_log2']} & {int(r['multisets']):,} & {r['classes']} & "
                     f"{'all hold' if ok else 'FAIL'}\\\\".replace(",", "\\,"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tex_facts(analytic_dir, formula_dir=None):
    r"""
    Return the dict of macros of the fragment (name to TeX text), from the CSV files in
    ``analytic_dir`` (and ``exact_formula.csv`` one level up).

    EXAMPLES::

        >>> d = Path(__file__).resolve().parents[3] / "data" / "gradient-bound" / "analytic"
        >>> f = tex_facts(d)
        >>> f["nineMaxRedAt"]
        '(6,23)'
    """
    d = Path(analytic_dir)
    formula_dir = d.parent if formula_dir is None else Path(formula_dir)
    ev = read_csv(d / "exact_vs_bounds.csv")
    allc = load_formula(formula_dir)
    fl = lambda r, k: float(r[k])  # noqa: E731
    mx = max(ev, key=lambda r: fl(r, "t_minus_log"))
    f = {
        "nineAllCells": str(len(allc)),
        "nineNewCells": str(len(allc) - 400),
        "nineCells": str(len(ev)),
        "nineMaxRed": tex_num(mx["t_minus_log"], 4, up=True),
        "nineMaxRedAt": f"({mx['n']},{mx['m']})",
        "nineThmBmax": tex_num(str(max(fl(r, "thmB_ratio") for r in ev)), 3, up=True),
        "nineThmGood": str(sum(fl(r, "thm_bound") <= 0.01 for r in ev)),
        "nineThmGoodMmax": str(max(int(r["m"]) for r in ev if fl(r, "thm_bound") <= 0.01)),
        "ninePropGood": str(sum(fl(r, "prop_bound") <= 0.1 for r in ev)),
        "ninePropGoodMmax": str(max(int(r["m"]) for r in ev if fl(r, "prop_bound") <= 0.1)),
        "nineStepGood": str(sum(fl(r, "step5_bound_QF") <= 0.1 for r in ev)),
        "nineStepGoodMmax": str(max(int(r["m"]) for r in ev if fl(r, "step5_bound_QF") <= 0.1)),
        "nineStepExactGood": str(sum(fl(r, "step5_exact_QF") <= 0.1 for r in ev)),
        "nineStepExactGoodMmax": str(max(int(r["m"]) for r in ev if fl(r, "step5_exact_QF") <= 0.1)),
        "nineAbove": str(sum(fl(r, "n_over_log") > 2 for r in ev)),
        "nineCOneFifth": tex_num(nstr(c1_limit(Fraction(1, 5)), 10), 2),
        "nineMStarFifth": tex_num(nstr(m_star_limit(Fraction(1, 5)), 10), 3, up=True),
        "nineCOneTenth": tex_num(nstr(c1_limit(Fraction(1, 10)), 10), 3),
    }
    ec = read_csv(d / "enum_checks.csv")
    f["nineEnumCells"] = str(len(ec))
    f["nineEnumMaxLog"] = str(max(int(r["matrices_log2"]) for r in ec))
    f["nineEnumAllOk"] = "yes" if all(all(r[k] == "1" for k in ("total_ok", "product", "identity", "chains", "recursion"))
                                      for r in ec) else "NO"
    fo = read_csv(d / "forks.csv")
    big = [r for r in fo if r["large"] == "1" and r["pmin"]]
    for b, name in (("1/5", "Fifth"), ("2/5", "TwoFifths")):
        ps = [Fraction(r["pmin"]) for r in big if r["b"] == b]
        Ms = [Fraction(r["M_max"]) for r in fo if r["b"] == b and r["M_max"]]
        f[f"ninePminLarge{name}"] = tex_num(nstr(min(ps), 10), 3)
        f[f"nineMmax{name}"] = tex_num(nstr(max(Ms), 10), 3, up=True)
        if b == "1/5":
            f["nineCOneRatio"] = str(int(_mp(min(ps)) / c1_limit(Fraction(1, 5))))
            f["nineMRatio"] = tex_num(nstr(m_star_limit(Fraction(1, 5)) / _mp(max(Ms)), 10), 2)
    return f


def write_tex(analytic_dir, path):
    r"""Write ``09-tables.tex``: the macros of :func:`tex_facts` and the four tables."""
    d = Path(analytic_dir)
    lines = ["% Generated by keytrees/scripts/gradient_analytic_9_data.py tex; do not edit.",
             "% Data: keytrees/data/gradient-bound/analytic/ and exact_formula.csv."]
    for k, v in tex_facts(d).items():
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
    fo = read_csv(d / "forks.csv")
    lines.append(r"\newcommand{\nineTableValues}{%" + "\n" + _value_table(read_csv(d / "exact_vs_bounds.csv")) + "}")
    lines.append(r"\newcommand{\nineTableStepFive}{%" + "\n" + _step5_table(read_csv(d / "step5.csv")) + "}")
    lines.append(r"\newcommand{\nineTableForksA}{%" + "\n" + _fork_table(fo, "1/5", 5, 32, (10, 24)) + "}")
    lines.append(r"\newcommand{\nineTableForksB}{%" + "\n" + _fork_table(fo, "2/5", 3, 32, (10, 24)) + "}")
    lines.append(r"\newcommand{\nineTableEnum}{%" + "\n" + _enum_table(read_csv(d / "enum_checks.csv")) + "}")
    Path(path).write_text("\n".join(lines) + "\n")
