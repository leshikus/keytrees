"""Checks of the (o, m, n) tree tables in data/omn (articles/omn-trees)."""
from __future__ import annotations

import csv
import doctest
from collections import defaultdict
from pathlib import Path

import pytest

import keytrees.omn
from keytrees.core import complete_tree_cost
from keytrees.omn import (
    brute_bounded_total,
    brute_total,
    brute_worst,
    character,
    complete_tree_total,
    deppe_three_lies,
    log2_ceil,
    pelc_one_lie,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "omn"
CELLS = [(m, n) for n in range(4) for m in range(n + 1)]


def load(kind: str) -> dict[tuple[int, int], dict[int, dict[str, str]]]:
    out: dict[tuple[int, int], dict[int, dict[str, str]]] = defaultdict(dict)
    with (DATA / f"{kind}.csv").open() as fh:
        for row in csv.DictReader(fh):
            out[int(row["m"]), int(row["n"])][int(row["o"])] = row
    return out


WORST, AVERAGE, MINMAX = load("worst"), load("average"), load("minmax")


def W(o: int, m: int, n: int) -> int:
    return int(WORST[m, n][o]["W"])


def test_doctests():
    assert doctest.testmod(keytrees.omn).failed == 0


def test_every_cell_present_and_contiguous():
    for data in (WORST, AVERAGE, MINMAX):
        for c in CELLS:
            assert sorted(data[c]) == list(range(1, max(data[c]) + 1))


@pytest.mark.parametrize("m,n", CELLS)
def test_brute_force_small(m, n):
    # Labelled objects, every subset as a question, leaves checked from the definition.
    for o in range(1, 6):
        assert brute_worst(o, m, n) == W(o, m, n)
        assert brute_total(o, m, n) == int(AVERAGE[m, n][o]["total"])
        assert brute_bounded_total(o, m, n) == int(MINMAX[m, n][o]["total"])
        assert int(MINMAX[m, n][o]["W"]) == W(o, m, n)


def test_no_errors():
    for o in WORST[0, 0]:
        assert W(o, 0, 0) == log2_ceil(o)
    for o, row in AVERAGE[0, 0].items():
        assert int(row["total"]) == complete_tree_total(o)
        assert int(row["total"]) == pytest.approx(o * complete_tree_cost(o))


def test_detection_only_theorem():
    # Proved in the article: W(o, 0, n) = ceil(log2 o) + n for o >= 2.
    for n in range(4):
        for o in range(2, max(WORST[0, n]) + 1):
            assert W(o, 0, n) == log2_ceil(o) + n


def test_average_theorem():
    # Proved in the article: A(o, m, n) = T(o) + n o for o >= 2, and 0 for o = 1.
    for (m, n), col in AVERAGE.items():
        for o, row in col.items():
            assert int(row["total"]) == (0 if o == 1 else complete_tree_total(o) + n * o)


def test_pelc_one_lie():
    assert len(WORST[1, 1]) >= 1000
    for o in WORST[1, 1]:
        assert W(o, 1, 1) == pelc_one_lie(o)


def load_singles() -> dict[int, int]:
    with (DATA / "worst_3_3_single.csv").open() as fh:
        return {int(r["o"]): int(r["W"]) for r in csv.DictReader(fh)}


def test_deppe_three_lies():
    top = min(265, max(WORST[3, 3]))
    assert top >= 140
    for o in range(1, top + 1):
        assert W(o, 3, 3) == deppe_three_lies(o)


def test_deppe_three_lies_single_runs():
    singles = load_singles()
    assert singles
    for o, w in singles.items():
        assert o > max(WORST[3, 3]) and w == deppe_three_lies(o)


def test_oeis_a286496():
    # A286496 (Renyi-Ulam liar numbers), terms for q = 0..15, as fetched 2026-11-10.
    terms = [1, 1, 1, 2, 2, 4, 8, 16, 28, 50, 92, 170, 314, 584, 1092, 2048]
    col = WORST[1, 1]
    for q, t in enumerate(terms):
        assert max(o for o in col if W(o, 1, 1) <= q) == t


def test_lower_bounds_and_character():
    for (m, n), col in WORST.items():
        for o, row in col.items():
            assert int(row["hamming"]) == character(o, m)
            assert W(o, m, n) >= int(row["hamming"])
            if o >= 2:
                assert W(o, m, n) >= m + n + 1 == int(row["pair"])


def test_two_lies_as_reported_in_the_survey():
    # Pelc 2002, Section 3.1.1, reporting Guzicki and Czyzowicz-Mundici-Pelc; their
    # exact characterisation was not available. W is the volume bound or one more,
    # and for o = 2^j the volume bound unless j = 2.
    for o in WORST[2, 2]:
        assert W(o, 2, 2) - character(o, 2) in (0, 1)
    j = 1
    while 2**j in WORST[2, 2]:
        assert W(2**j, 2, 2) - character(2**j, 2) == (1 if j == 2 else 0)
        j += 1
    assert j >= 8


def test_monotone_in_m_and_n():
    for m, n in CELLS:
        for o in WORST[m, n]:
            if n + 1 <= 3 and o in WORST[m, n + 1]:
                assert W(o, m, n + 1) >= W(o, m, n)
            if m + 1 <= n and o in WORST[m + 1, n]:
                assert W(o, m + 1, n) >= W(o, m, n)


def test_shift_theorem():
    # Theorem 4: W(o, m, n) >= W(o, m, m) + n - m for o >= 2.
    for m, n in CELLS:
        for o in WORST[m, n]:
            if o >= 2 and o in WORST[m, m]:
                assert W(o, m, n) >= W(o, m, m) + n - m


def test_stated_run_ends():
    # The article: the excess runs end at 16, 28, 50, values of N_{m,m}(q), m = 1, 2, 3.
    for m in (1, 2, 3):
        col = WORST[m, m]
        values = {max(o for o in col if W(o, m, m) <= q) for q in range(min(W(51, m, m), 40))}
        assert {16, 28, 50} <= values


def test_conjecture_shift_on_computed_range():
    # Conjecture 1 of the article, checked only on the computed range.
    for m, n in CELLS:
        for o in WORST[m, n]:
            if o >= 2 and o in WORST[m, m]:
                assert W(o, m, n) == W(o, m, m) + n - m


def test_minmax_excess_on_computed_range():
    # A* >= T(o) + n o always; equality for m = 0 (proved); Conjecture 2 for m >= 1.
    excess: dict[int, set[int]] = defaultdict(set)
    for (m, n), col in MINMAX.items():
        for o, row in col.items():
            if o < 2:
                continue
            e = int(row["total"]) - complete_tree_total(o) - n * o
            assert e >= 0
            if m == 0:
                assert e == 0
            else:
                excess[o].add(e)
    common = min(max(MINMAX[c]) for c in CELLS if c[0] >= 1)
    for o in range(2, common + 1):
        assert len(excess[o]) == 1
    assert excess[16] == {16} and excess[13] == {1} and excess[17] == {0}
