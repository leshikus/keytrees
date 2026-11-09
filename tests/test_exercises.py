"""Check the numeric answers given in the exercises appendix."""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from keytrees.capacity import blahut_arimoto, order_vertices, worst_case
from keytrees.core import Table, entropy, greedy_tree, huffman_cost, optimal_cost
from keytrees.errors import ball, identify
from keytrees.flora import DIANTHUS_MIDDLE_SIBERIA, STELLARIA_ORDER


def test_corduliinae_cost():
    assert 0.6 * 1 + 0.3 * 2 + 0.1 * 2 == pytest.approx(1.4)


def test_moehringia_costs_with_frequencies():
    character_1_first = 0.15 * 1 + 0.15 * 2 + 0.7 * 2
    character_2_first = 0.7 * 1 + 0.15 * 2 + 0.15 * 2
    assert character_1_first == pytest.approx(1.85)
    assert character_2_first == pytest.approx(1.3)


def test_kraft_lists():
    assert sum(2.0**-n for n in (1, 2, 3, 3)) <= 1
    assert sum(2.0**-n for n in (1, 1, 2)) > 1


def test_huffman_and_entropy():
    p = [0.4, 0.3, 0.2, 0.1]
    assert huffman_cost(p) == pytest.approx(1.9)
    assert entropy(p) == pytest.approx(1.846, abs=5e-4)


def test_complete_tree_excess_maximum():
    x = 2 * math.log(2)
    value = 2 - 2 / x - math.log2(x)
    assert value == pytest.approx(0.0861, abs=5e-5)
    grid = [1 + i / 100000 for i in range(100000)]
    assert max(2 - 2 / t - math.log2(t) for t in grid) == pytest.approx(value, abs=1e-8)


def test_greedy_rules_depend_only_on_balance():
    m = 12
    up = [(a * a + (m - a) ** 2) / m for a in range(1, m // 2 + 1)]
    md = [(a * math.log2(a) + (m - a) * math.log2(m - a)) / m for a in range(1, m // 2 + 1)]
    assert up == sorted(up, reverse=True)
    assert md == sorted(md, reverse=True)


def test_dianthus_tie_break_on_character_two():
    t = DIANTHUS_MIDDLE_SIBERIA
    order = [1, 0, 2, 3]
    reordered = Table(t.objects, tuple(t.tests[i] for i in order), tuple(t.answers[i] for i in order))
    assert greedy_tree(reordered).cost(reordered.uniform()) == pytest.approx(16 / 6)
    assert optimal_cost(t) == pytest.approx(16 / 6)


def test_two_words_of_length_two():
    total = Fraction(0)
    for a, b in itertools.combinations(range(4), 2):
        total += 1 if (a ^ b) == 3 else Fraction(3, 2)
    assert total / 6 == Fraction(4, 3)


def test_limit_of_two_words_for_the_trie():
    t = 1.0
    for n in range(2, 60):
        h = 2 ** (n - 1)
        both = math.comb(h, 2) / math.comb(2 * h, 2)
        t = 1 + 2 * both * t
    assert t == pytest.approx(2, abs=1e-6)


def test_t5_for_r3():
    t = {2: Fraction(1), 3: Fraction(5, 3)}
    for m in (4, 5):
        t[m] = (2 ** (m - 1) + sum(math.comb(m - 1, k - 1) * t[k] for k in range(2, m))) / Fraction(2 ** (m - 1) - 1)
    assert t[4] == Fraction(16, 7)
    assert t[5] == Fraction(274, 105)
    assert math.log2(5) + 29 / 28 - math.log2(10) / 5 == pytest.approx(2.693, abs=5e-4)


def test_seven_words_of_length_three():
    for missing in range(8):
        words = [w for w in range(8) if w != missing]

        def depth_sum(group, position):
            if len(group) <= 1:
                return 0
            one = [w for w in group if (w >> position) & 1]
            zero = [w for w in group if not (w >> position) & 1]
            return len(group) + depth_sum(one, position - 1) + depth_sum(zero, position - 1)

        assert Fraction(depth_sum(words, 2), 7) == Fraction(20, 7) == 3 - Fraction(1, 7)


def test_markov_value():
    assert 1 - (29 / 28) / (0.5 * 10) == pytest.approx(0.793, abs=5e-4)


def test_two_species_channel():
    Q = order_vertices(2, [(0, 1)])
    capacity, _ = blahut_arimoto(Q, eps=1e-10)
    assert capacity == pytest.approx(math.log2(5 / 4), abs=1e-8)
    assert capacity == pytest.approx(0.322, abs=5e-4)
    assert worst_case([1, 1], Q)[1] == pytest.approx(1)


def test_chain_of_three():
    assert blahut_arimoto(order_vertices(3, [(0, 1), (1, 2)]))[0] == pytest.approx(0.4835, abs=5e-5)


def test_stellaria_up_sets_and_vertices():
    assert order_vertices(8, STELLARIA_ORDER, reduce=False).shape[0] == 65
    assert order_vertices(8, STELLARIA_ORDER).shape[0] == 8


@pytest.mark.parametrize("t", [1, 2, 3])
def test_two_words_elimination_costs(t):
    n = 2 * t + 1
    code = ["0" * n, "1" * n]
    assert identify(code, code[0], t) == (0, t + 1)
    worst = 0
    for word in ball(code[0], t):
        found, tests = identify(code, word, t)
        assert found == 0
        worst = max(worst, tests)
    assert worst == 2 * t + 1


def test_weights_after_first_answer():
    code = ["00000", "11111"]
    consistent = [
        sum(w[0] == "0" for w in ball(c, 1)) for c in code
    ]
    assert consistent == [5, 1]


