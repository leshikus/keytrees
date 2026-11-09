"""Checks of numbers stated in the 1999 papers and of the corrected statements."""
from __future__ import annotations

import csv
import doctest
import itertools
import math
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import keytrees.capacity
import keytrees.core
import keytrees.errors
import keytrees.flora
from keytrees.capacity import blahut_arimoto, channel_output, order_vertices, worst_case
from keytrees.core import (
    Table,
    complete_tree_cost,
    greedy_tree,
    huffman_cost,
    huffman_lengths,
    optimal_cost,
)
from keytrees.errors import ball, identify, minimum_distance, volume_bound
from keytrees.flora import (
    DIANTHUS_MIDDLE_SIBERIA,
    MOEHRINGIA,
    SAGINA,
    STELLARIA_FLORA_TREE,
    STELLARIA_MINIMAX_TREE,
    STELLARIA_NOVOSIBIRSK,
    STELLARIA_ORDER,
)

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize(
    "module", [keytrees, keytrees.core, keytrees.capacity, keytrees.errors, keytrees.flora]
)
def test_doctests(module):
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0
    assert result.failed == 0


def test_sagina_costs():
    assert optimal_cost(SAGINA) == pytest.approx(2.0)
    assert greedy_tree(SAGINA).cost(SAGINA.uniform()) == pytest.approx(2.0)


def test_dianthus_gradient_is_not_optimal():
    tree = greedy_tree(DIANTHUS_MIDDLE_SIBERIA)
    assert tree.cost(DIANTHUS_MIDDLE_SIBERIA.uniform()) == pytest.approx(17 / 6)
    assert optimal_cost(DIANTHUS_MIDDLE_SIBERIA) == pytest.approx(16 / 6)


def test_moehringia_cost():
    assert optimal_cost(MOEHRINGIA) == pytest.approx(5 / 3)


def test_stellaria_flora_tree():
    depths = STELLARIA_FLORA_TREE.depths()
    assert sorted(depths.values()) == [2, 2, 2, 3, 4, 5, 6, 6]
    uniform = {name: 1 / 8 for name in depths}
    assert STELLARIA_FLORA_TREE.cost(uniform) == pytest.approx(3.75)


def _stellaria_q():
    return order_vertices(8, STELLARIA_ORDER)


def test_stellaria_channel_matches_paper():
    Q = _stellaria_q()
    assert Q.shape == (8, 8)
    assert Q[0] == pytest.approx(np.full(8, 1 / 8))
    assert Q[1] == pytest.approx(np.array([0] + [1 / 7] * 7))
    assert (Q[2:].max(axis=1) == 1).all()


def test_stellaria_capacity_matches_closed_form():
    capacity, q = blahut_arimoto(_stellaria_q(), eps=1e-10)
    p = channel_output(_stellaria_q(), q)
    denom = 6 + 2 / 8**4
    assert capacity == pytest.approx(math.log2(denom), abs=1e-8)
    assert p[0] == pytest.approx(1 / (denom * 8**4), abs=1e-7)
    assert p[1] == pytest.approx(1 / (denom * 8**4), abs=1e-7)
    assert p[2:] == pytest.approx(np.full(6, 1 / denom), abs=1e-7)
    assert q[1] == pytest.approx(0.0, abs=1e-5)


def test_kuhn_tucker_inequality_in_the_example():
    # The draft wrote 18/7 and 2^18 < 7^7; (1/7) log2 8^4 is 12/7.
    assert math.log2(8**4) / 7 == pytest.approx(12 / 7)
    assert 2**12 < 7**7


def test_stellaria_minimax_tree_numbers():
    names = STELLARIA_NOVOSIBIRSK.objects
    Q = _stellaria_q()
    flora = STELLARIA_FLORA_TREE.depths()
    minimax = STELLARIA_MINIMAX_TREE.depths()
    assert worst_case([flora[n] for n in names], Q)[0] == pytest.approx(6)
    assert worst_case([flora[n] for n in names], Q)[1] == pytest.approx(6)
    cost, redundancy = worst_case([minimax[n] for n in names], Q)
    assert cost == pytest.approx(3 + 1 / 8)
    assert redundancy == pytest.approx(3)


def test_minimax_tree_uses_valid_tests():
    """Every path of the published tree follows the published character table."""
    t = STELLARIA_NOVOSIBIRSK
    index = {name: i for i, name in enumerate(t.objects)}

    def check(node, members):
        if node.name is not None:
            assert members == {index[node.name]}
            return
        row = t.answers[int(node.test) - 1]
        yes = {i for i in members if row[i] == 1}
        no = {i for i in members if row[i] == 0}
        assert yes | no == members, "test undefined on a candidate"
        check(node.yes, yes)
        check(node.no, no)

    check(STELLARIA_MINIMAX_TREE, set(range(8)))


def test_huffman_recovers_minimax_lengths():
    Q = _stellaria_q()
    _, q = blahut_arimoto(Q, eps=1e-10)
    lengths = huffman_lengths(list(channel_output(Q, q)))
    assert sorted(lengths) == [2, 3, 3, 3, 3, 3, 4, 4]


def test_complete_tree_matches_huffman():
    for m in range(2, 300):
        assert complete_tree_cost(m) == pytest.approx(huffman_cost([1 / m] * m))


def test_undefined_values_block_a_test():
    t = Table.from_rows("abc", {"x": [1, None, 0], "y": [1, 0, 0], "z": [0, 0, 1]})
    assert greedy_tree(t).test == "y"


# ---- recurrences and bounds ----------------------------------------------------------


def exact_table(s, nmax):
    """Exact rational recurrence, independent of the C++ implementation."""
    t = {(0, 0): Fraction(0), (0, 1): Fraction(0)}
    for n in range(1, nmax + 1):
        H = 2 ** (n - 1)
        t[(n, 0)] = t[(n, 1)] = Fraction(0)
        for m in range(2, 2**n + 1):
            weights = [Fraction(math.comb(H, k) * math.comb(H, m - k), math.comb(2 * H, m)) for k in range(m + 1)]
            inner = sum(
                weights[k] * (1 + Fraction(k, m) * t[(n - 1, k)] + Fraction(m - k, m) * t[(n - 1, m - k)])
                for k in range(1, m)
                if weights[k]
            )
            useless = weights[0] + weights[m]
            charge = 0 if m <= s else 1
            prev = t.get((n - 1, m), Fraction(0))
            t[(n, m)] = inner + useless * (charge + prev)
    return t


def test_table_values_for_s3_are_exact():
    t = exact_table(3, 5)
    for n in range(2, 6):
        assert t[(n, 2)] == 1
        assert t[(n, 3)] == Fraction(5, 3)
        m = 2**n
        assert t[(n, m)] == n
        assert t[(n, m - 1)] == n - Fraction(1, m - 1)
        assert t[(n, m - 2)] == n - Fraction(2, m - 2)


def test_16_over_7_is_the_supremum_for_four_words():
    t = exact_table(3, 7)
    values = [t[(n, 4)] for n in range(2, 8)]
    assert all(v < Fraction(16, 7) for v in values)
    assert all(a <= b for a, b in zip(values[1:], values[2:]))
    # The closing inequality in the draft: 1 + c2 + (5/2) c3 + 4 c4 * (4/7) <= 16/7.
    assert Fraction(16, 7) - values[-1] < Fraction(1, 50)


def test_literal_random_position_violates_the_theorem_at_two_words():
    t = exact_table(1, 7)
    bound = 1 + 29 / 28 - 1  # log2 2 + 29/28 - log2(4)/2
    assert float(t[(2, 2)]) == pytest.approx(4 / 3)
    assert float(t[(5, 2)]) > bound


def _rows(name):
    with open(DATA / "exact" / name, newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.skipif(not (DATA / "exact" / "brute.csv").exists(), reason="run the C++ experiments first")
def test_brute_force_matches_cpp_recurrence_and_gradient_formula():
    for row in _rows("brute.csv"):
        n, m = int(row["n"]), int(row["m"])
        for variant in ("s1", "s2", "s3", "patricia"):
            assert float(row[f"t_{variant}"]) == pytest.approx(float(row[f"rec_{variant}"]), abs=1e-9)
        assert float(row["t_optimal"]) <= float(row["t_gradient"]) + 1e-12
        assert float(row["t_gradient"]) <= float(row["t_patricia"]) + 1e-9 or m > 5
        if m == 4 and n >= 2:
            assert float(row["t_gradient"]) == pytest.approx(2 + float(row["p_formula"]) / 4, abs=1e-9)
        if m == 5 and n >= 3:
            assert float(row["t_gradient"]) == pytest.approx(12 / 5 + 2 * float(row["p_formula"]) / 5, abs=1e-9)


@pytest.mark.skipif(not (DATA / "exact" / "limit.csv").exists(), reason="run the C++ experiments first")
def test_limit_constants_reach_published_values():
    constants = {
        "s1": math.gamma(1) * 0 + 0.5772156649015329 / math.log(2) + 0.5,
        "s2": 0.5772156649015329 / math.log(2) + 0.5 - 1 / (2 * math.log(2)),
        "s3": 0.5772156649015329 / math.log(2) + 0.5 - 1 / (2 * math.log(2)) - 1 / (8 * math.log(2)),
        "patricia": 0.5772156649015329 / math.log(2) - 0.5,
    }
    tail = {name: [] for name in constants}
    for row in _rows("limit.csv"):
        if int(row["m"]) >= 4096:
            tail[row["variant"]].append(float(row["redundancy"]))
    for name, value in constants.items():
        assert abs(sum(tail[name]) / len(tail[name]) - value) < 2e-4


# ---- error tolerant keys -------------------------------------------------------------


def _random_code(m, n, d, rng):
    code = []
    while len(code) < m:
        word = "".join(rng.choice("01") for _ in range(n))
        if all(sum(a != b for a, b in zip(word, c)) >= d for c in code):
            code.append(word)
    return code


@pytest.mark.parametrize("t", [0, 1, 2])
def test_elimination_key_is_correct_for_all_error_patterns(t):
    rng = random.Random(t + 1)
    code = _random_code(6, 11, 2 * t + 1, rng)
    assert minimum_distance(code) >= 2 * t + 1
    for species, word in enumerate(code):
        for received in ball(word, t):
            found, tests = identify(code, received, t)
            assert found == species
            assert tests <= len(word)


def test_ball_size():
    assert len(ball("0" * 10, 2)) == 1 + 10 + 45
    assert len(set(ball("0" * 10, 2))) == 56


def test_balls_of_a_valid_code_are_disjoint():
    code = _random_code(5, 12, 5, random.Random(5))
    words = list(itertools.chain.from_iterable(ball(w, 2) for w in code))
    assert len(words) == len(set(words))


def test_volume_bound_matches_known_lie_threshold():
    # Pelc's one-lie game: 2^20 objects need 25 questions by the volume bound.
    assert volume_bound(1_000_000, 1) == 25
