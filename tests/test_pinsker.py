"""Checks of the Pinsker chapter and of the Huffman counterexample for chapter 5."""
from __future__ import annotations

import doctest
import math
from fractions import Fraction

import mpmath
import pytest

import keytrees.pinsker
from keytrees.capacity import blahut_arimoto, channel_output, order_vertices, worst_case
from keytrees.core import huffman_lengths
from keytrees.pinsker import parametrisation, remainder, taylor_coefficients

COEFFICIENTS = taylor_coefficients(100)


def test_doctests():
    result = doctest.testmod(keytrees.pinsker, optionflags=doctest.ELLIPSIS)
    assert result.attempted > 0 and result.failed == 0


def test_printed_coefficients():
    c = COEFFICIENTS
    assert c[2] == Fraction(1, 2)
    assert c[4] == Fraction(1, 2**2 * 3**2)
    assert c[6] == Fraction(1, 2 * 3**3 * 5)
    assert c[8] == Fraction(221, 2**3 * 3**5 * 5**2 * 7)
    assert c[10] == Fraction(299, 2 * 3**8 * 5**2 * 7)
    assert c[12] == Fraction(5983, 2**2 * 3**9 * 5 * 7**2 * 11)


def test_sign_pattern():
    assert all(COEFFICIENTS[n] > 0 for n in range(2, 61, 2))
    assert all(COEFFICIENTS[n] < 0 for n in range(62, 97, 2))
    assert COEFFICIENTS[98] > 0


def test_series_matches_parametrisation():
    mpmath.mp.dps = 60
    v, l = parametrisation(mpmath.mpf("0.5"), mpmath)
    series = sum(mpmath.mpf(c.numerator) / c.denominator * v**n for n, c in COEFFICIENTS.items())
    assert abs(series - l) < mpmath.mpf(10) ** -40


def test_r50_negative_near_ten_and_r48_positive():
    mpmath.mp.dps = 150
    r50 = remainder(mpmath.mpf("9.92125"), COEFFICIENTS, 50, mpmath)
    assert -0.0041 < float(r50) < -0.0034
    for t in ("0.5", "0.8", "2", "5", "9.92", "20"):
        assert remainder(mpmath.mpf(t), COEFFICIENTS, 48, mpmath) > 0


def test_exercise_values():
    mpmath.mp.dps = 50
    t = mpmath.findroot(lambda x: parametrisation(x, mpmath)[0] - 1, 1.2)
    assert float(parametrisation(t, mpmath)[1]) == pytest.approx(0.5323, abs=5e-5)
    assert 0.5 * math.log(3) == pytest.approx(0.5493, abs=5e-5)
    t8 = sum(float(COEFFICIENTS[n]) for n in (2, 4, 6, 8))
    assert t8 == pytest.approx(0.5321, abs=5e-5)


def test_huffman_tree_can_exceed_capacity_plus_one():
    """Chapter 5: a Huffman tree for p* can have worst-case redundancy above C(Q)+1."""
    order = [(0, 8), (1, 8), (3, 4), (3, 8), (4, 6), (6, 7), (8, 9)]
    Q = order_vertices(10, order)
    capacity, q = blahut_arimoto(Q, eps=1e-11)
    p = channel_output(Q, q)
    lengths = huffman_lengths(list(p))
    _, redundancy = worst_case(lengths, Q)
    # Rounded lengths always stay within C+1.
    rounded = [math.ceil(-math.log2(x)) for x in p]
    assert sum(2.0**-n for n in rounded) <= 1 + 1e-9
    assert worst_case(rounded, Q)[1] < capacity + 1
    assert capacity == pytest.approx(2.306, abs=5e-4)
    assert redundancy == pytest.approx(5 - math.log2(3))
    assert redundancy > capacity + 1
