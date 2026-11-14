"""Checks of the table function t^+_{n,m} of articles/gradient-bound (section "The bound as a table function").

The engine build/cpp/gradient_tplus computes t^+ with upward rounding; here its
output is checked against exact rationals and against mpmath interval
arithmetic (keytrees.gradient_tplus).
"""
from __future__ import annotations

import csv
import doctest
import math
import os
import subprocess
from fractions import Fraction
from pathlib import Path

import mpmath
import pytest
from mpmath import iv

import keytrees.gradient_tplus as gt
from keytrees.gradient_bound import complete_cost

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gradient-bound"
ENGINE = Path(os.environ.get("KEYTREES_BUILD", ROOT / "build" / "cpp")) / "gradient_tplus"

NS = list(range(1, 13)) + [16, 20, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 4096, 2**16, 2**20]
MS = list(range(1, 17)) + [24, 32, 48, 64, 128, 256, 512, 1024, 2**12, 2**16, 2**20, 2**24, 2**32]


def grid():
    with open(DATA / "tplus.csv", newline="") as f:
        return {(int(r["n"]), int(r["m"])): r for r in csv.DictReader(f)}


def exact_values():
    with open(DATA / "exact_formula.csv", newline="") as f:
        return {(int(r["n"]), int(r["m"])): Fraction(int(r["numerator"]), int(r["denominator"])) for r in csv.DictReader(f)}


def engine(*args, stdin=None):
    if not ENGINE.exists():
        pytest.skip("build the engines first (cmake --build build/cpp)")
    return subprocess.run([str(ENGINE), *map(str, args), "--data", str(DATA)], input=stdin, capture_output=True,
                          text=True, check=True, cwd=ROOT).stdout


def frac(x):
    """An mpmath binary number as an exact Fraction."""
    man, exp = mpmath.mpf(x).man_exp
    return Fraction(man) * Fraction(2) ** exp


def log2_upper(m):
    iv.prec = 120
    return frac((iv.log(iv.mpf(m)) / iv.log(2)).b)


def log2_lower(m):
    iv.prec = 120
    return frac((iv.log(iv.mpf(m)) / iv.log(2)).a)


def test_doctests():
    result = doctest.testmod(gt, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0 and result.failed == 0


def test_grid_cells():
    g = grid()
    assert set(g) == {(n, m) for n in NS for m in MS if m <= 2**n}
    ingredients = ("exact", "trivial", "fixed", "union", "explicit", "cylinder")
    for (n, m), r in g.items():
        t = Fraction(r["tplus"])
        if m == 1:
            assert t == 0 and r["ingredient"] == "m=1"
            continue
        assert t >= complete_cost(m), (n, m)
        values = {k: Fraction(r[k]) for k in ingredients if r[k]}
        assert t == min(values.values()) and values[r["ingredient"]] == t
        cmax = Fraction(m + 1, 2) - Fraction(1, m)
        assert values["trivial"] >= min(Fraction(n), cmax)
        # computed from the double of t^+, which its 17-digit decimal exceeds by less than 1e-16 t^+
        assert Fraction(r["tplus_minus_log2m"]) >= t * (1 - Fraction(1, 10**16)) - log2_lower(m)


def test_grid_reproduces(tmp_path):
    out = tmp_path / "tplus.csv"
    engine("grid", out)
    assert out.read_text() == (DATA / "tplus.csv").read_text()


def test_exact_cells_are_below_every_ingredient():
    """On all exact cells t^+ >= t, also when the exact value itself is left out."""
    ex = exact_values()
    assert len(ex) == 473
    pairs = "".join(f"{n} {m}\n" for n, m in ex)
    for flags in ((), ("--no-exact",)):
        rows = list(csv.DictReader(engine("cells", *flags, stdin=pairs).splitlines()))
        assert len(rows) == 473
        for r in rows:
            key = int(r["n"]), int(r["m"])
            assert Fraction(r["tplus"]) >= ex[key], (key, flags)
            if not flags:
                assert Fraction(r["exact"]) >= ex[key]


def test_grid_exact_column():
    ex = exact_values()
    for key, r in grid().items():
        if r.get("exact"):
            assert Fraction(r["exact"]) >= ex[key]
            assert Fraction(r["exact"]) - ex[key] < Fraction(1, 10**12)


def parse_params(text):
    out = {}
    for tok in text.replace("(s1=m)", "").split():
        k, v = tok.split("=")
        out[k] = v
    return out


def check(engine_value, interval, rel=1e-9):
    """The engine's value is an upper bound of the interval and close to it."""
    hi = frac(interval.b)
    v = Fraction(engine_value)
    assert v >= hi
    assert v - hi <= rel * max(1, abs(hi))


FIXED_SAMPLE = [(4, 4), (10, 4), (20, 6), (48, 9), (64, 16), (96, 16), (256, 24), (512, 32), (1024, 48), (4096, 64)]
UNION_SAMPLE = [(32, 12), (24, 24), (64, 64), (128, 32), (24, 2**12), (64, 2**32), (128, 2**32), (64, 2**16)]
UNION_SLOW = [(1024, 2**20), (2**20, 2**32), (4096, 1024)]


@pytest.mark.parametrize("cell", FIXED_SAMPLE)
def test_fixed_against_intervals(cell):
    r = grid()[cell]
    check(r["fixed"], gt.fixed_bound(*cell))


def _union(cell):
    r = grid()[cell]
    p = parse_params(r["union_params"])
    s1 = int(p["s1"])
    check(r["union"], gt.union_bound(*cell, int(p["j"]), s1, float(p["x"])))


@pytest.mark.parametrize("cell", UNION_SAMPLE)
def test_union_against_intervals(cell):
    _union(cell)


@pytest.mark.slow
@pytest.mark.parametrize("cell", UNION_SLOW)
def test_union_against_intervals_slow(cell):
    _union(cell)


def _kappa(n, m, j):
    out = engine("kappa", n, m, j).split()
    return [0, 0] + [int(v) for v in out[1::2]]


def _cylinder(cell):
    r = grid()[cell]
    p = parse_params(r["cylinder_params"])
    s1 = int(p["s1"])
    kap = _kappa(*cell, int(p["j"]))[: min(s1, cell[1]) + 1]
    check(r["cylinder"], gt.cylinder_bound(*cell, kap, s1, float(p["x"])))


CYLINDER_SAMPLE = [(8, 64), (12, 64), (16, 24), (32, 48), (10, 256), (12, 512)]
CYLINDER_SLOW = [(24, 4096), (48, 4096), (48, 2**20)]


@pytest.mark.parametrize("cell", CYLINDER_SAMPLE)
def test_cylinder_against_intervals(cell):
    """Any requirement gives a valid bound, so the engine's requirement is checked as it is."""
    _cylinder(cell)


@pytest.mark.slow
@pytest.mark.parametrize("cell", CYLINDER_SLOW)
def test_cylinder_against_intervals_slow(cell):
    _cylinder(cell)


def test_explicit_against_intervals():
    with open(DATA / "gradient_constants.csv", newline="") as f:
        ck = {int(r["K"]): Fraction(r["c_K_upper"]) + Fraction(1, 10**8) for r in csv.DictReader(f)}
    seen = 0
    for (n, m), r in grid().items():
        if not r["explicit"]:
            continue
        K = int(parse_params(r["explicit_params"])["K"])
        assert 2**n >= m**K  # n >= K log m
        assert n >= (2 + Fraction(61, 100) * log2_upper(2 * n)) * log2_upper(m) + 6
        check(r["explicit"], gt.explicit_bound(n, m, ck[K]))
        seen += 1
    assert seen > 20


def test_where_each_ingredient_wins():
    """The paragraph "Which ingredient wins" of the article."""
    g = grid()
    ex = exact_values()
    win = {}
    for key, r in g.items():
        win.setdefault(r["ingredient"], set()).add(key)
    assert set(win) == {"m=1", "exact", "fixed", "union", "cylinder", "trivial"}  # (e) never wins
    for (n, m), r in g.items():
        if m == 1:
            continue
        lm = math.log2(m)
        if (n, m) in ex:
            assert r["ingredient"] == "exact"
        elif n >= 5 * m**1.5:
            assert r["ingredient"] == "fixed"
        elif n >= 2 * lm and r["ingredient"] != "fixed":
            assert m >= 6
            if r["ingredient"] == "union":  # a tie
                assert n == 2**20 and Fraction(r["union"]) == Fraction(r["cylinder"])
            else:
                assert r["ingredient"] == "cylinder"
        if r["ingredient"] == "fixed":
            assert n >= m**1.5 / 2
        if r["ingredient"] == "trivial":
            assert n < 2 * lm
        if m == 2**n:
            assert Fraction(r["trivial"]) == n and Fraction(r["tplus"]) == n
        if r["explicit"]:
            assert Fraction(r["union"]) <= Fraction(r["explicit"])
    assert len(win["union"]) == 4
    assert sum(1 for n, m in win["cylinder"] if n < 2 * math.log2(m)) == 18


def test_weak_region():
    """The paragraph "Where the bound is weak" and item (6) of the introduction."""
    g = grid()
    for (n, m), r in g.items():
        if m == 1:
            continue
        v = Fraction(r["tplus_minus_log2m"])
        lm = math.log2(m)
        if m >= 2**12 and lm < n <= 2 * lm:
            assert v > Fraction(55, 100)
        if m >= 2**12 and 2 * lm < n <= 3 * lm:
            assert Fraction(32, 100) < v < Fraction(54, 100)
        if m >= 2**12 and 3 * lm < n <= 8 * lm:
            assert Fraction(12, 100) < v < Fraction(32, 100)
        if m >= 2**16 and lm < n < 2 * lm:
            assert r["ingredient"] == "trivial"
        if n >= 2 * lm:
            assert v < Fraction(59, 100)
        if n >= 8 * lm:
            assert v < Fraction(13, 100)
        if n >= 16 * lm:
            assert v <= Fraction(89, 1000)
    r = g[24, 2**12]  # (d) alone against t^+, rounded up as in the table
    assert Fraction(r["union"]) - 12 > Fraction(2743, 1000) and Fraction(r["tplus_minus_log2m"]) <= Fraction(555, 1000)
    assert r["ingredient"] == "cylinder"


def test_small_forks_split_one_to_three():
    """At n = 3 log m the winning requirement of (f) allows 1:3 splits of four columns."""
    for cell in [(36, 2**12), (60, 2**20)]:
        out = engine("cell", *cell)
        line = next(x for x in out.splitlines() if "(f)" in x)
        j = int(line.split("j=")[1].split()[0])
        assert _kappa(*cell, j)[4] == 1


def test_article_table_is_generated():
    import importlib.util

    codes = Path(os.environ.get("CODES_DIR", ROOT.parent / "codes"))
    path = codes / "articles" / "gradient-bound" / "tplus_table.tex"
    if not path.exists():
        pytest.skip("the codes repository (articles) is not checked out next to keytrees")
    spec = importlib.util.spec_from_file_location("gbd", ROOT / "scripts" / "gradient_bound_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = path.read_text()
    g = grid()
    for n in mod.TPLUS_NS:
        for m in mod.TPLUS_MS:
            if (n, m) in g:
                assert mod.tplus_cell(g[n, m]) in text
