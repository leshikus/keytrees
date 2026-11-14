"""Checks of Step 9 (codes: articles/gradient-bound/analytic/09-exact.tex): exact values for small m, n.

Every number stated in 09-exact.tex is either a macro of the generated 09-tables.tex (checked
by regenerating it) or checked by a test below. Exact rational arithmetic throughout.
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import doctest
import gzip
import math
import os
import subprocess
from fractions import Fraction
from math import comb
from pathlib import Path

import mpmath
import pytest

import keytrees.gradient_analytic_9 as g9
from keytrees.gradient_bound import complete_cost, complete_redundancy_sup, rho_upper

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gradient-bound"
OUT = DATA / "analytic"
ENGINE = ROOT / "build" / "cpp" / "gradient_enum"
CODES = Path(os.environ.get("CODES_DIR", ROOT.parent / "codes"))
TEXT = CODES / "articles" / "gradient-bound" / "analytic"


def test_doctests():
    result = doctest.testmod(g9, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 50
    assert result.failed == 0


# ---------------------------------------------------------------- three computations agree


@pytest.mark.parametrize("n,m", [(1, 4), (2, 5), (3, 4), (4, 3), (2, 6), (3, 5), (4, 4), (5, 3)])
def test_brute_equals_product_formula(n, m):
    assert g9.brute_classes(n, m) == g9.product_classes(n, m)


@pytest.mark.skipif(not ENGINE.exists(), reason="build/cpp/gradient_enum is not built")
@pytest.mark.parametrize("n,m", [(3, 5), (4, 4), (2, 7), (3, 6), (5, 4)])
def test_engine_symmetry_reduction_is_exact(n, m):
    run = lambda mode: subprocess.run([str(ENGINE), mode, str(n), str(m), "4"], capture_output=True, text=True, check=True).stdout  # noqa: E731
    sym = run("sym")
    assert sym == run("brute")
    assert g9.parse_enum(sym) == g9.product_classes(n, m)


@pytest.mark.skipif(not ENGINE.exists(), reason="build/cpp/gradient_enum is not built")
def test_engine_against_python_brute_force():
    out = subprocess.run([str(ENGINE), "sym", "3", "5", "2"], capture_output=True, text=True, check=True).stdout
    assert g9.parse_enum(out) == g9.brute_classes(3, 5)


def test_stored_enumerations():
    """Every stored enumeration: totals, product formula, product identity, chains, recursion."""
    rows = g9.enum_rows(OUT / "enum")
    assert [(r["n"], r["m"]) for r in rows] == g9.ENUM_CELLS
    for r in rows:
        assert all(r[k] == 1 for k in ("total_ok", "product", "identity", "chains", "recursion")), r
    stored = g9.read_csv(OUT / "enum_checks.csv")
    assert [{k: str(v) for k, v in r.items()} for r in rows] == stored


def test_enumeration_multiset_weights():
    """sum over multisets of (m-1)!/prod(mult!) = 2^(n(m-1)): the weights of Lemma sym."""
    for n, m in ((2, 4), (3, 3), (2, 5)):
        words = range(2**n)
        tot = 0
        for ms in __import__("itertools").combinations_with_replacement(words, m - 1):
            w = math.factorial(m - 1)
            for x in set(ms):
                w //= math.factorial(ms.count(x))
            tot += w
        assert tot == 2 ** (n * (m - 1))


# ---------------------------------------------------------------- the data files


def test_exact_vs_bounds_file_and_validity():
    rows = g9.exact_rows(DATA)
    stored = g9.read_csv(OUT / "exact_vs_bounds.csv")
    assert [{k: str(v) for k, v in r.items()} for r in rows] == stored
    for r in rows:
        n, m = r["n"], r["m"]
        ex = Fraction(r["excess"])
        lam = mpmath.log(m, 2)
        red = g9._mp(Fraction(r["t"])) - lam
        assert ex >= 0
        assert g9.theorem1_excess_bound(n, m) >= ex
        assert mpmath.mpf(r["prop_bound"]) >= red - mpmath.mpf(10) ** -9
        assert mpmath.mpf(r["step5_exact_QF"]) >= red - mpmath.mpf(10) ** -9
        assert mpmath.mpf(r["step5_bound_QF"]) >= mpmath.mpf(r["step5_exact_QF"]) - mpmath.mpf(10) ** -9
        qf = Fraction(r["QF"])
        for pi in (None, Fraction(1, m + 1)):
            assert qf <= g9.step5_union_bound(n, m, pi) and qf <= g9.step5_grouped_bound(n, m, pi)


def test_step5_and_forks_files():
    assert [{k: str(v) for k, v in r.items()} for r in g9.step5_rows()] == g9.read_csv(OUT / "step5.csv")
    assert [{k: str(v) for k, v in r.items()} for r in g9.fork_rows()] == g9.read_csv(OUT / "forks.csv")


def test_step5_failure_against_enumeration():
    for n, m in ((4, 8), (5, 8), (8, 5), (4, 12)):
        cl = g9.parse_enum(gzip.open(OUT / "enum" / f"enum_{n}_{m}.txt.gz", "rt").read())
        recs = g9.fork_records(cl, n, Fraction(0), m)
        assert g9.step5_failure(n, m) == sum(r["bad"] * r["W"] for r in recs) / 2 ** (n * m)


def test_formula_extension_files():
    t = g9.load_formula(DATA)
    runs = g9.read_csv(OUT / "formula_runs.csv")
    assert all((int(r["n"]), int(r["m"])) in t for r in runs)
    assert all(float(r["seconds"]) <= 1800 for r in runs)
    path = OUT / "formula_out_of_reach.csv"
    if path.exists():
        assert all((int(r["n"]), int(r["m"])) not in t for r in g9.read_csv(path))
    assert len(t) == 400 + len(runs)


@pytest.mark.skipif(not TEXT.exists(), reason="the codes repository is not checked out next to keytrees")
def test_generated_tex_is_current(tmp_path):
    g9.write_tex(OUT, tmp_path / "t.tex")
    assert (tmp_path / "t.tex").read_text() == (TEXT / "09-tables.tex").read_text()


# ---------------------------------------------------------------- numbers stated in 09-exact.tex


def ev():
    return {(int(r["n"]), int(r["m"])): r for r in g9.read_csv(OUT / "exact_vs_bounds.csv")}


def test_section2_theorem_41():
    e = ev()
    assert round(float(e[24, 4]["thm_ratio"]), 5) == 1.00003
    assert 95 < float(e[24, 8]["thm_ratio"]) < 105
    assert 1.5e7 < float(e[24, 12]["thm_ratio"]) < 2.5e7
    good = {}
    for (n, m), r in e.items():
        if float(r["thm_bound"]) <= 0.01:
            good.setdefault(m, []).append(n)
    assert max(good) == 12
    assert [min(good[m]) for m in range(4, 9)] == [7, 12, 16, 20, 23]
    for m in range(4, 9):  # from that n on, in every computed cell
        assert all(n in good[m] for (n, mm) in e if mm == m and n >= min(good[m]))
    for m in range(9, 13):
        assert good[m] == [math.isqrt(2 * m**3 - 1) + 1]


def test_section2_proposition_52():
    def tail(s, x):
        th = 0.5 + math.sqrt(x / s)
        return 4 * x / (s * (1 - th))

    x = math.log(2) / 2
    assert [math.floor(tail(s, x) * 1000) / 1000 for s in (8, 16, 32)] == [0.593, 0.245, 0.109]
    assert all(tail(16, x + d) > tail(16, x) for d in (1e-3, 0.1, 1.0))  # increasing in x
    for m, val in ((8, 0.375), (16, 0.4375)):
        kap = [0, 0] + [max(1, s // 2 - 1) for s in range(2, m + 1)]
        assert abs(float(rho_upper(kap, m)[m]) - val) < 1e-3
    e = ev()
    for (n, m), r in e.items():
        assert n < 20 * math.log2(m)  # Corollary 5.4 never applies
        if m >= 8 and n == math.ceil(math.log2(m)):
            assert int(r["prop_j"]) >= 1


def test_section2_theorems_and_assembly():
    e = ev()
    gam = complete_redundancy_sup()
    assert max(float(r["t_minus_log"]) for r in e.values()) < gam + 0.01
    worst = max(e, key=lambda k: float(e[k]["t_minus_log"]))
    assert worst == (6, 23)
    for m, first in ((4, 7), (8, 17)):
        ns = sorted(n for (n, mm) in e if mm == m)
        ok = [n for n in ns if float(e[n, m]["step5_exact_QF"]) <= float(e[n, m]["t_minus_log"]) + 0.01]
        assert ok[0] == first and ok == [n for n in ns if n >= first]
    assert round(float(e[24, 16]["step5_bound_QF"]), 2) == 6.48
    assert round(float(e[24, 16]["t_minus_log"]), 6) == 0.000434
    # Remark step7: w = floor((n - log m)/(2 log(e n))) <= 3 < 17 <= ceil(log 1e5)
    w = max(math.floor((n - math.log2(m)) / (2 * math.log2(math.e * n))) for (n, m) in e)
    assert w <= 3 < 17 == math.ceil(math.log2(10**5))
    zero = sorted((n, m) for (n, m), r in e.items() if Fraction(r["excess"]) == 0)
    expect = [(2, 4)] + [(3, m) for m in range(5, 9)] + [(4, m) for m in range(12, 17)] + [(5, m) for m in range(27, 33)]
    assert zero == sorted(expect) and len(zero) == 16
    assert all(n == math.ceil(math.log2(m)) for n, m in zero)


def test_section3_step5_table():
    rows = {(int(r["m"]), int(r["n"])): r for r in g9.read_csv(OUT / "step5.csv")}
    keys = ("union_exact_pi", "grouped_exact_pi", "union_lemma_pi", "grouped_lemma_pi")
    ratio, best = {}, {}
    for m in (4, 6, 8, 12, 16):
        for n in (8, 12, 16, 24):
            r = rows[m, n]
            b = [mpmath.mpf(r[k]) for k in keys]
            q = g9._mp(Fraction(r["QF"]))
            ratio[m, n] = min(b) / q
            best[m, n] = keys[b.index(min(b))]
            if m >= 12:
                assert b[0] > 1
            if m >= 8:
                assert b[1] > 0.1
            assert b[2] > b[0] and b[3] > b[1]
    r4 = [ratio[4, n] for n in (8, 12, 16, 24)]
    assert round(float(min(r4)), 1) == 4.8 and round(float(max(r4)), 1) == 6.9
    r6 = [ratio[6, n] for n in (8, 12, 16, 24)]
    assert 20.5 < min(r6) and max(r6) < 31.5
    rr = [ratio[m, n] for m in (8, 12, 16) for n in (8, 12, 16, 24)]
    assert 7 <= min(rr) < 7.5 and 361.5 < max(rr) < 362.5
    for m in (4, 6, 8, 12, 16):
        vals = [ratio[m, n] for n in (8, 12, 16, 24)]
        if m != 4:
            assert vals == sorted(vals)  # grows with n
    union = {k for k, v in best.items() if v == "union_exact_pi"}
    assert union == {(4, n) for n in (8, 12, 16, 24)} | {(6, n) for n in (12, 16, 24)} | {(8, 24)}
    assert all(v in ("union_exact_pi", "grouped_exact_pi") for v in best.values())
    # pi_m is attained at the largest even u <= m
    for m in range(2, 40):
        u = m - m % 2
        assert g9.perfect_min(m) == Fraction(comb(u, u // 2), 2**u)


def test_section4_parameters():
    b5, b25, b10 = Fraction(1, 5), Fraction(2, 5), Fraction(1, 10)
    assert all(b5 * b5 * s < 1 for s in range(2, 25))
    assert b5 < math.sqrt(math.pi) / 8 < b25 and 4 * b25 / math.sqrt(math.pi) > 0.5
    even = [s for s in range(4, 40, 2) if g9.good_fork(s, s // 2 - 1, b25, 3)]
    odd = [s for s in range(5, 40, 2) if g9.good_fork(s, (s - 3) // 2, b25, 3)]
    assert even[0] == 8 and even == list(range(8, 40, 2))
    assert odd[0] == 15 and odd == list(range(15, 40, 2))
    assert [s for s in range(3, 60, 2) if g9.good_fork(s, s // 2, b10, 2)][0] == 25
    assert round(float(g9.c1_limit(b5)), 4) == 0.0025 and g9.c1_limit(b25) == 0
    assert g9.m_star_limit(b25) == mpmath.inf


def test_section4_fork_tables():
    fo = g9.read_csv(OUT / "forks.csv")
    tab = lambda b, s1: [r for r in fo if (r["b"], r["s1"], r["m"]) == (b, str(s1), "32") and r["n"] in ("10", "24")]  # noqa: E731
    A, B = tab("1/5", 5), tab("2/5", 3)
    ratio = lambda r: Fraction(r["bad_max"]) / Fraction(r["bad_avg"])  # noqa: E731
    la = [r for r in A if r["large"] == "1"]
    lb = [r for r in B if r["large"] == "1"]
    assert max(ratio(r) for r in la) <= 10
    assert 5.4e5 < max(ratio(r) for r in lb) < 5.6e5
    assert all(0.94 <= float(r["c1lim_power"]) <= 0.99 for r in la)
    assert all(Fraction(r["bad_max"]) < Fraction(1, 5) for r in la)
    small24 = [r for r in A + B if r["large"] == "0" and r["n"] == "24"]
    assert small24 and all(Fraction(r["bad_avg"]) < Fraction(1, 10**4) for r in small24)
    assert all(Fraction(r["bad_max"]) == 1 and Fraction(r["pmin"]) == 0 for r in A + B if r["i"] in ("4", "5"))
    root = [r for r in A if r["i"] == "0"][0]
    assert round(float(Fraction(root["pmin"])), 3) == 0.403
    d1 = [r for r in A if r["i"] == "1"][0]
    assert round(float(Fraction(d1["pmin"])), 3) == 0.183
    pw = {sz: g9.chain_pointwise(24, sz, Fraction(1, 5), 5)["pmin"] for sz in ((32, 15), (32, 16), (32, 17))}
    assert min(pw, key=pw.get) == (32, 16) and pw[32, 16] == Fraction(d1["pmin"])
    assert round(comb(16, 8) / 2**16, 3) == 0.196
    # the dead-row case: a perfect fork of size 3 followed by size 2
    sz = (32, 15, 7, 3, 2)
    assert g9.chain_pointwise(24, sz, Fraction(1, 5), 5)["pmin"] == 0
    assert g9.row_table((3, 2), ((1, True),), (1,)) == {(0,): 2}
