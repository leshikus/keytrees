#!/usr/bin/env python3
"""Re-check the printed constants of articles/gradient-bound from scratch.

    .venv/bin/python scripts/verify_certificates.py           # checksums, c_K, mu, t^+ sample (about a minute)
    .venv/bin/python scripts/verify_certificates.py --full    # also recompute c_b (r_b up to s = 4000, slow)
    .venv/bin/python scripts/verify_certificates.py --write-checksums   # after regenerating the data

The script does not trust the stored decimals. It recomputes kappa_K in integers (and re-checks
its defining inequality in exact fractions), rho_{kappa_K} and the tails in mpmath interval
arithmetic, and compares the upper endpoints with the printed constants as exact rationals.
The versions it was run with are compared with requirements-certificates.txt. Exit status 0
means every check passed.
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import csv
import hashlib
import math
import os
import platform
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))

import mpmath  # noqa: E402
from mpmath import iv  # noqa: E402

import keytrees.gradient_bound as gb  # noqa: E402
import keytrees.gradient_tplus as gt  # noqa: E402

DATA = ROOT / "data" / "gradient-bound"
SUMS = DATA / "SHA256SUMS"
PINS = ROOT / "requirements-certificates.txt"
ENGINE = Path(os.environ.get("KEYTREES_BUILD", ROOT / "build" / "cpp")) / "gradient_tplus"

# The constants as printed in the article (constants.tex); checked against that file when it is present.
PRINTED = {"cGtwenty": "0.316", "cGthirty": "0.194", "cGforty": "0.159", "cGsixty": "0.127",
           "cGthirtyfull": "0.196", "cyltail": "0.0176", "cbtwo": "0.181", "cbthree": "0.129",
           "cbfour": "0.109", "cortwo": "0.197", "corthree": "0.145", "tmu": "0.0095"}
K_NAMES = {20: "cGtwenty", 30: "cGthirty", 40: "cGforty", 60: "cGsixty"}
B_NAMES = {2: "cbtwo", 3: "cbthree", 4: "cbfour"}
# t^+ cells of the article's table, one or more per winning ingredient: (n, m) -> printed t^+ - log m
TPLUS_SAMPLE = {(6, 4): "0.013", (12, 12): "0.082", (48, 8): "0.001", (256, 24): "0.082",
                (8, 32): "0.600", (16, 24): "0.282", (12, 64): "0.518", (12, 1024): "2.000"}

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        failures.append(what)


def frac(x) -> Fraction:
    man, exp = mpmath.mpf(x).man_exp
    return Fraction(man) * Fraction(2) ** exp


def iv_add_upper(*xs) -> Fraction:
    iv.prec = 64
    total = iv.mpf(0)
    for x in xs:
        total += iv.mpf(x.numerator) / x.denominator if isinstance(x, Fraction) else iv.mpf(x)
    return frac(total.b)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_files() -> list[Path]:
    return sorted(p for p in DATA.glob("*.csv"))


def write_checksums() -> None:
    SUMS.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in data_files()))
    print(f"wrote {SUMS.relative_to(ROOT)}")


def checksums() -> None:
    listed = {}
    for line in SUMS.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        listed[name] = digest
    check(set(listed) == {p.name for p in data_files()}, "SHA256SUMS lists exactly the data files")
    for name, digest in sorted(listed.items()):
        path = DATA / name
        check(path.exists() and sha256(path) == digest, f"checksum of {name}")


def versions() -> None:
    pins = dict(re.findall(r"(?m)^([A-Za-z0-9_-]+)==(\S+)", PINS.read_text()))
    py = re.search(r"(?m)^# python (\S+)", PINS.read_text()).group(1)
    have = {"mpmath": mpmath.__version__}
    print(f"python {platform.python_version()} (pinned {py}), mpmath {mpmath.__version__} (pinned {pins['mpmath']})")
    if have["mpmath"] != pins["mpmath"] or platform.python_version() != py:
        print("note  versions differ from requirements-certificates.txt; the checks below still apply")


def printed_macros() -> None:
    codes = Path(os.environ.get("CODES_DIR", ROOT.parent / "codes"))
    path = codes / "articles" / "gradient-bound" / "constants.tex"
    if not path.exists():
        print("skip  constants.tex not found (codes repository not checked out); using the values in this script")
        return
    mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([0-9.]+)\}", path.read_text()))
    check(all(mac.get(k) == v for k, v in PRINTED.items()), "printed values in this script equal constants.tex")


def rows(name: str) -> list[dict]:
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def gradient_constants() -> dict[int, Fraction]:
    tail = gb.cylinder_tail(500, 1)
    check(Fraction(PRINTED["cyltail"]) >= frac(tail), f"4/(500(1-theta)) <= {PRINTED['cyltail']}")
    stored = {int(r["K"]): r for r in rows("gradient_constants.csv")}
    out = {}
    for K, name in K_NAMES.items():
        kap = gb.kappa_column(K, 500)
        ok = all(kap[s] == 1 or gb.fail_probability(s, kap[s]) ** K <= Fraction(1, 2 ** (s + 3)) for s in range(2, 501))
        check(ok, f"K={K}: kappa_K(s) satisfies f_k(s)^K <= 2^-(s+3) for s <= 500 (exact fractions)")
        up = gb.rho_upper(kap, 500)
        c = iv_add_upper(max(up), tail)
        out[K] = c
        printed = Fraction(PRINTED[name])
        check(printed >= c, f"K={K}: c_K <= {PRINTED[name]} (upper endpoint {float(c):.10f}, gap {float(printed - c):.2e})")
        check(abs(Fraction(stored[K]["c_K_upper"]) - c) <= Fraction(1, 10**8), f"K={K}: stored c_K agrees to 1e-8")
    full = out[30] + Fraction(max(Fraction(26, 16000), Fraction(26, 1000 * 500**2) + Fraction(1, 2**10))) / (1 - Fraction(1, 2**57))
    check(Fraction(PRINTED["cGthirtyfull"]) > full, f"c_30 + error term < {PRINTED['cGthirtyfull']} ({float(full):.6f})")
    return out


def blocked_constants(full: bool) -> None:
    stored = {int(r["b"]): r for r in rows("blocked_constants.csv")}
    alpha = float(stored[2]["alpha"])
    check(gb.best_tail_alpha(4000) == alpha, f"alpha = {alpha} minimises the tail constant on the grid")
    mu = frac(gb.tail_mu(4000, alpha))
    check(Fraction(PRINTED["tmu"]) >= mu, f"mu <= {PRINTED['tmu']} ({float(mu):.8f})")
    for b, name in B_NAMES.items():
        if full:
            r = max(gb.chain_upper(b, 4000))
            check(abs(Fraction(stored[b]["max_r_upper"]) - frac(r)) <= Fraction(1, 10**8), f"b={b}: stored max r_b agrees to 1e-8")
        else:
            r = Fraction(stored[b]["max_r_upper"]) + Fraction(5, 10**9)  # stored to 8 decimals
        c = iv_add_upper(r, mu)
        printed = Fraction(PRINTED[name])
        check(printed >= c, f"b={b}: c_b <= {PRINTED[name]} ({'recomputed' if full else 'stored max r_b'}, gap {float(printed - c):.2e})")
        if b < 4:
            cor = (c + Fraction(1, 64)) / (1 - Fraction(1, 2**13))
            cname = {2: "cortwo", 3: "corthree"}[b]
            check(Fraction(PRINTED[cname]) >= cor, f"b={b}: (c_b + 1/64)/(1 - 2^-13) <= {PRINTED[cname]}")
    if not full:
        print("note  c_b used the stored max r_b; run with --full to recompute it (r_b for s <= 4000)")


def log2_lower(m: int) -> Fraction:
    iv.prec = 120
    return frac((iv.log(iv.mpf(m)) / iv.log(2)).a)


def tplus_sample() -> None:
    grid = {(int(r["n"]), int(r["m"])): r for r in rows("tplus.csv")}
    exact = {(int(r["n"]), int(r["m"])): Fraction(int(r["numerator"]), int(r["denominator"]))
             for r in rows("exact_formula.csv")}
    for (n, m), printed in TPLUS_SAMPLE.items():
        r = grid[n, m]
        ing = r["ingredient"]
        if ing == "exact":
            upper = exact[n, m]
        elif ing == "fixed":
            upper = frac(gt.fixed_bound(n, m).b)
        elif ing == "trivial":
            upper = gt.trivial_bound(n, m)
        elif ing == "cylinder":
            if not ENGINE.exists():
                print(f"skip  ({n},{m}): build/cpp/gradient_tplus is needed for the requirement of (f)")
                continue
            p = dict(t.split("=") for t in r["cylinder_params"].replace("(s1=m)", "").split())
            out = subprocess.run([str(ENGINE), "kappa", str(n), str(m), p["j"], "--data", str(DATA)],
                                 capture_output=True, text=True, check=True).stdout.split()
            s1 = int(p["s1"])
            kap = ([0, 0] + [int(v) for v in out[1::2]])[: min(s1, m) + 1]
            upper = frac(gt.cylinder_bound(n, m, kap, s1, float(p["x"])).b)
        else:
            raise ValueError(ing)
        stored = Fraction(r["tplus"])
        check(stored >= upper and Fraction(printed) >= upper - log2_lower(m),
              f"t^+({n},{m}), ingredient {ing}: printed {printed} >= recomputed bound - log m")


def main() -> int:
    if sys.argv[1:] == ["--write-checksums"]:
        write_checksums()
        return 0
    full = sys.argv[1:] == ["--full"]
    if sys.argv[1:] not in ([], ["--full"]):
        print(__doc__)
        return 2
    versions()
    checksums()
    printed_macros()
    gradient_constants()
    blocked_constants(full)
    tplus_sample()
    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
