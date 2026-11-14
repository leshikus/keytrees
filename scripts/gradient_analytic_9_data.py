#!/usr/bin/env python3
"""Data for Step 9 of the analytic estimate (articles/gradient-bound/analytic/09-exact.tex).

    .venv/bin/python scripts/gradient_analytic_9_data.py formula <threads> [timeout]
        extend data/gradient-bound/exact_formula.csv by the cells of FORMULA_CELLS,
        cheapest first; a run longer than <timeout> seconds (default 1800) is stopped
        and the cell is listed in data/gradient-bound/analytic/formula_out_of_reach.csv
    .venv/bin/python scripts/gradient_analytic_9_data.py enum <threads>
        exact enumeration of all n x m matrices (with symmetry) for the cells of
        ENUM_CELLS; stores the engine output in data/gradient-bound/analytic/enum/
    .venv/bin/python scripts/gradient_analytic_9_data.py data
        the tables data/gradient-bound/analytic/*.csv from exact_formula.csv, the
        enumerations and the product formula
    .venv/bin/python scripts/gradient_analytic_9_data.py tex
        tables for 09-exact.tex, written to $CODES_DIR/articles/gradient-bound/analytic/
        (CODES_DIR defaults to ../codes)

Every value is exact; nothing is sampled.
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import csv
import gzip
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from gradient_bound_data import combine, tuple_count  # noqa: E402

from keytrees import gradient_analytic_9 as g9  # noqa: E402

DATA = ROOT / "data" / "gradient-bound"
OUT = DATA / "analytic"
ENGINE = ROOT / "build" / "cpp" / "gradient_bound"
ENUM = ROOT / "build" / "cpp" / "gradient_enum"

# New cells of the path formula: (m, first n, last n).
FORMULA_RANGES = [
    (17, 10, 24), (18, 10, 24), (19, 10, 24), (20, 10, 16), (21, 10, 14), (22, 10, 13),
    (23, 11, 12), (24, 11, 12), (25, 11, 11), (26, 9, 9), (27, 9, 9), (28, 8, 8),
    (29, 8, 8), (30, 7, 7), (31, 7, 7), (32, 7, 7),
]


def formula(threads: str, timeout: float = 1800.0) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = DATA / "exact_formula.csv"
    done = {(int(r["n"]), int(r["m"])) for r in csv.DictReader(open(path))}
    miss_path = OUT / "formula_out_of_reach.csv"
    missed = set()
    if miss_path.exists():
        missed = {(int(r["n"]), int(r["m"])) for r in csv.DictReader(open(miss_path))}
    else:
        miss_path.write_text("n,m,tuples,timeout_seconds\n")
    cells = [(n, m) for m, a, b in FORMULA_RANGES for n in range(a, b + 1)]
    cells = [c for c in cells if c not in done and c not in missed]
    cells.sort(key=lambda c: tuple_count(*c) * c[0])
    log = OUT / "formula_runs.csv"
    if not log.exists():
        log.write_text("n,m,tuples,seconds,threads\n")
    for n, m in cells:
        t0 = time.time()
        try:
            out = subprocess.run([str(ENGINE), "formula", str(n), str(m), threads], capture_output=True,
                                 text=True, check=True, timeout=timeout).stdout
        except subprocess.TimeoutExpired:
            with open(miss_path, "a") as f:
                f.write(f"{n},{m},{tuple_count(n, m)},{int(timeout)}\n")
            continue
        _, _, value = combine(out)
        with open(path, "a") as f:
            f.write(f"{n},{m},{value.numerator},{value.denominator}\n")
        with open(log, "a") as f:
            f.write(f"{n},{m},{tuple_count(n, m)},{time.time() - t0:.1f},{threads}\n")
        print(n, m, f"{time.time() - t0:.1f}s", flush=True)


def enum(threads: str) -> None:
    """Run gradient_enum sym on ENUM_CELLS; the output is stored gzipped in data/."""
    raw = OUT / "enum"
    raw.mkdir(parents=True, exist_ok=True)
    for n, m in g9.ENUM_CELLS:
        path = raw / f"enum_{n}_{m}.txt.gz"
        if path.exists():
            continue
        t0 = time.time()
        text = subprocess.run([str(ENUM), "sym", str(n), str(m), threads], capture_output=True, text=True,
                              check=True).stdout
        with gzip.open(path, "wt", compresslevel=9) as f:
            f.write(text)
        print(n, m, f"{time.time() - t0:.1f}s", flush=True)


def data() -> None:
    g9.write_data(DATA, OUT / "enum")


def tex() -> None:
    codes = Path(os.environ.get("CODES_DIR", ROOT.parent / "codes"))
    g9.write_tex(DATA / "analytic", codes / "articles" / "gradient-bound" / "analytic" / "09-tables.tex")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["formula"] and len(a) in (2, 3):
        formula(a[1], float(a[2]) if len(a) == 3 else 1800.0)
    elif a[:1] == ["enum"] and len(a) == 2:
        enum(a[1])
    elif a == ["data"]:
        data()
    elif a == ["tex"]:
        tex()
    else:
        print(__doc__)
        sys.exit(2)
