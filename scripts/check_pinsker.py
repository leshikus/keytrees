"""Independently reproduce 'Best Pinsker bound equals Taylor polynomial of degree 49'.

Writes data/pinsker/coefficients.csv, R48.csv and summary.txt. Values of R are sampled, not interval bounds.
"""
from __future__ import annotations

import csv
import sys
import time
from fractions import Fraction
from pathlib import Path

import mpmath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))
from keytrees.pinsker import remainder, taylor_coefficients  # noqa: E402

OUT = ROOT / "data" / "pinsker"
EXTENDED = 400


def tail_polynomial_positive(c: dict[int, Fraction]) -> tuple[bool, int]:
    """Lemma 3, t > 8: t - sum (nu+1) t_nu (2-1/t)^nu > 0 after multiplying by t^47.

    Written as a polynomial in x = t - 8, all coefficients should be positive.
    """
    # Polynomial in t: t^48 - sum_nu (nu+1) t_nu (2t-1)^(nu) t^(47+1-nu) ... uses degree 48 in nu - 1 form.
    from math import comb

    degree = 48
    poly = [Fraction(0)] * (degree + 1)
    poly[degree] += 1  # t * t^47
    for nu, value in c.items():
        if nu > 48:
            continue
        # dL/dV term of T48 is nu t_nu V^(nu-1); (2-1/t)^(nu-1) t^47 = (2t-1)^(nu-1) t^(48-nu)
        k = nu - 1
        for j in range(k + 1):
            poly[j + 48 - nu] -= nu * value * comb(k, j) * 2**j * (-1) ** (k - j)
    # Taylor shift to x = t - 8.
    shifted = [Fraction(0)] * (degree + 1)
    for i, a in enumerate(poly):
        for j in range(i + 1):
            shifted[j] += a * comb(i, j) * 8 ** (i - j)
    return all(x > 0 for x in shifted), sum(1 for x in shifted if x <= 0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    c = taylor_coefficients(EXTENDED)
    lines = [f"coefficients to degree {EXTENDED} in {time.time() - start:.1f}s"]
    with (OUT / "coefficients.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(["degree", "numerator", "denominator", "value"])
        for nu in sorted(c):
            writer.writerow([nu, c[nu].numerator, c[nu].denominator, f"{float(c[nu]):.6e}"])

    negative = [nu for nu in sorted(c) if c[nu] < 0]
    positive = [nu for nu in sorted(c) if c[nu] > 0]
    lines.append(f"t2..t60 positive: {all(c[n] > 0 for n in range(2, 61, 2))}")
    lines.append(f"t62..t96 negative: {all(c[n] < 0 for n in range(62, 97, 2))}")
    blocks, current, sign = [], 2, c[2] > 0
    for nu in range(4, EXTENDED + 1, 2):
        if (c[nu] > 0) != sign:
            blocks.append((current, nu - 2, "+" if sign else "-"))
            current, sign = nu, c[nu] > 0
    blocks.append((current, EXTENDED, "+" if sign else "-"))
    lines.append(f"sign blocks to degree {EXTENDED}: {blocks}")
    lines.append(f"counts: {len(positive)} positive, {len(negative)} negative")

    mpmath.mp.dps = 150
    # Theorem 2: R50 is negative near t = 9.92.
    values = [mpmath.mpf("9.9212") + i * mpmath.mpf("0.00001") for i in range(11)]
    r50 = [remainder(t, c, 50, mpmath) for t in values]
    lines.append(f"R50 on [9.9212, 9.9213]: [{float(min(r50)):.5f}, {float(max(r50)):.5f}]")

    # Scan R48 and R50 over t.
    grid = [mpmath.mpf(i) / 200 for i in range(1, 8001)]  # up to t = 40
    r48 = [remainder(t, c, 48, mpmath) for t in grid]
    least = min(r48)
    lines.append(f"R48 sampled on (0, 40], 8000 points: all positive = {all(x > 0 for x in r48)}; minimum {mpmath.nstr(least, 5)}")
    r50all = [remainder(t, c, 50, mpmath) for t in grid]
    at = min(range(len(grid)), key=lambda i: r50all[i])
    lines.append(f"R50 sampled: minimum {mpmath.nstr(r50all[at], 5)} at t={float(grid[at])}")
    lines.append(f"limit of normalised derivative, 50*t50 = {float(50 * c[50]):.4e}")

    ok, bad = tail_polynomial_positive(c)
    lines.append(f"t > 8: shifted polynomial has all positive coefficients = {ok} ({bad} not positive)")

    with (ROOT / "data" / "pinsker" / "R48.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "R48", "R50"])
        for i in range(0, 2400, 4):
            writer.writerow([float(grid[i]), float(r48[i]), float(r50all[i])])

    text = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(text, "ascii")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
