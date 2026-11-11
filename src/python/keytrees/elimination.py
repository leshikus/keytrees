r"""
Exact quantities for the elimination key on random codes

Reference implementation for ``articles/elimination-keys`` (new in 2026). A code is
`m` independent uniform binary words; the elimination key drops a species after
`t + 1` disagreements. With tests in a fixed order a wrong species survives `r`
tests with probability `p_r = V(r, t) 2^{-r}`, independently of the others, which
gives the exact mean cost of Theorem 1 of the article.

EXAMPLES::

    >>> volume(5, 2)
    16
    >>> survival(3, 1)
    0.5
    >>> round(order_cost(2**10, 1), 4)
    15.303
"""
from __future__ import annotations

import math
from math import comb

EULER_GAMMA = 0.5772156649015329
MELLIN_CONSTANT = EULER_GAMMA / math.log(2) + 0.5


def volume(r: int, t: int) -> int:
    r"""
    Return `V(r, t) = \sum_{i \le t} \binom{r}{i}`, with `V(r, t) = 0` for `t < 0`.

    EXAMPLES::

        >>> [volume(4, t) for t in range(-1, 5)]
        [0, 1, 5, 11, 15, 16]
    """
    if t < 0:
        return 0
    return sum(comb(r, i) for i in range(min(t, r) + 1))


def survival(r: int, t: int) -> float:
    r"""
    Return `p_r = V(r, t) 2^{-r}`, the chance that a uniform word stays within `t`
    disagreements of a fixed answer word on `r` tests.

    EXAMPLES::

        >>> survival(0, 0), survival(2, 1), survival(10, 0)
        (1.0, 0.75, 0.0009765625)
    """
    return math.ldexp(float(volume(r, t)), -r)


def tail(r: int, t: int, m: int) -> float:
    r"""
    Return `P(T > r) = 1 - (1 - p_r)^{m-1}` for the fixed-order key.

    EXAMPLES::

        >>> tail(0, 1, 5), tail(1, 0, 2)
        (1.0, 0.5)
    """
    p = survival(r, t)
    if p >= 1.0:
        return 1.0
    return -math.expm1(float(m - 1) * math.log1p(-p))


def order_cost(m: int, t: int, n: int | None = None) -> float:
    r"""
    Return the exact mean number of tests of the fixed-order elimination key on
    `m` uniform words of length `n` (``None`` for unbounded length).

    The same value holds for any error pattern of weight at most `t` chosen
    independently of the wrong words (Theorem 1).

    EXAMPLES::

        >>> order_cost(2, 0)
        2.0
        >>> round(order_cost(2**20, 0) - 20 - MELLIN_CONSTANT, 5)
        0.0
    """
    if m < 2:
        return 0.0
    total, r = 0.0, 0
    while n is None or r < n:
        term = tail(r, t, m)
        total += term
        r += 1
        if r > 2 * t + 4 and term < 1e-17:
            break
    return total


def adversarial_order_cost(m: int, t: int, n: int | None = None) -> float:
    r"""
    Return the exact mean of the largest number of tests over all error patterns
    of weight at most `t`, for the fixed-order key: the threshold becomes `2t`.

    EXAMPLES::

        >>> adversarial_order_cost(64, 1) == order_cost(64, 2)
        True
    """
    return order_cost(m, 2 * t, n)


def beta(r: int, t: int) -> float:
    r"""
    Return `\beta_r = V(r, t-1) / V(r, t)`, so that `p_{r+1} = p_r (1 + \beta_r)/2`.

    EXAMPLES::

        >>> beta(3, 1), beta(0, 2)
        (0.25, 1.0)
    """
    return volume(r, t - 1) / volume(r, t)


def threshold(m: int, t: int, count: int | None = None) -> int:
    r"""
    Return `r_0`, the least `r` with `count \cdot V(r, t) \le 2^r`; by default
    ``count = m - 1``.

    EXAMPLES::

        >>> threshold(2**10, 0), threshold(2**10, 1)
        (10, 14)
    """
    c = m - 1 if count is None else count
    r = 0
    while c * volume(r, t) > 2**r:
        r += 1
    return r


def order_bounds(m: int, t: int) -> tuple[float, float]:
    r"""
    Return the explicit lower and upper bounds of Theorem 1 on the mean cost.

    The lower bound uses `s = \lceil r_0/2 \rceil` (or `s = t` if larger) and
    `\lambda = 2/(1+\beta_s)`.

    EXAMPLES::

        >>> lo, hi = order_bounds(2**12, 2)
        >>> lo <= order_cost(2**12, 2) <= hi
        True
    """
    r0 = threshold(m, t)
    upper = r0 + 2 / (1 - beta(r0, t))
    s = max(t, (r0 + 1) // 2)
    lam = 2 / (1 + beta(s, t))
    deficit = sum(math.exp(-lam**k) for k in range(r0 - s)) + s * math.exp(-lam ** (r0 - 1 - s))
    return r0 - deficit, upper


def real_volume(x: float, t: int) -> float:
    r"""
    Return `\sum_{i \le t} \binom{x}{i}` for real `x`.

    EXAMPLES::

        >>> real_volume(4.0, 2)
        11.0
    """
    total, c = 0.0, 1.0
    for i in range(t + 1):
        total += c
        c = c * (x - i) / (i + 1)
    return total


def x_hat(m: int, t: int) -> float:
    r"""
    Return the real root `\hat x > 2t` of `2^x = (m-1) V(x, t)`.

    EXAMPLES::

        >>> round(x_hat(2**10 + 1, 0), 10)
        10.0
    """
    target = math.log2(m - 1)
    x = target + t * math.log2(max(target, 2.0)) + 2 * t + 1
    for _ in range(200):
        x = target + math.log2(real_volume(x, t))
    return x


def fluctuation_bound(terms: int = 20) -> float:
    r"""
    Return `(2/\ln 2) \sum_{k \ge 1} |\Gamma(2\pi i k/\ln 2)|`, a bound on the
    periodic term `P` of Theorem 2, using `|\Gamma(iy)|^2 = \pi/(y \sinh \pi y)`.

    EXAMPLES::

        >>> 1.5e-6 < fluctuation_bound() < 1.6e-6
        True
    """
    total = 0.0
    for k in range(1, terms + 1):
        y = 2 * math.pi * k / math.log(2)
        total += math.sqrt(math.pi / (y * math.sinh(math.pi * y)))
    return 2 / math.log(2) * total


def harmonic_sum(x: float) -> float:
    r"""
    Return `S(x) = \sum_{r \ge 0} (1 - e^{-2^{x-r}})`.

    EXAMPLES::

        >>> abs(harmonic_sum(30.25) - 30.25 - MELLIN_CONSTANT) < 2e-6
        True
    """
    total, r = 0.0, 0
    while True:
        term = -math.expm1(-(2.0 ** (x - r)))
        total += term
        r += 1
        if r > x + 2 and term < 1e-18:
            return total


def kraft_threshold(m: int, t: int) -> int:
    r"""
    Return `r_m`, the least `D` with `m V(D, t) \le 2^D`. Theorem 3: every key that
    identifies under at most `t` errors has mean worst-case cost above `r_m - 1`,
    once `r_m - 1 \ge D_t`.

    EXAMPLES::

        >>> kraft_threshold(2**10, 1)
        14
    """
    return threshold(m, t, count=m)


def kraft_start(t: int) -> int:
    r"""
    Return `D_t`, the least `D \ge 5t` such that `(8/5)^{D'-5t} \ge 1 + D'/2` for
    all `D' \ge D`.

    EXAMPLES::

        >>> [kraft_start(t) for t in range(4)]
        [0, 9, 15, 21]
    """
    def holds(d: int) -> bool:
        return 1.6 ** (d - 5 * t) >= 1 + d / 2

    d = 10 * t + 200
    while d > 5 * t and holds(d - 1):
        d -= 1
    return d


def oracle_bound(m: int, n: int, t: int) -> float:
    r"""
    Return `\sum_{d<n} (1 - \binom{n}{d}(1-p_d)^{m-1})^+`, a lower bound on the mean
    cost of every rule, even one told the true species, with no errors
    (Theorem 4).

    EXAMPLES::

        >>> oracle_bound(2**10, 40, 1) < order_cost(2**10, 1)
        True
    """
    total = 0.0
    for d in range(n):
        p = survival(d, t)
        if p >= 1.0:
            total += 1.0
            continue
        if d > 2 * t + 4 and (m - 1) * p < 1e-18:
            break  # each later term is at most (m-1) p_d, a geometric tail below 1e-17
        log_term = math.lgamma(n + 1) - math.lgamma(d + 1) - math.lgamma(n - d + 1) + (m - 1) * math.log1p(-p)
        if log_term >= 0:
            continue
        total += -math.expm1(log_term)
    return total


def free_no_error_cost(m: int, t: int) -> int:
    r"""
    Return `\lceil \log_2 m \rceil + t`, the no-error cost of the free-question key
    of Proposition 5 for every species.

    EXAMPLES::

        >>> free_no_error_cost(1000, 3)
        13
    """
    return (m - 1).bit_length() + t


def exact_rows() -> list[dict[str, float | int]]:
    r"""
    Return the rows of ``data/elimination-keys/exact.csv``: for `t \le 3` and
    `m = 2^L`, the exact fixed-order mean (unbounded length), the bounds of
    Theorem 1, the approximation `\hat x + \gamma/\ln 2 + 1/2` of Theorem 2, the
    adversarial mean, the threshold `r_m - 1` of Theorem 3, and the bound of
    Theorem 4 for `n = 4L` and `n = m` (capped at `2^{20}`).

    EXAMPLES::

        >>> rows = exact_rows()
        >>> rows[0]["t"], rows[0]["L"]
        (0, 4)
    """
    rows = []
    for t in range(4):
        for L in (4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 40, 64, 80):
            m = 2**L
            lo, hi = order_bounds(m, t)
            rows.append({
                "t": t, "L": L,
                "order_mean": order_cost(m, t),
                "thm1_lower": lo, "thm1_upper": hi,
                "r0": threshold(m, t),
                "thm2_approx": x_hat(m, t) + MELLIN_CONSTANT,
                "adversarial_mean": adversarial_order_cost(m, t),
                "kraft_bound": kraft_threshold(m, t) - 1,
                "oracle_bound_4L": oracle_bound(m, 4 * L, t),
                "oracle_bound_linear": oracle_bound(m, min(m, 2**20), t),
            })
    return rows


def read_sweep(path: str) -> list[dict[str, str]]:
    r"""
    Read ``data/elimination-keys/sweep.csv`` written by ``elimination sweep``.
    """
    import csv

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def overhead_slope(rows: list[dict[str, str]], regime: str, rule: str, t: int,
                   errors: str = "none", min_log: int = 8) -> float:
    r"""
    Return the least-squares slope of ``mean - log2 m`` against ``log2 log2 m``
    over the sweep rows with ``log2 m >= min_log``.

    EXAMPLES::

        >>> rows = [{"regime": "r", "rule": "x", "t": "1", "errors": "none", "m": str(2**L),
        ...          "mean_tests": str(L + 2 * math.log2(L))} for L in (8, 16, 32)]
        >>> round(overhead_slope(rows, "r", "x", 1), 6)
        2.0
    """
    pts = []
    for r in rows:
        if r["regime"] == regime and r["rule"] == rule and int(r["t"]) == t and r["errors"] == errors:
            L = int(r["m"]).bit_length() - 1
            if L >= min_log:
                pts.append((math.log2(L), float(r["mean_tests"]) - L))
    k = len(pts)
    mx = sum(x for x, _ in pts) / k
    my = sum(y for _, y in pts) / k
    return sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)


def sweep_table(rows: list[dict[str, str]]) -> str:
    r"""
    Return the LaTeX table of the article from the sweep rows.
    """
    def get(regime, L, t, rule, errors="none"):
        for r in rows:
            if (r["regime"], int(r["m"]), int(r["t"]), r["rule"], r["errors"]) == (regime, 2**L, t, rule, errors):
                return float(r["mean_tests"])
        raise KeyError((regime, L, t, rule, errors))

    lines = [r"\begin{table}[ht]", r"\centering\small",
             r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{order} & \multicolumn{3}{c}{weighted} & leader & oracle & Thm.~\ref{thm:oracle}\\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
             r"$n$ & $\log m$ & $t$ & exact & sim. & none & random & first & none & none & bound\\", r"\midrule"]
    for regime, label, Ls in (("4log", r"$4\log m$", (8, 12, 16)), ("linear", r"$m$", (8, 11))):
        for L in Ls:
            n = 4 * L if regime == "4log" else 2**L
            for t in (1, 2, 3):
                vals = [order_cost(2**L, t, n), get(regime, L, t, "order"), get(regime, L, t, "weighted"),
                        get(regime, L, t, "weighted", "random"), get(regime, L, t, "weighted", "first"),
                        get(regime, L, t, "leader"), get(regime, L, t, "oracle"), oracle_bound(2**L, n, t)]
                lines.append(f"{label} & {L} & {t} & " + " & ".join(f"{v:.2f}" for v in vals) + r"\\")
        lines.append(r"\midrule" if regime == "4log" else r"\bottomrule")
    lines += [r"\end{tabular}",
              r"\caption{Mean number of tests on random codes (simulation, 8 codes $\times$ 75 specimens), "
              r"except the columns ``exact'' (Theorem~\ref{thm:order}) and ``bound'' (Theorem~\ref{thm:oracle}, "
              r"valid even for the oracle). Errors: none, $t$ at random positions, or on the first $t$ tests asked. "
              r"Data: \texttt{data/elimination-keys/sweep.csv}.}\label{tab:sweep}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def exact_order_mean(m: int, n: int, t: int) -> "Fraction":
    r"""
    Return the exact rational `\sum_{r<n} (1 - (1-p_r)^{m-1})` of Theorem 1.

    EXAMPLES::

        >>> exact_order_mean(3, 4, 1)
        Fraction(59, 16)
    """
    from fractions import Fraction

    total = Fraction(0)
    for r in range(n):
        total += 1 - (1 - Fraction(volume(r, t), 2**r)) ** (m - 1)
    return total


def exact_table(rows: list[dict[str, str]], sizes: list[tuple[str, int, int, int]]) -> str:
    r"""
    Return the LaTeX table of exact means for ``sizes``, a list of
    ``(model, m, n, t)``, from the rows of ``exact_means.csv``.
    """
    index = {(r["model"], int(r["m"]), int(r["n"]), int(r["t"]), r["errors"], r["rule"]): float(r["mean"]) for r in rows}

    def cell(key):
        return f"{index[key]:.3f}" if key in index else "--"

    lines = [r"\begin{table}[ht]", r"\centering\small", r"\begin{tabular}{lrrrrrrrrrrr}", r"\toprule",
             r" & & & & \multicolumn{5}{c}{no errors} & $t$ errors & \multicolumn{2}{c}{worst}\\",
             r"\cmidrule(lr){5-9}\cmidrule(lr){10-10}\cmidrule(lr){11-12}",
             r"tables & $m$ & $n$ & $t$ & order & weighted & leader & optimal & oracle & weighted & order & weighted\\",
             r"\midrule"]
    for model, m, n, t in sizes:
        base = (model, m, n, t)
        vals = [cell(base + ("0", "order")), cell(base + ("0", "weighted")), cell(base + ("0", "leader")),
                cell(base + ("0", "optimal")), cell(base + ("0", "oracle")), cell(base + (str(t), "weighted")),
                cell(base + ("worst", "order")), cell(base + ("worst", "weighted"))]
        lines.append(f"{model} & {m} & {n} & {t} & " + " & ".join(vals) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Exact mean number of tests, over all tables of the given kind, all species and all error "
              r"sets of the given size, rounded to three decimals. A dash: the optimal key was not computed for this size. "
              r"Exact fractions: \texttt{data/elimination-keys/exact\_means.csv}.}\label{tab:exactmeans}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import csv
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "table":
        sys.stdout.write(sweep_table(read_sweep(sys.argv[2])))
        sys.exit(0)
    out = sys.argv[1] if len(sys.argv) > 1 else "data/elimination-keys/exact.csv"
    rows = exact_rows()
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
