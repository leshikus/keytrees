"""Generate figures and small LaTeX tables for the monograph from data/."""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "monograph" / "figures"
GEN = ROOT / "monograph" / "generated"
INK = "#222222"
COLORS = {"s1": "#b2182b", "s2": "#ef8a62", "s3": "#2166ac", "patricia": "#1b7837"}
LABELS = {"s1": r"$R_1$ (trie)", "s2": r"$R_2$", "s3": r"$R_3$", "patricia": r"$R_\infty$ (PATRICIA)"}

plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"}
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def bound_excess(m: float) -> float:
    return 29 / 28 - math.log2(2 * m) / m


def complete_redundancy(m: int) -> float:
    level = m.bit_length() - 1
    return level + 2 - 2 ** (level + 1) / m - math.log2(m)


def g_function() -> None:
    m = 100
    log_m = math.log2(m)
    A = math.log2(2 * m) * (1 - 1 / m)
    B = 4 / m**3 * (m + log_m - 1)

    def g(k: float) -> float:
        f = k / m * math.log2(2 * k) * (1 - 1 / k) + (m - k) / m * math.log2(2 * (m - k)) * (1 - 1 / (m - k))
        return A - B * k * (m - k) - f

    xs = [1 + i * (m - 2) / 800 for i in range(801)]
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    ax.plot(xs, [g(x) for x in xs], color=COLORS["s3"], linewidth=1.5)
    ax.axhline(0, color=INK, linewidth=0.6)
    ax.set_xlabel("$k$")
    ax.set_ylabel("$g(k)$")
    fig.savefig(FIG / "g_function.pdf")
    plt.close(fig)


def limit_redundancy() -> None:
    table = rows(DATA / "exact" / "limit.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    tails: dict[str, list[float]] = defaultdict(list)
    series: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    for row in table:
        m = int(row["m"])
        if m >= 4096:
            tails[row["variant"]].append(float(row["redundancy"]))
        if m <= 2048:
            series[row["variant"]][0].append(m)
            series[row["variant"]][1].append(float(row["redundancy"]))
    for name in ("s1", "s2", "s3", "patricia"):
        ax.plot(*series[name], color=COLORS[name], label=LABELS[name], linewidth=1.4)
    grid = sorted(set(series["s3"][0]))
    ax.plot(grid, [bound_excess(m) for m in grid], color=INK, linestyle="--", linewidth=1.0, label="bound of the theorem")
    ax.plot(grid, [complete_redundancy(m) for m in grid], color="#777777", linewidth=0.9, label="complete tree (free tests)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of words $m$")
    ax.set_ylabel(r"redundancy $t_m-\log_2 m$")
    ax.legend(frameon=False, fontsize=8, loc="center right", ncols=1)
    fig.savefig(FIG / "limit_redundancy.pdf")
    plt.close(fig)
    for name, values in tails.items():
        (GEN / f"limit_{name}.tex").write_text(f"${sum(values) / len(values):.5f}$%\n", encoding="ascii")


def rerun_vs_legacy() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.0), sharex=True)
    for ax, (coef, tag, legacy) in zip(axes, [("1.2", "1_2", "res2_out1_2"), ("1.5", "1_5", "res2_out1_5"), ("2", "2_0", "res2_out2_0")]):
        new = rows(DATA / "rerun" / f"uniform_{tag}.csv")
        old = rows(DATA / "legacy" / f"{legacy}.csv")
        ax.plot([int(r["m"]) for r in new], [float(r["up_minus_huffman"]) for r in new], color=COLORS["s3"], linewidth=1.1, label="re-run, 2026")
        ax.plot([int(r["m"]) for r in old], [float(r["greedy_minus_huffman"]) for r in old], color=COLORS["s1"], linewidth=0.9, alpha=0.85, label="saved run, 1999")
        ax.set_ylabel("excess over\ncomplete tree")
        ax.set_title(rf"$n=\lceil {coef}\,\log_2 m\rceil$ tests", fontsize=10, loc="left")
        for k in (8, 16, 32, 64):
            ax.axvline(k, color="#cccccc", linewidth=0.6, zorder=0)
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("number of species $m$")
    fig.savefig(FIG / "rerun_vs_legacy.pdf")
    plt.close(fig)


def gap_table() -> None:
    bands = [(6, 15), (16, 31), (32, 48)]
    lines = []
    for coef, tag in [("1.2", "1_2"), ("1.5", "1_5"), ("2.0", "2_0")]:
        table = rows(DATA / "rerun" / f"uniform_{tag}.csv")
        for low, high in bands:
            selected = [r for r in table if low <= int(r["m"]) <= high]
            mean = lambda key: sum(float(r[key]) for r in selected) / len(selected)
            greedy_opt = mean("up_minus_optimal")
            opt_free = sum(float(r["optimal"]) - float(r["huffman"]) for r in selected) / len(selected)
            random_free = sum(float(r["random"]) - float(r["huffman"]) for r in selected) / len(selected)
            lines.append(
                f"{coef} & {low}--{high} & {mean('up_minus_huffman'):.4f} & {greedy_opt:.4f} & {opt_free:.4f} & {random_free:.4f}\\\\"
            )
        lines.append(r"\addlinespace")
    (GEN / "gap_table.tex").write_text("\n".join(lines[:-1]) + "\n", encoding="ascii")


def random_prob_table() -> None:
    lines = []
    for coef, tag in [("1.2", "1_2"), ("1.5", "1_5"), ("2.0", "2_0")]:
        table = rows(DATA / "rerun" / f"random_{tag}.csv")
        mean = lambda fn: sum(fn(r) for r in table) / len(table)
        lines.append(
            f"{coef} & {mean(lambda r: float(r['greedy_up']) - float(r['huffman'])):.4f} & "
            f"{mean(lambda r: float(r['greedy_md']) - float(r['huffman'])):.4f} & "
            f"{mean(lambda r: float(r['huffman']) - float(r['entropy'])):.4f}\\\\"
        )
    (GEN / "random_prob_table.tex").write_text("\n".join(lines) + "\n", encoding="ascii")


def ecc_outputs() -> None:
    table = rows(DATA / "ecc" / "sweep.csv")
    index = {(int(r["m"]), int(r["t"]), int(r["errors"]), r["method"]): r for r in table}
    configs = sorted({(int(r["m"]), int(r["n"])) for r in table})

    # Table: no-error and t-error costs for each method.
    lines = []
    for m, n in configs:
        if m not in (16, 64, 256, 512):
            continue
        for t in (1, 2, 3):
            cell = lambda method, e: float(index[(m, t, e, method)]["mean_tests"]) if (m, t, e, method) in index else None
            fmt = lambda v: f"{v:.2f}" if v is not None else "---"
            bound = index[(m, t, 0, "elim_balance")]["volume_bound"]
            lines.append(
                f"{m} & {n} & {t} & {bound} & {fmt(cell('elim_berlekamp', 0))} & {fmt(cell('elim_berlekamp', t))} & "
                f"{fmt(cell('elim_balance', 0))} & {fmt(cell('elim_sequential', 0))} & {fmt(cell('sphere_word', 0))}\\\\"
            )
        lines.append(r"\addlinespace")
    (GEN / "ecc_table.tex").write_text("\n".join(lines[:-1]) + "\n", encoding="ascii")

    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    palette = {1: COLORS["s3"], 2: COLORS["patricia"], 3: COLORS["s1"]}
    ms = [m for m, _ in configs]
    for t in (1, 2, 3):
        no_error = [float(index[(m, t, 0, "elim_berlekamp")]["mean_tests"]) - math.log2(m) for m in ms]
        all_errors = [float(index[(m, t, t, "elim_berlekamp")]["mean_tests"]) - math.log2(m) for m in ms]
        bound = [float(index[(m, t, 0, "elim_berlekamp")]["volume_bound"]) - math.log2(m) for m in ms]
        ax.plot(ms, no_error, marker="o", markersize=3, color=palette[t], label=f"$t={t}$, no errors")
        ax.plot(ms, all_errors, marker="s", markersize=3, color=palette[t], linestyle=":", label=f"$t={t}$, $t$ errors")
        ax.plot(ms, bound, color=palette[t], linestyle="--", linewidth=0.8, label=f"$t={t}$, worst-case volume bound")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of species $m$")
    ax.set_ylabel(r"tests beyond $\log_2 m$")
    ax.legend(frameon=False, fontsize=7, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.savefig(FIG / "ecc_overhead.pdf")
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    g_function()
    limit_redundancy()
    rerun_vs_legacy()
    gap_table()
    random_prob_table()
    ecc_outputs()
    print("figures:", sorted(p.name for p in FIG.iterdir()))
    print("tables:", sorted(p.name for p in GEN.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
