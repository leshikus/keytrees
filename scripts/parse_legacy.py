"""Turn the saved 1999 RTA outputs into CSV tables under data/legacy.

Each output file ends each group of samples with lines of the form
``AVERAGE  <m>  <value>`` and ``DISPERSION  <value>``; the column is ``Gr-Hf``, the
excess of the greedy tree over the Huffman cost. Some runs stop with
``Error 108: Expected end of line``: GENDATA gave up finding distinct species at low test
counts, its error text reached RTA through the pipe, and RTA could not parse it. So a file
may cover fewer species than its name suggests.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

AVERAGE = re.compile(r"^AVERAGE\s+(?:\d+\s+)?(\d+)\s+(-?\d+\.\d+)")
DISPERSION = re.compile(r"^DISPERSION\s+(-?\d+\.\d+)")


def parse(path: Path) -> tuple[list[tuple[int, float, float]], bool]:
    rows: list[tuple[int, float, float]] = []
    pending: tuple[int, float] | None = None
    aborted = False
    for line in path.read_text("ascii", "replace").splitlines():
        if line.startswith("Error"):
            aborted = True
        if (match := AVERAGE.match(line)) is not None:
            pending = (int(match.group(1)), float(match.group(2)))
        elif (match := DISPERSION.match(line)) is not None and pending is not None:
            rows.append((pending[0], pending[1], float(match.group(1))))
            pending = None
    return rows, aborted


def main() -> int:
    results, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.csv"
    with summary.open("w", newline="", encoding="ascii") as handle:
        table = csv.writer(handle)
        table.writerow(["file", "groups", "m_min", "m_max", "aborted"])
        for folder in sorted(p for p in results.iterdir() if p.is_dir()):
            for path in sorted(folder.iterdir()):
                if not path.name.startswith("out"):
                    continue
                rows, aborted = parse(path)
                if not rows:
                    continue
                name = f"{folder.name}_{path.name}"
                with (out / f"{name}.csv").open("w", newline="", encoding="ascii") as sink:
                    writer = csv.writer(sink)
                    writer.writerow(["m", "greedy_minus_huffman", "standard_error"])
                    writer.writerows(rows)
                table.writerow([name, len(rows), rows[0][0], rows[-1][0], aborted])
                print(name, len(rows), rows[0][0], rows[-1][0], "aborted" if aborted else "complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
