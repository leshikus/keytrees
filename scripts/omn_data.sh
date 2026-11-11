#!/usr/bin/env bash
# Recompute data/omn (articles/omn-trees): all 0 <= m <= n <= 3, the three
# quantities in parallel (30 processes). Each run goes over o = 1, 2, ... and
# stops after the first o that takes more than 60 s (600 s for W with
# m = n = 3), at o = 4000, or when the run's overall limit expires (30 min for
# W, 45 min for W with m = n = 3, 15 min for A, 25 min for A*). Afterwards each
# o up to 265 beyond the W(3,3) run is tried alone, with 420 s each.
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S src/cpp -B build/cpp -G Ninja -DCMAKE_CXX_COMPILER="${CXX:-clang++}"
cmake --build build/cpp
cells=build/omn-cells
rm -rf "$cells" && mkdir -p "$cells"
for n in 0 1 2 3; do
  for m in $(seq 0 "$n"); do
    for spec in worst:1800 average:900 minmax:1500; do
      kind=${spec%%:*}
      limit=${spec##*:}
      per=60
      if [ "$kind$m$n" = worst33 ]; then limit=2700; per=600; fi
      (timeout "$limit" build/cpp/omn "$kind" "$m" "$n" 4000 "$per" \
        "$cells/${kind}_${m}_${n}.csv" > "$cells/${kind}_${m}_${n}.log" 2>&1 || true) &
    done
  done
done
wait
mkdir -p "$cells/single"
top=$(tail -n 1 "$cells/worst_3_3.csv" | cut -d, -f1)
seq $((top + 1)) 265 | xargs -P 24 -I{} sh -c \
  "timeout 420 build/cpp/omn wone 3 3 {} > $cells/single/{}.row 2>/dev/null || true"
.venv/bin/python scripts/omn_tables.py "$cells"
