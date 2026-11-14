#!/usr/bin/env bash
# Rebuild everything from the repository on Linux: C++ engines, data, tests,
# figures and the PDF. Run from anywhere; takes a few minutes.
# Set CXXFLAGS=-stdlib=libc++ if the default C++ library lacks <print>.
set -euo pipefail
cd "$(dirname "$0")/.."
# The monograph and the decoded artifacts are in the codes repository.
CODES_DIR=${CODES_DIR:-../codes}
export CODES_DIR

cmake -S src/cpp -B build/cpp -G Ninja -DCMAKE_CXX_COMPILER="${CXX:-clang++}"
cmake --build build/cpp

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -e '.[dev]'

mkdir -p data/exact data/ecc data/rerun data/legacy
build/cpp/exact_cost brute data/exact/brute.csv
build/cpp/exact_cost identity 40
build/cpp/exact_cost limit 8192 data/exact/limit.csv
build/cpp/exact_cost recurrence 40 2048 data/exact/recurrence.csv
build/cpp/ecc sweep 20261109 150 data/ecc/sweep.csv
seed=19990325
for c in 1.2 1.5 2.0; do
  tag=${c/./_}
  build/cpp/rta sweep "$c" 6 127 200 "$seed" "data/rerun/uniform_$tag.csv" uniform 48; seed=$((seed + 1))
  build/cpp/rta sweep "$c" 6 127 200 "$seed" "data/rerun/random_$tag.csv" random 0; seed=$((seed + 1))
done
.venv/bin/python scripts/parse_legacy.py "$CODES_DIR/artifacts/results-1999" data/legacy

.venv/bin/python scripts/verify_certificates.py --full
.venv/bin/python -m pytest -q -m ""
.venv/bin/python scripts/make_figures.py
"$CODES_DIR/scripts/build_monograph.sh"
