# keytrees

Code behind *Cost Estimates for Decision Trees* (the monograph) and the articles
that follow it: trees for identification, search and aggregation. The texts are
in the separate `codes` repository; check it out next to this one, or set
`CODES_DIR`.

- `src/cpp/`: heavy computation, C++23 built with LLVM. One executable per file.
- `src/python/keytrees/`: everything else, Python with Sage-style doctests.
- `tests/`: checks of every number stated in the texts.
- `data/`: outputs behind every table and figure.
- `scripts/reproduce.sh`: rebuilds the engines and data, runs the tests,
  regenerates the figures and builds the monograph.

## Build and test (Linux)

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
cmake -S src/cpp -B build/cpp -G Ninja -DCMAKE_CXX_COMPILER=clang++
cmake --build build/cpp
```

Packages on Ubuntu: `clang llvm lld cmake ninja-build python3-venv libgmp-dev
libboost-dev`; for the texts, `texlive-full latexmk biber`.

## Licence

GPL-2.0-or-later; see `LICENSE`.
