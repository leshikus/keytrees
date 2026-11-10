# keytrees

The code for the `codes` repository (the frozen monograph and the articles in
`codes/articles/`). Read `../codes/CLAUDE.md` for the text rules; they apply to
everything here that states a number.

- Heavy computation in C++23 (`src/cpp/`, built with clang; CMake makes one
  executable per `.cpp`), anything else in Python (`src/python/keytrees/`, Sage-style
  doctests).
- Every number in a text needs a test in `tests/` or a file in `data/`. Tests for
  an article are named `tests/test_<article-folder>.py`, data go in
  `data/<article-folder>/`.
- Code is MIT-licensed and the repo is public. Keep personal data out, and keep
  out code for the aggregation model (A. P. Ryzhov's unpublished work): it lives
  in the `codes` repository.
- Commit and push when work is verified.
