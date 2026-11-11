#!/usr/bin/env bash
# Exact mean costs of the elimination key (articles/elimination-keys).
# Writes data/elimination-keys/exact_means.csv. A configuration that exceeds the
# per-run limit is left out, so the file lists exactly what was finished.
set -u
cd "$(dirname "$0")/.."
out=data/elimination-keys/exact_means.csv
limit=${1:-240}
echo "model,m,n,t,errors,rule,num,den,mean,tables" > "$out"
run() {  # model m n t optimal
  local tmp
  tmp=$(mktemp)
  if timeout "$limit" build/cpp/elimination_exact "$@" > "$tmp"; then cat "$tmp" >> "$out"
  else echo "skipped $* (status $?)" >&2; fi
  rm -f "$tmp"
}
for t in 0 1 2 3; do
  for n in $(seq 2 12); do run all 2 "$n" "$t" 1; done
  for n in $(seq 2 10); do run all 3 "$n" "$t" $((n <= 8)); done
  for n in $(seq 2 9);  do run all 4 "$n" "$t" $((n <= 7)); done
  for n in $(seq 3 7);  do run all 5 "$n" "$t" $((n <= 5)); done
  for n in $(seq 3 6);  do run all 6 "$n" "$t" $((n <= 5)); done
  for n in $(seq 3 5);  do run all 8 "$n" "$t" $((n <= 4)); done
done
for t in 0 1; do run all 16 3 "$t" 0; run all 16 4 "$t" 0; done
for n in 5 6 7 8 9 10; do run dist 4 "$n" 1 $((n <= 7)); done
for n in 6 7 8; do run dist 8 "$n" 1 0; done
run dist 16 7 1 0; run dist 16 8 1 0
for n in 8 9 10 11 12 13; do run dist 4 "$n" 2 0; done
for n in 10 11 12; do run dist 8 "$n" 2 0; done
for n in 11 12 13 14; do run dist 4 "$n" 3 0; done
for n in 13 14; do run dist 8 "$n" 3 0; done
