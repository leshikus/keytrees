// gradient_bound: the gradient (greedy) key on random tables, for
// articles/gradient-bound.
//
// A table is an n x m binary matrix with m distinct columns, uniform over all
// such matrices. The gradient key G tests, at each fork, the first row of
// largest quality min(k, s - k), where k of the s columns at the fork have a
// one in that row. Cost is the mean leaf depth.
//
// The recursion "chain" belongs to a construction that uses L consecutive groups
// of b rows, one group per level; see the article.
//
// Every result here is exact; nothing is sampled.
//
// Subcommands:
//   exact <n> <m>              brute force over all m-subsets of {0,1}^n: t_{n,m}(G);
//                              prints "n,m,numerator,denominator"
//   formula <n> <m> <threads>  the path formula of the article: the number
//                              N = sum over tables of the depth of column 1, as
//                              residues modulo 62-bit primes (combined by
//                              scripts/gradient_bound_data.py);
//                              prints "n,m,paths,tuples,p:r,p:r,..."
//   chain <b> <Smax> <out.csv> [charge]
//                              the recursion r_b(s) of the article, in double
//                              precision; with "charge" a group without a
//                              splitting row costs 1
//
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <fstream>
#include <functional>
#include <numeric>
#include <print>
#include <string>
#include <thread>
#include <vector>

namespace {

using u64 = std::uint64_t;
using u128 = unsigned __int128;

std::string to_string(u128 v) {
  if (v == 0) return "0";
  std::string s;
  while (v) {
    s.push_back(static_cast<char>('0' + static_cast<int>(v % 10)));
    v /= 10;
  }
  std::reverse(s.begin(), s.end());
  return s;
}

// ---------------------------------------------------------------- exact

struct Exact {
  int n = 0, m = 0;
  std::vector<u64> rowmask;  // rowmask[r]: bit j set when column j has a 1 in row r

  int best_row(u64 S, int lo, int hi) const {
    const int s = std::popcount(S);
    int best = -1, bestq = 0;
    for (int r = lo; r < hi; ++r) {
      const int k = std::popcount(S & rowmask[static_cast<std::size_t>(r)]);
      const int q = std::min(k, s - k);
      if (q > bestq) {
        bestq = q;
        best = r;
      }
    }
    return best;
  }

  // Sum of leaf depths on the column subset S (a bitmask over the m columns).
  u64 depth_sum(u64 S) const {
    const int s = std::popcount(S);
    if (s <= 1) return 0;
    const int best = best_row(S, 0, n);  // distinct columns guarantee a splitting row
    const u64 one = S & rowmask[static_cast<std::size_t>(best)];
    return static_cast<u64>(s) + depth_sum(one) + depth_sum(S & ~one);
  }

  u128 total = 0;
  u128 count = 0;
  std::vector<int> cols;

  void enumerate(int next, int chosen) {
    if (chosen == m) {
      total += depth_sum(m == 64 ? ~u64{0} : (u64{1} << m) - 1);
      ++count;
      return;
    }
    const int words = 1 << n;
    for (int c = next; c <= words - (m - chosen); ++c) {
      for (int r = 0; r < n; ++r)
        if ((c >> r) & 1) rowmask[static_cast<std::size_t>(r)] |= u64{1} << chosen;
      enumerate(c + 1, chosen + 1);
      for (int r = 0; r < n; ++r) rowmask[static_cast<std::size_t>(r)] &= ~(u64{1} << chosen);
    }
  }
};

int cmd_exact(int n, int m) {
  if (n < 1 || n > 20 || m < 1 || m > 63 || m > (1 << n)) {
    std::println(stderr, "exact: need 1 <= n <= 20, 1 <= m <= min(63, 2^n)");
    return 2;
  }
  Exact e;
  e.n = n;
  e.m = m;
  e.rowmask.assign(static_cast<std::size_t>(n), 0);
  e.enumerate(0, 0);
  // t = total / (m * count)
  std::println("{},{},{},{}", n, m, to_string(e.total), to_string(e.count * static_cast<u128>(m)));
  return 0;
}

// ---------------------------------------------------------------- path formula
//
// N = sum_{tables} depth(column 1) = sum over chains S_0 > S_1 > ... > S_d of the
// number of tables in which column 1 follows that chain. Möbius inversion over
// set partitions removes the distinct-columns condition, after which the rows are
// independent; a dynamic programme over the rows in order handles "first row".

struct Mont {
  u64 p = 0, pinv = 0, r2 = 0;
  explicit Mont(u64 prime) : p(prime) {
    u64 x = 1;
    for (int i = 0; i < 6; ++i) x *= 2 - p * x;
    pinv = x;
    const u128 r = (~u128{0}) % p + 1;  // 2^128 mod p, possibly equal to p
    r2 = static_cast<u64>(r % p);
  }
  u64 mul(u64 a, u64 b) const {
    const u128 t = static_cast<u128>(a) * b;
    const u64 q = static_cast<u64>(t) * pinv;
    const u64 u = static_cast<u64>((static_cast<u128>(q) * p) >> 64);
    const u64 hi = static_cast<u64>(t >> 64);
    return hi >= u ? hi - u : hi - u + p;
  }
  u64 add(u64 a, u64 b) const {
    const u64 s = a + b;
    return s >= p ? s - p : s;
  }
  u64 in(u64 a) const { return mul(a % p, r2); }
  u64 out(u64 a) const { return mul(a, 1); }
};

bool is_prime(u64 n) {
  if (n < 2 || n % 2 == 0) return n == 2;
  u64 d = n - 1;
  int r = 0;
  while (d % 2 == 0) d /= 2, ++r;
  auto mulmod = [n](u64 a, u64 b) { return static_cast<u64>(static_cast<u128>(a) * b % n); };
  for (u64 a : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL, 37ULL}) {
    if (a % n == 0) continue;
    u64 x = 1, b = a, e = d;
    for (; e; e >>= 1, b = mulmod(b, b))
      if (e & 1) x = mulmod(x, b);
    if (x == 1 || x == n - 1) continue;
    bool composite = true;
    for (int i = 1; i < r && composite; ++i) {
      x = mulmod(x, x);
      if (x == n - 1) composite = false;
    }
    if (composite) return false;
  }
  return true;
}

struct Part {
  std::vector<u64> poly;     // prod over blocks (1 + x^a), degree = cell size
  std::vector<u64> coef;     // Möbius weight times number of set partitions, per prime (Montgomery)
  std::size_t blocks = 0;
};

struct Formula {
  int n = 0, m = 0;
  std::vector<Mont> mod;
  std::vector<std::vector<Part>> parts;  // parts[a]: integer partitions of a
  std::vector<std::vector<u64>> binom;   // exact binomials, m <= 62

  void build_parts() {
    parts.assign(static_cast<std::size_t>(m) + 1, {});
    for (int a = 1; a <= m; ++a) {
      std::vector<int> cur;
      std::function<void(int, int)> rec = [&](int left, int top) {
        if (left == 0) {
          add_part(a, cur);
          return;
        }
        for (int v = std::min(left, top); v >= 1; --v) {
          cur.push_back(v);
          rec(left - v, v);
          cur.pop_back();
        }
      };
      rec(a, a);
    }
  }

  void add_part(int a, const std::vector<int>& lam) {
    Part pt;
    pt.blocks = lam.size();
    pt.poly.assign(static_cast<std::size_t>(a) + 1, 0);
    pt.poly[0] = 1;
    for (int v : lam)
      for (int k = a; k >= v; --k) pt.poly[static_cast<std::size_t>(k)] += pt.poly[static_cast<std::size_t>(k - v)];
    // |coef| = a! / (prod lambda_i * prod mult!), computed exactly; a <= 33 fits 128 bits.
    u128 c = 1;
    for (int i = 2; i <= a; ++i) c *= static_cast<u128>(i);
    for (int v : lam) c /= static_cast<u128>(v);
    for (std::size_t i = 0; i < lam.size();) {
      std::size_t j = i;
      while (j < lam.size() && lam[j] == lam[i]) ++j;
      for (std::size_t f = 2; f <= j - i; ++f) c /= static_cast<u128>(f);
      i = j;
    }
    const bool negative = (a - static_cast<int>(lam.size())) % 2 != 0;
    for (const Mont& md : mod) {
      u64 v = md.in(static_cast<u64>(c % md.p));
      if (negative && v) v = md.p - v;
      pt.coef.push_back(v);
    }
    parts[static_cast<std::size_t>(a)].push_back(std::move(pt));
  }

  // State reused for one chain.
  struct Work {
    int d = 0;
    std::vector<int> s, a, q;
    // lists[level][set]: polynomials; set 0 is the free rows, set j+1 the row chosen at level j.
    std::vector<std::vector<std::vector<u64>>> wl;                // wl[depth]: polys for free rows
    std::vector<std::vector<std::vector<std::vector<u64>>>> cl;   // cl[depth][j]: polys for chosen row j
    std::vector<std::vector<u64>> f;                              // per prime dp
    std::vector<u64> wm;
    std::vector<std::vector<u64>> cm;
    std::vector<u64> acc;
    u64 tuples = 0;
  };

  // new = trunc(v * g) to quality <= qmax on size s (qmax may be 0 or negative for "none").
  static void step(const std::vector<u64>& v, const std::vector<u64>& g, int s, int qmax, std::vector<u64>& out) {
    out.assign(static_cast<std::size_t>(s) + 1, 0);
    if (qmax < 0) return;
    const int vs = static_cast<int>(v.size()) - 1;
    const int gs = static_cast<int>(g.size()) - 1;
    auto fill = [&](int lo, int hi) {
      for (int k = lo; k <= hi; ++k) {
        u64 sum = 0;
        for (int i = std::max(0, k - gs); i <= std::min(vs, k); ++i)
          sum += v[static_cast<std::size_t>(i)] * g[static_cast<std::size_t>(k - i)];
        out[static_cast<std::size_t>(k)] = sum;
      }
    };
    if (2 * qmax >= s) {
      fill(0, s);
    } else {
      fill(0, qmax);
      fill(s - qmax, s);
    }
  }

  void descend(Work& w, int level, const std::vector<u64>& coef) {
    // Levels are processed from d-1 down to 0; "level" is the next one to process.
    if (level < 0) {
      finish(w, coef);
      return;
    }
    const int depth = w.d - 1 - level;  // lists at wl[depth] are ready
    for (const Part& pt : parts[static_cast<std::size_t>(w.a[static_cast<std::size_t>(level)])]) {
      auto& src = w.wl[static_cast<std::size_t>(depth)];
      auto& dst = w.wl[static_cast<std::size_t>(depth) + 1];
      const int s = w.s[static_cast<std::size_t>(level)];
      const int q = w.q[static_cast<std::size_t>(level)];
      dst.resize(src.size() * 2);
      for (std::size_t i = 0; i < src.size(); ++i) {
        step(src[i], pt.poly, s, q - 1, dst[2 * i]);      // bit 0: row before r_level, strict
        step(src[i], pt.poly, s, q, dst[2 * i + 1]);      // bit 1: row after r_level
      }
      for (int j = level + 1; j < w.d; ++j) {
        auto& cs = w.cl[static_cast<std::size_t>(depth)][static_cast<std::size_t>(j)];
        auto& cd = w.cl[static_cast<std::size_t>(depth) + 1][static_cast<std::size_t>(j)];
        cd.resize(cs.size() * 2);
        for (std::size_t i = 0; i < cs.size(); ++i) {
          step(cs[i], pt.poly, s, q - 1, cd[2 * i]);
          step(cs[i], pt.poly, s, q, cd[2 * i + 1]);
        }
      }
      // The row chosen at this level: pattern fixed on S_level.
      auto& own = w.cl[static_cast<std::size_t>(depth) + 1][static_cast<std::size_t>(level)];
      own.assign(1, std::vector<u64>(static_cast<std::size_t>(s) + 1, 0));
      own[0][static_cast<std::size_t>(w.s[static_cast<std::size_t>(level) + 1])] += 1;
      own[0][static_cast<std::size_t>(w.a[static_cast<std::size_t>(level)])] += 1;
      std::vector<u64> c2(coef.size());
      for (std::size_t i = 0; i < coef.size(); ++i) c2[i] = mod[i].mul(coef[i], pt.coef[i]);
      descend(w, level - 1, c2);
    }
  }

  void finish(Work& w, const std::vector<u64>& coef) {
    ++w.tuples;
    const int d = w.d;
    const std::size_t full = (std::size_t{1} << d);
    auto& W = w.wl[static_cast<std::size_t>(d)];
    auto& C = w.cl[static_cast<std::size_t>(d)];
    // Sums of coefficients; list index bit (d-1-level) is level "level" in processing order.
    std::vector<u64> wsum(full);
    bool any = false;
    for (std::size_t i = 0; i < full; ++i) {
      u64 v = 0;
      for (u64 c : W[i]) v += c;
      // index i has bit (level) at position level, as level 0 is appended last.
      wsum[i] = v;
      any = any || v;
    }
    if (d > 0 && !any) {
      // Rows are all constrained away; only chosen rows could remain, and n > d needs free rows.
      if (n > d) return;
    }
    std::vector<std::vector<u64>> csum(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) {
      const auto& list = C[static_cast<std::size_t>(j)];
      csum[static_cast<std::size_t>(j)].resize(list.size());
      for (std::size_t i = 0; i < list.size(); ++i) {
        u64 v = 0;
        for (u64 c : list[i]) v += c;
        csum[static_cast<std::size_t>(j)][i] = v;
      }
    }
    for (std::size_t pi = 0; pi < mod.size(); ++pi) {
      const Mont& md = mod[pi];
      auto& f = w.f[pi];
      std::fill(f.begin(), f.begin() + static_cast<std::ptrdiff_t>(full), 0);
      f[0] = md.in(1);
      std::vector<u64>& wm = w.wm;
      for (std::size_t i = 0; i < full; ++i) wm[i] = md.in(wsum[i]);
      for (int j = 0; j < d; ++j)
        for (std::size_t i = 0; i < csum[static_cast<std::size_t>(j)].size(); ++i)
          w.cm[static_cast<std::size_t>(j)][i] = md.in(csum[static_cast<std::size_t>(j)][i]);
      for (int row = 0; row < n; ++row) {
        for (std::size_t mask = full; mask-- > 0;) {
          const int pc = std::popcount(mask);
          if (pc > row + 1 || pc < d - (n - row - 1)) {
            f[mask] = 0;
            continue;
          }
          u64 v = md.mul(f[mask], wm[mask]);
          for (std::size_t rest = mask; rest; rest &= rest - 1) {
            const int j = std::countr_zero(rest);
            const std::size_t before = mask ^ (std::size_t{1} << j);
            if (f[before]) {
              const u64 c = w.cm[static_cast<std::size_t>(j)][before & ((std::size_t{1} << j) - 1)];
              v = md.add(v, md.mul(f[before], c));
            }
          }
          f[mask] = v;
        }
      }
      w.acc[pi] = md.add(w.acc[pi], md.mul(f[full - 1], coef[pi]));
    }
  }

  // Adds the chain's contribution to total.
  void chain(const std::vector<int>& s, std::vector<u64>& total, u64& tuples) {
    Work w;
    w.d = static_cast<int>(s.size()) - 1;
    const int d = w.d;
    w.s = s;
    w.a.resize(static_cast<std::size_t>(d));
    w.q.resize(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) {
      w.a[static_cast<std::size_t>(j)] = s[static_cast<std::size_t>(j)] - s[static_cast<std::size_t>(j) + 1];
      w.q[static_cast<std::size_t>(j)] = std::min(w.a[static_cast<std::size_t>(j)], s[static_cast<std::size_t>(j) + 1]);
    }
    w.wl.resize(static_cast<std::size_t>(d) + 1);
    w.cl.assign(static_cast<std::size_t>(d) + 1, std::vector<std::vector<std::vector<u64>>>(static_cast<std::size_t>(d)));
    const std::size_t full = std::size_t{1} << d;
    w.f.assign(mod.size(), std::vector<u64>(full, 0));
    w.wm.assign(full, 0);
    w.cm.resize(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) w.cm[static_cast<std::size_t>(j)].assign(std::size_t{1} << j, 0);
    w.acc.assign(mod.size(), 0);
    std::vector<u64> one(mod.size());
    for (std::size_t i = 0; i < mod.size(); ++i) one[i] = mod[i].in(1);
    for (const Part& inner : parts[static_cast<std::size_t>(s.back())]) {
      w.wl[0].assign(1, inner.poly);
      std::vector<u64> c(mod.size());
      for (std::size_t i = 0; i < mod.size(); ++i) c[i] = mod[i].mul(one[i], inner.coef[i]);
      descend(w, d - 1, c);
    }
    for (std::size_t i = 0; i < mod.size(); ++i) {
      u64 mult = mod[i].in(1);
      for (int j = 0; j < d; ++j)
        mult = mod[i].mul(mult, mod[i].in(binom[static_cast<std::size_t>(s[static_cast<std::size_t>(j)] - 1)]
                                              [static_cast<std::size_t>(s[static_cast<std::size_t>(j) + 1] - 1)]));
      total[i] = mod[i].add(total[i], mod[i].mul(mult, w.acc[i]));
    }
    tuples += w.tuples;
  }

  // A chain with no table in A_{n,m} is skipped; see the article for why the sum there is 0.
  bool feasible_prefix(int j, int sj, int next) const {
    const int a = sj - next, q = std::min(a, next);
    if (n - j <= 0) return false;
    if (static_cast<long long>(q) * (n - j) < sj - 1) return false;
    const int bits = n - j - 1;
    if (bits < 40 && (a > (1LL << bits) || next > (1LL << bits))) return false;
    return true;
  }

  void enumerate(std::vector<int>& s, std::vector<std::vector<int>>& out) const {
    out.push_back(s);
    const int j = static_cast<int>(s.size()) - 1;
    const int sj = s.back();
    for (int next = 2; next < sj; ++next) {
      if (!feasible_prefix(j, sj, next)) continue;
      s.push_back(next);
      enumerate(s, out);
      s.pop_back();
    }
  }
};

int cmd_formula(int n, int m, int threads) {
  if (n < 1 || m < 2 || m > 33 || (n < 40 && m > (1LL << n))) {
    std::println(stderr, "formula: need 2 <= m <= min(33, 2^n)");
    return 2;
  }
  const double bits = static_cast<double>(m) * n + std::log2(static_cast<double>(m)) + 2;
  const int nprimes = static_cast<int>(std::ceil(bits / 61.0)) + 1;  // one spare, as a check
  Formula fm;
  fm.n = n;
  fm.m = m;
  for (u64 c = (u64{1} << 62) - 1; static_cast<int>(fm.mod.size()) < nprimes; c -= 2)
    if (is_prime(c)) fm.mod.emplace_back(c);
  fm.binom.assign(static_cast<std::size_t>(m) + 1, std::vector<u64>(static_cast<std::size_t>(m) + 1, 0));
  for (int i = 0; i <= m; ++i) {
    fm.binom[static_cast<std::size_t>(i)][0] = 1;
    for (int j = 1; j <= i; ++j)
      fm.binom[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
          fm.binom[static_cast<std::size_t>(i) - 1][static_cast<std::size_t>(j) - 1] +
          (j < i ? fm.binom[static_cast<std::size_t>(i) - 1][static_cast<std::size_t>(j)] : 0);
  }
  fm.build_parts();
  std::vector<std::vector<int>> chains;
  std::vector<int> s{m};
  fm.enumerate(s, chains);
  // Heaviest chains first, for load balance.
  auto weight = [&](const std::vector<int>& c) {
    double w = std::ldexp(1.0, static_cast<int>(c.size()));
    for (std::size_t j = 0; j + 1 < c.size(); ++j) w *= static_cast<double>(fm.parts[static_cast<std::size_t>(c[j] - c[j + 1])].size());
    return w * static_cast<double>(fm.parts[static_cast<std::size_t>(c.back())].size());
  };
  std::sort(chains.begin(), chains.end(), [&](const auto& x, const auto& y) { return weight(x) > weight(y); });
  std::atomic<std::size_t> next{0};
  std::vector<std::vector<u64>> totals(static_cast<std::size_t>(threads), std::vector<u64>(fm.mod.size(), 0));
  std::vector<u64> tuples(static_cast<std::size_t>(threads), 0);
  {
    std::vector<std::jthread> pool;
    for (int t = 0; t < threads; ++t)
      pool.emplace_back([&, t] {
        for (std::size_t i = next++; i < chains.size(); i = next++)
          fm.chain(chains[i], totals[static_cast<std::size_t>(t)], tuples[static_cast<std::size_t>(t)]);
      });
  }
  std::string line = std::format("{},{},{},{}", n, m, chains.size(), std::accumulate(tuples.begin(), tuples.end(), u64{0}));
  for (std::size_t i = 0; i < fm.mod.size(); ++i) {
    u64 v = 0;
    for (int t = 0; t < threads; ++t) v = fm.mod[i].add(v, totals[static_cast<std::size_t>(t)][i]);
    line += std::format(",{}:{}", fm.mod[i].p, fm.mod[i].out(v));
  }
  std::println("{}", line);
  return 0;
}

// ---------------------------------------------------------------- chain

double one_minus_h(double x) {
  if (x <= 0.0 || x >= 1.0) return 1.0;
  return 1.0 + x * std::log2(x) + (1.0 - x) * std::log2(1.0 - x);
}

int cmd_chain(int b, int smax, const char* path, bool charge) {
  std::vector<double> r(static_cast<std::size_t>(smax) + 1, 0.0);
  std::ofstream out(path);
  out << "s,r_b\n1,0\n";
  for (int s = 2; s <= smax; ++s) {
    // Binomial(s, 1/2) pmf in log space.
    std::vector<double> pm(static_cast<std::size_t>(s) + 1);
    for (int k = 0; k <= s; ++k)
      pm[static_cast<std::size_t>(k)] =
          std::exp(std::lgamma(s + 1.0) - std::lgamma(k + 1.0) - std::lgamma(s - k + 1.0) - s * std::log(2.0));
    const int half = s / 2;
    // tail[q] = P(one row has quality >= q) = P(q <= B <= s - q)
    std::vector<double> tail(static_cast<std::size_t>(half) + 2, 0.0);
    double acc = 0.0;
    for (int q = half; q >= 0; --q) {
      acc += pm[static_cast<std::size_t>(q)];
      if (s - q != q) acc += pm[static_cast<std::size_t>(s - q)];
      tail[static_cast<std::size_t>(q)] = acc;
    }
    auto best_tail = [&](int q) {  // P(best of b has quality >= q)
      if (q > half) return 0.0;
      return 1.0 - std::pow(1.0 - std::min(1.0, tail[static_cast<std::size_t>(q)]), b);
    };
    const double p0 = 1.0 - best_tail(1);
    // Skipping (G_b): a block without a splitting row is passed over at no cost.
    // Charging: such a test costs 1 and leaves the set as it is.
    double sum = charge ? p0 : 0.0;
    for (int q = 1; q <= half; ++q) {
      const double pq = best_tail(q) - best_tail(q + 1);
      const double x = static_cast<double>(q) / s;
      sum += pq * (one_minus_h(x) + x * r[static_cast<std::size_t>(q)] + (1.0 - x) * r[static_cast<std::size_t>(s - q)]);
    }
    r[static_cast<std::size_t>(s)] = sum / (1.0 - p0);
    std::println(out, "{},{:.10f}", s, r[static_cast<std::size_t>(s)]);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "exact" && argc == 4) return cmd_exact(std::atoi(argv[2]), std::atoi(argv[3]));
  if (cmd == "formula" && argc == 5) return cmd_formula(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]));
  if (cmd == "chain" && (argc == 5 || (argc == 6 && std::string(argv[5]) == "charge")))
    return cmd_chain(std::atoi(argv[2]), std::atoi(argv[3]), argv[4], argc == 6);
  std::println(stderr, "usage: gradient_bound exact n m | formula n m threads | chain b Smax out.csv [charge]");
  return 2;
}
