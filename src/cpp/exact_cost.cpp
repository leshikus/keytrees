// exact_cost: average cost of search trees on random sets of binary words.
//
// Words: a uniformly random m-subset of {0,1}^n. All costs are the average
// number of tested positions needed to identify a word of the subset.
//
// Variants are indexed by a threshold s. A node holding more than s words tests
// a uniformly random unused position, and that test is charged even when it
// does not split the words. A node holding at most s words tests only positions
// that do split them. So:
//   s = 1    plain trie: the literal "random position" algorithm R
//   s = 2    t(n,2) = 1 enforced
//   s = 3    t(n,2) = 1, t(n,3) = 5/3 enforced: the recurrence behind the 1999
//            table and the bound log2 m + 29/28 - log2(2m)/m
//   s = inf  no useless test is ever charged: a PATRICIA trie
//
// Subcommands:
//   recurrence <Nmax> <Mmax> <out.csv>   finite-n recurrences for all variants
//   limit <Mmax> <out.csv>               n -> infinity limit for all variants
//   brute <out.csv>                      exhaustive check for small n and m
//   identity <Nmax>                      checks the combinatorial identity
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numbers>
#include <print>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Real = long double;
constexpr int kInf = std::numeric_limits<int>::max();
constexpr int kThresholds[] = {1, 2, 3, kInf};

std::string variant_name(int s) { return s == kInf ? "patricia" : "s" + std::to_string(s); }

Real bound_1999(int m) {
  const Real x = static_cast<Real>(m);
  return std::log2(x) + 29.0L / 28.0L - std::log2(2.0L * x) / x;
}

// c[k] = C(H,k) C(H,m-k) / C(2H,m): probability of k ones in a tested position
// when H = 2^(n-1). Computed by a ratio recurrence, which stays accurate for
// large H, where differences of log-gamma values would lose all their digits.
void split_weights(Real H, int m, std::vector<Real>& c) {
  c.assign(static_cast<std::size_t>(m) + 1, 0.0L);
  const int k0 = (static_cast<Real>(m) > H) ? static_cast<int>(static_cast<Real>(m) - H) : 0;
  const int k1 = (static_cast<Real>(m) > H) ? static_cast<int>(H) : m;
  // Start at k0: when k0 = 0 the value is prod (H-i)/(2H-i). Otherwise build it
  // as C(H,k0) C(H,m-k0) / C(2H,m) with m-k0 = H, so C(H,m-k0) = 1.
  Real v = 1.0L;
  if (k0 == 0) {
    for (int i = 0; i < m; ++i) v *= (H - i) / (2.0L * H - i);
  } else {
    // C(H,k0) / C(2H,m), pairing factors to avoid overflow.
    for (int i = 0; i < m; ++i) v *= (i < k0 ? (H - i) / (k0 - i) : 1.0L) * (m - i) / (2.0L * H - i);
  }
  c[static_cast<std::size_t>(k0)] = v;
  for (int k = k0; k < k1; ++k) {
    v *= (static_cast<Real>(m - k) / (k + 1)) * ((H - k) / (H - m + k + 1));
    c[static_cast<std::size_t>(k) + 1] = v;
  }
}

// One step of the recurrence. prev[j] = t(n-1, j); w[k] = split weights.
Real step(int m, int s, const std::vector<Real>& w, const std::vector<Real>& prev) {
  if (m <= 1) return 0.0L;
  const Real dm = static_cast<Real>(m);
  Real inner = 0.0L;
  for (int k = 1; k < m; ++k) {
    const Real term = (static_cast<Real>(k) / dm) * prev[static_cast<std::size_t>(k)] +
                      (static_cast<Real>(m - k) / dm) * prev[static_cast<std::size_t>(m - k)];
    inner += w[static_cast<std::size_t>(k)] * (1.0L + term);
  }
  const Real useless = w[0] + w[static_cast<std::size_t>(m)];
  // The useless split passes all words on to n-1 positions; it costs 1 if charged.
  if (m <= s) return inner + useless * prev[static_cast<std::size_t>(m)];
  return inner + useless * (1.0L + prev[static_cast<std::size_t>(m)]);
}

int cmd_recurrence(int nmax, int mmax, const std::string& out_path) {
  std::ofstream out(out_path);
  out << "variant,n,m,t,redundancy,bound_1999\n";
  out.precision(18);
  for (int s : kThresholds) {
    std::vector<Real> prev(static_cast<std::size_t>(mmax) + 1, 0.0L), cur = prev, w;
    Real worst_excess = -1e9L, worst_drop = 0.0L;
    int worst_n = 0, worst_m = 0;
    for (int n = 1; n <= nmax; ++n) {
      const Real H = std::ldexp(1.0L, n - 1);
      const int top = (n < 31) ? std::min(mmax, 1 << n) : mmax;
      for (int m = 0; m <= mmax; ++m) cur[static_cast<std::size_t>(m)] = 0.0L;
      for (int m = 2; m <= top; ++m) {
        split_weights(H, m, w);
        const Real t = step(m, s, w, prev);
        cur[static_cast<std::size_t>(m)] = t;
        // Is t(n,m) nondecreasing in n? Only meaningful once t(n-1,m) exists.
        if (n > 1 && m <= (n - 1 < 31 ? (1 << (n - 1)) : mmax))
          worst_drop = std::max(worst_drop, prev[static_cast<std::size_t>(m)] - t);
        const Real excess = t - bound_1999(m);
        if (excess > worst_excess) { worst_excess = excess; worst_n = n; worst_m = m; }
        const bool keep = m <= 64 || std::has_single_bit(static_cast<unsigned>(m)) || m % 64 == 0;
        if (keep && (n <= 16 || n % 4 == 0 || n == nmax)) {
          out << variant_name(s) << ',' << n << ',' << m << ',' << t << ','
              << t - std::log2(static_cast<Real>(m)) << ',' << bound_1999(m) << '\n';
        }
      }
      prev.swap(cur);
    }
    std::println("{:9}: max over n<={} m<={} of t - bound = {:+.6f} at (n,m)=({},{}); largest decrease of t in n = {:.3e}",
                 variant_name(s), nmax, mmax, static_cast<double>(worst_excess), worst_n, worst_m,
                 static_cast<double>(worst_drop));
  }
  return 0;
}

int cmd_limit(int mmax, const std::string& out_path) {
  std::ofstream out(out_path);
  out << "variant,m,t,redundancy,bound_1999\n";
  out.precision(18);
  for (int s : kThresholds) {
    std::vector<Real> t(static_cast<std::size_t>(mmax) + 1, 0.0L);
    Real lo = 1e9L, hi = -1e9L, sum = 0.0L, excess_max = -1e9L;
    int count = 0, excess_m = 0;
    for (int m = 2; m <= mmax; ++m) {
      // Binomial weights in the limit; C(m,k)/2^m by the ratio recurrence.
      Real w = std::ldexp(1.0L, -m), acc = 0.0L;
      const Real dm = static_cast<Real>(m);
      for (int k = 1; k < m; ++k) {
        w *= static_cast<Real>(m - k + 1) / k;
        acc += w * (1.0L + (static_cast<Real>(k) / dm) * t[static_cast<std::size_t>(k)] +
                    (static_cast<Real>(m - k) / dm) * t[static_cast<std::size_t>(m - k)]);
      }
      const Real useless = std::ldexp(1.0L, 1 - m);
      // Solve t = acc + useless * (charge + t) for t.
      const Real charge = (m <= s) ? 0.0L : 1.0L;
      const Real v = (acc + useless * charge) / (1.0L - useless);
      t[static_cast<std::size_t>(m)] = v;
      const Real r = v - std::log2(dm);
      const Real excess = v - bound_1999(m);
      if (excess > excess_max) { excess_max = excess; excess_m = m; }
      if (m >= mmax / 2) { lo = std::min(lo, r); hi = std::max(hi, r); sum += r; ++count; }
      const bool keep = m <= 256 || m % 16 == 0;
      if (keep) out << variant_name(s) << ',' << m << ',' << v << ',' << r << ',' << bound_1999(m) << '\n';
    }
    std::println("{:9}: redundancy on [{},{}]: mean {:.8f}, range [{:.8f},{:.8f}]; max(t-bound) {:+.6f} at m={}",
                 variant_name(s), mmax / 2, mmax, static_cast<double>(sum / count), static_cast<double>(lo),
                 static_cast<double>(hi), static_cast<double>(excess_max), excess_m);
  }
  const Real g = std::numbers::egamma_v<Real> / std::numbers::ln2_v<Real>;
  const Real l2 = std::numbers::ln2_v<Real>;
  std::println("closed forms: trie {:.10f}  s2 {:.10f}  s3 {:.10f}  patricia {:.10f}",
               static_cast<double>(g + 0.5L), static_cast<double>(g + 0.5L - 1.0L / (2 * l2)),
               static_cast<double>(g + 0.5L - 1.0L / (2 * l2) - 1.0L / (8 * l2)), static_cast<double>(g - 0.5L));
  return 0;
}

// ---------- exhaustive enumeration ----------

struct Key {
  std::uint32_t set;
  std::uint32_t mask;
  bool operator==(const Key&) const = default;
};
struct KeyHash {
  std::size_t operator()(const Key& k) const noexcept {
    return std::hash<std::uint64_t>{}((static_cast<std::uint64_t>(k.set) << 32) ^ k.mask);
  }
};

class Brute {
 public:
  explicit Brute(int n) : n_(n) {
    const int words = 1 << n;
    for (int p = 0; p < n; ++p) {
      std::uint32_t mk = 0;
      for (int w = 0; w < words; ++w)
        if ((w >> p) & 1) mk |= (std::uint32_t{1} << w);
      ones_[p] = mk;
    }
  }

  // Expected sum of depths of the words of `set` for threshold s.
  double random_tree(std::uint32_t set, std::uint32_t mask, int s) {
    const int m = std::popcount(set);
    if (m <= 1) return 0.0;
    auto& memo = memo_[index(s)];
    const Key key{set, mask};
    if (auto it = memo.find(key); it != memo.end()) return it->second;
    double sum = 0.0;
    int choices = 0;
    const bool only_informative = m <= s;
    for (int p = 0; p < n_; ++p) {
      if (!((mask >> p) & 1)) continue;
      const std::uint32_t a = set & ones_[p], b = set & ~ones_[p];
      const bool informative = a && b;
      if (only_informative && !informative) continue;
      ++choices;
      sum += m + random_tree(a, mask & ~(1u << p), s) + random_tree(b, mask & ~(1u << p), s);
    }
    const double v = sum / choices;
    memo.emplace(key, v);
    return v;
  }

  // Minimal sum of depths over all trees.
  int optimal(std::uint32_t set) {
    const int m = std::popcount(set);
    if (m <= 1) return 0;
    if (auto it = opt_.find(set); it != opt_.end()) return it->second;
    int best = std::numeric_limits<int>::max();
    for (int p = 0; p < n_; ++p) {
      const std::uint32_t a = set & ones_[p], b = set & ~ones_[p];
      if (a && b) best = std::min(best, m + optimal(a) + optimal(b));
    }
    opt_.emplace(set, best);
    return best;
  }

  // Gradient algorithm: the first position maximising min(k, m-k).
  int gradient(std::uint32_t set) {
    const int m = std::popcount(set);
    if (m <= 1) return 0;
    int quality = -1, chosen = 0;
    for (int p = 0; p < n_; ++p) {
      const int k = std::popcount(set & ones_[p]);
      const int q = std::min(k, m - k);
      if (q > quality) { quality = q; chosen = p; }
    }
    return m + gradient(set & ones_[chosen]) + gradient(set & ~ones_[chosen]);
  }

  // Largest min(k, m-k) over all positions.
  int matrix_quality(std::uint32_t set) const {
    const int m = std::popcount(set);
    int q = 0;
    for (int p = 0; p < n_; ++p) {
      const int k = std::popcount(set & ones_[p]);
      q = std::max(q, std::min(k, m - k));
    }
    return q;
  }

 private:
  static std::size_t index(int s) { return s == kInf ? 3u : static_cast<std::size_t>(s - 1); }
  int n_;
  std::uint32_t ones_[6]{};
  std::unordered_map<Key, double, KeyHash> memo_[4];
  std::unordered_map<std::uint32_t, int> opt_;
};

template <class F>
void for_each_subset(int words, int m, F&& f) {
  if (m == 0 || m > words) return;
  std::uint64_t v = (std::uint64_t{1} << m) - 1;
  const std::uint64_t limit = std::uint64_t{1} << words;
  while (v < limit) {
    f(static_cast<std::uint32_t>(v));
    const std::uint64_t c = v & (~v + 1);
    const std::uint64_t r = v + c;
    v = (((r ^ v) >> 2) / c) | r;
  }
}

Real falling(Real x, int k) {
  Real v = 1.0L;
  for (int i = 0; i < k; ++i) v *= x - i;
  return v;
}

Real binom(int a, int b) {
  if (b < 0 || b > a) return 0.0L;
  Real v = 1.0L;
  for (int i = 0; i < b; ++i) v = v * (a - i) / (i + 1);
  return v;
}

// Closed forms from the 1999 drafts for quality <= 1 probability.
Real p_quality_le1(int n, int m) {
  // Corrected upper summation limit: i runs to m, not to n.
  Real count = std::pow(static_cast<Real>(m + 1), static_cast<Real>(n));
  for (int i = 2; i <= m; ++i) {
    const Real sign = (i % 2 == 0) ? 1.0L : -1.0L;
    count -= sign * (i - 1) * binom(m, i) * std::pow(static_cast<Real>(m - i + 1), static_cast<Real>(n));
  }
  return std::ldexp(1.0L, n) * count / falling(std::ldexp(1.0L, n), m);
}

int cmd_brute(const std::string& out_path) {
  std::ofstream out(out_path);
  out.precision(15);
  out << "n,m,sets,t_s1,t_s2,t_s3,t_patricia,t_gradient,t_optimal,rec_s1,rec_s2,rec_s3,rec_patricia,"
         "p_quality_le1,p_formula\n";
  double max_gap = 0.0, max_p_gap = 0.0;
  const int mmax_for[] = {0, 2, 4, 8, 16, 7};
  for (int n = 1; n <= 5; ++n) {
    Brute b(n);
    // Finite-n recurrence values for the same (n, m).
    const int mtop = mmax_for[n];
    std::vector<std::vector<Real>> rec(4);
    for (int vi = 0; vi < 4; ++vi) {
      std::vector<Real> prev(static_cast<std::size_t>(mtop) + 1, 0.0L), cur = prev, w;
      for (int nn = 1; nn <= n; ++nn) {
        const Real H = std::ldexp(1.0L, nn - 1);
        std::fill(cur.begin(), cur.end(), 0.0L);
        for (int m = 2; m <= std::min(mtop, 1 << nn); ++m) {
          split_weights(H, m, w);
          cur[static_cast<std::size_t>(m)] = step(m, kThresholds[vi], w, prev);
        }
        prev.swap(cur);
      }
      rec[static_cast<std::size_t>(vi)] = prev;
    }
    for (int m = 2; m <= mtop; ++m) {
      double sums[4] = {0, 0, 0, 0};
      double sum_grad = 0, sum_opt = 0;
      std::uint64_t sets = 0, low_quality = 0;
      for_each_subset(1 << n, m, [&](std::uint32_t set) {
        ++sets;
        for (int vi = 0; vi < 4; ++vi) sums[vi] += b.random_tree(set, (1u << n) - 1u, kThresholds[vi]);
        sum_grad += b.gradient(set);
        sum_opt += b.optimal(set);
        if (b.matrix_quality(set) <= 1) ++low_quality;
      });
      const double denom = static_cast<double>(sets) * m;
      out << n << ',' << m << ',' << sets;
      for (double v : sums) out << ',' << v / denom;
      out << ',' << sum_grad / denom << ',' << sum_opt / denom;
      for (int vi = 0; vi < 4; ++vi) {
        const double r = static_cast<double>(rec[static_cast<std::size_t>(vi)][static_cast<std::size_t>(m)]);
        out << ',' << r;
        max_gap = std::max(max_gap, std::abs(r - sums[vi] / denom));
      }
      // The published probability counts ordered matrices; the same ratio holds for subsets.
      const double p_ex = static_cast<double>(low_quality) / static_cast<double>(sets);
      const double p_f = static_cast<double>(p_quality_le1(n, m));
      // The canonical-representative count needs k != m-k to be unambiguous, so m = 2 is excluded.
      if (m >= 3) max_p_gap = std::max(max_p_gap, std::abs(p_ex - p_f));
      out << ',' << p_ex << ',' << p_f << '\n';
    }
  }
  std::println("brute force vs recurrence: max |difference| = {:.3e}", max_gap);
  std::println("quality<=1 probability vs corrected formula (m>=3): max |difference| = {:.3e}", max_p_gap);
  return 0;
}

int cmd_identity(int nmax) {
  Real worst = 0.0L, worst_sum = 0.0L;
  std::vector<Real> w;
  for (int n = 1; n <= nmax; ++n) {
    const Real H = std::ldexp(1.0L, n - 1);
    const int top = n < 12 ? (1 << n) : 2048;
    for (int m = 2; m <= top; ++m) {
      split_weights(H, m, w);
      Real sum = 0.0L, total = 0.0L;
      for (int k = 0; k <= m; ++k) {
        sum += w[static_cast<std::size_t>(k)] * k * (m - k);
        total += w[static_cast<std::size_t>(k)];
      }
      const Real rhs = (2.0L * H / (2.0L * H - 1.0L)) * m * (m - 1) / 4.0L;
      worst = std::max(worst, std::abs(sum - rhs) / rhs);
      worst_sum = std::max(worst_sum, std::abs(total - 1.0L));
    }
  }
  std::println("identity relative error <= {:.3e}; weights sum to 1 within {:.3e}",
               static_cast<double>(worst), static_cast<double>(worst_sum));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "recurrence" && argc == 5) return cmd_recurrence(std::atoi(argv[2]), std::atoi(argv[3]), argv[4]);
  if (cmd == "limit" && argc == 4) return cmd_limit(std::atoi(argv[2]), argv[3]);
  if (cmd == "brute" && argc == 3) return cmd_brute(argv[2]);
  if (cmd == "identity" && argc == 3) return cmd_identity(std::atoi(argv[2]));
  std::println(stderr, "usage: exact_cost recurrence Nmax Mmax out.csv | limit Mmax out.csv | brute out.csv | identity Nmax");
  return 2;
}
