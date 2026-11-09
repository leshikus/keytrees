// rta: C++23 port of the 1998 GENDATA | RTA experiment pipeline by A. A. Fedotov.
//
// The original programs passed text tables through a pipe and used the C library
// rand(), so its exact random data cannot be reproduced. This port keeps the
// model (random binary tests with distinct species and distinct tests, tests
// count = ceil(coef * log2 m)) and the methods, and adds what the original left
// as a stub: the true optimal tree for the given tests.
//
//   rta sweep <coef> <mmin> <mmax> <samples> <seed> <out.csv> [uniform|random] [opt_mmax]
//
// Output columns are averages over samples for one m:
//   entropy      lower bound H(p)
//   shannon      sum p_i ceil(-log2 p_i)
//   huffman      optimal cost when any test may be used
//   greedy_up    original "Greedy Up" estimate
//   greedy_md    original "Greedy Md" estimate, an entropy-like criterion
//   random       Patricia-style random informative test
//   optimal      exact optimum for the given tests, only for m <= opt_mmax
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <print>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Set = std::array<std::uint64_t, 2>;  // up to 128 species
struct SetHash {
  std::size_t operator()(const Set& s) const noexcept {
    return std::hash<std::uint64_t>{}(s[0] * 0x9E3779B97F4A7C15ULL ^ (s[1] + 0x7F4A7C15ULL));
  }
};

Set operator&(const Set& a, const Set& b) { return {a[0] & b[0], a[1] & b[1]}; }
Set andnot(const Set& a, const Set& b) { return {a[0] & ~b[0], a[1] & ~b[1]}; }
int count(const Set& s) { return std::popcount(s[0]) + std::popcount(s[1]); }
bool empty(const Set& s) { return !(s[0] | s[1]); }

struct Instance {
  int m = 0, n = 0;
  std::vector<double> p;
  std::vector<Set> test;  // test[j] holds the species for which the test is positive
  double weight(const Set& s) const {
    double w = 0.0;
    for (int word = 0; word < 2; ++word) {
      std::uint64_t bits = s[static_cast<std::size_t>(word)];
      while (bits) {
        const int i = std::countr_zero(bits);
        w += p[static_cast<std::size_t>(word * 64 + i)];
        bits &= bits - 1;
      }
    }
    return w;
  }
};

double entropy(const std::vector<double>& p) {
  double h = 0.0;
  for (double x : p)
    if (x > 0) h -= x * std::log2(x);
  return h;
}

double shannon(const std::vector<double>& p) {
  double s = 0.0;
  for (double x : p)
    if (x > 0) s += x * std::ceil(-std::log2(x));
  return s;
}

double huffman(const std::vector<double>& p) {
  std::priority_queue<double, std::vector<double>, std::greater<>> q(p.begin(), p.end());
  double cost = 0.0;
  while (q.size() > 1) {
    const double a = q.top();
    q.pop();
    const double b = q.top();
    q.pop();
    cost += a + b;
    q.push(a + b);
  }
  return cost;
}

enum class Rule { Up, Md, Random };

class Builder {
 public:
  Builder(const Instance& inst, Rule rule, std::mt19937_64& rng) : I(inst), rule_(rule), rng_(rng) {}

  double cost(const Set& s) {
    const int size = count(s);
    if (size <= 1) return 0.0;
    int best = -1;
    double best_score = std::numeric_limits<double>::infinity();
    int seen = 0;
    for (int j = 0; j < I.n; ++j) {
      const Set a = s & I.test[static_cast<std::size_t>(j)];
      const Set b = andnot(s, I.test[static_cast<std::size_t>(j)]);
      if (empty(a) || empty(b)) continue;
      if (rule_ == Rule::Random) {
        ++seen;
        if (std::uniform_int_distribution<int>(1, seen)(rng_) == 1) best = j;
        continue;
      }
      const double na = count(a), nb = count(b), pa = I.weight(a), pb = I.weight(b);
      const double score = rule_ == Rule::Up ? na * pa + nb * pb : std::log2(na) * pa + std::log2(nb) * pb;
      if (score < best_score) { best_score = score; best = j; }
    }
    if (best < 0) std::abort();  // species are distinct, so some test must split them
    const Set a = s & I.test[static_cast<std::size_t>(best)];
    const Set b = andnot(s, I.test[static_cast<std::size_t>(best)]);
    return I.weight(s) + cost(a) + cost(b);
  }

 private:
  const Instance& I;
  Rule rule_;
  std::mt19937_64& rng_;
};

class Optimal {
 public:
  explicit Optimal(const Instance& inst) : I(inst) {}
  double cost(const Set& s) {
    if (count(s) <= 1) return 0.0;
    if (auto it = memo_.find(s); it != memo_.end()) return it->second;
    double best = std::numeric_limits<double>::infinity();
    const double here = I.weight(s);
    for (int j = 0; j < I.n; ++j) {
      const Set a = s & I.test[static_cast<std::size_t>(j)];
      const Set b = andnot(s, I.test[static_cast<std::size_t>(j)]);
      if (empty(a) || empty(b)) continue;
      best = std::min(best, here + cost(a) + cost(b));
    }
    memo_.emplace(s, best);
    return best;
  }
  std::size_t states() const { return memo_.size(); }

 private:
  const Instance& I;
  std::unordered_map<Set, double, SetHash> memo_;
};

// The original generator: tests drawn at random, each new test distinct from the
// earlier ones, and the whole draw repeated until all species are distinct.
Instance generate(int m, int n, bool random_probs, std::mt19937_64& rng) {
  Instance I;
  I.m = m;
  I.n = n;
  I.p.assign(static_cast<std::size_t>(m), 1.0 / m);
  if (random_probs) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double s = 0.0;
    for (auto& x : I.p) s += (x = u(rng));
    for (auto& x : I.p) x /= s;
    std::sort(I.p.rbegin(), I.p.rend());
  }
  // Draw m distinct n-bit words directly. Conditioned on distinct species this is the same
  // distribution as the original rejection loop, which stalls near m ~ 2^(n/2): that stall
  // ended the 1999 runs at low test counts, with "Failed to find system of distinguishable
  // criterions" reaching RTA through the pipe as "Error 108".
  const std::uint64_t word_mask = n >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << n) - 1;
  for (;;) {
    std::vector<std::uint64_t> words;
    words.reserve(static_cast<std::size_t>(m));
    while (static_cast<int>(words.size()) < m) {
      const std::uint64_t w = rng() & word_mask;
      if (std::find(words.begin(), words.end(), w) == words.end()) words.push_back(w);
    }
    I.test.assign(static_cast<std::size_t>(n), Set{});
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        if ((words[static_cast<std::size_t>(i)] >> j) & 1)
          I.test[static_cast<std::size_t>(j)][static_cast<std::size_t>(i / 64)] |= std::uint64_t{1} << (i % 64);
    // The original also required distinct tests.
    auto sorted = I.test;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end()) return I;
  }
}

int sweep(double coef, int mmin, int mmax, int samples, std::uint64_t seed, const std::string& path, bool random_probs,
          int opt_mmax) {
  if (mmax > 127) {
    std::println(stderr, "mmax must be at most 127");
    return 2;
  }
  std::ofstream out(path);
  out.precision(10);
  out << "m,n,samples,entropy,shannon,huffman,greedy_up,greedy_md,random,optimal,up_minus_huffman,up_minus_huffman_se,"
         "up_minus_optimal\n";
  std::mt19937_64 rng(seed);
  for (int m = mmin; m <= mmax; ++m) {
    const int n = std::max(static_cast<int>(std::ceil(coef * std::log2(m))),
                           static_cast<int>(std::ceil(std::log2(m))));
    if (n > 62) continue;
    double e = 0, sh = 0, hf = 0, up = 0, md = 0, rd = 0, op = 0, diff2 = 0, up_opt = 0;
    int opt_runs = 0;
    for (int run = 0; run < samples; ++run) {
      const Instance I = generate(m, n, random_probs, rng);
      Set all{};
      for (int i = 0; i < m; ++i) all[static_cast<std::size_t>(i / 64)] |= std::uint64_t{1} << (i % 64);
      const double h = huffman(I.p);
      const double g = Builder(I, Rule::Up, rng).cost(all);
      e += entropy(I.p);
      sh += shannon(I.p);
      hf += h;
      up += g;
      md += Builder(I, Rule::Md, rng).cost(all);
      rd += Builder(I, Rule::Random, rng).cost(all);
      diff2 += (g - h) * (g - h);
      if (m <= opt_mmax) {
        const double o = Optimal(I).cost(all);
        op += o;
        up_opt += g - o;
        ++opt_runs;
      }
    }
    const double N = samples;
    const double mean_diff = (up - hf) / N;
    const double var = std::max(0.0, diff2 / N - mean_diff * mean_diff);
    out << m << ',' << n << ',' << samples << ',' << e / N << ',' << sh / N << ',' << hf / N << ',' << up / N << ','
        << md / N << ',' << rd / N << ',';
    if (opt_runs) out << op / opt_runs;
    out << ',' << mean_diff << ',' << std::sqrt(var / N) << ',';
    if (opt_runs) out << up_opt / opt_runs;
    out << '\n';
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "sweep" && argc >= 8) {
    const bool random_probs = argc > 8 && std::string(argv[8]) == "random";
    const int opt_mmax = argc > 9 ? std::atoi(argv[9]) : 0;
    return sweep(std::atof(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]),
                 std::strtoull(argv[6], nullptr, 10), argv[7], random_probs, opt_mmax);
  }
  std::println(stderr, "usage: rta sweep coef mmin mmax samples seed out.csv [uniform|random] [opt_mmax]");
  return 2;
}
