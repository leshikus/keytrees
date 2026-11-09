// ecc: prototype for identification keys that tolerate up to t wrong test answers.
//
// Species are n-bit words at pairwise Hamming distance >= 2t+1. A specimen's
// answers equal its word except on at most t positions. Errors are persistent:
// repeating a test would give the same answer, so a test is never repeated.
//
// Elimination key (new in this repository, 2026): keep for each species the
// number of disagreements so far, drop a species once it has more than t, and
// stop when a single species remains. Under the distance assumption the true
// species always survives and the walk always ends with exactly one survivor.
//
// Sphere key (1998 AUGMENT program): build an ordinary tree on all words in the
// radius-t balls around the species, so each ball word is a leaf. Variant
// "center" stops as soon as all remaining ball words belong to one species.
//
//   ecc sweep <seed> <runs> <out.csv>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace {

using Word = std::uint64_t;

double binom(int a, int b) {
  if (b < 0 || b > a) return 0.0;
  double v = 1.0;
  for (int i = 0; i < b; ++i) v = v * (a - i) / (i + 1);
  return v;
}

double volume(int n, int t) {
  // Cached for n <= 64 and t <= 3, the only values reached in the inner loop.
  static const auto table = [] {
    std::vector<double> v(65 * 4);
    for (int a = 0; a <= 64; ++a)
      for (int b = 0; b <= 3; ++b) {
        double s = 0.0;
        for (int i = 0; i <= b; ++i) s += binom(a, i);
        v[static_cast<std::size_t>(a * 4 + b)] = s;
      }
    return v;
  }();
  return table[static_cast<std::size_t>(n * 4 + t)];
}

// Random code with minimum distance d, built by random insertion.
std::vector<Word> random_code(int m, int n, int d, std::mt19937_64& rng) {
  const Word mask = n == 64 ? ~Word{0} : (Word{1} << n) - 1;
  for (int attempt = 0; attempt < 400; ++attempt) {
    std::vector<Word> code;
    for (int tries = 0; tries < 200000 && static_cast<int>(code.size()) < m; ++tries) {
      const Word w = rng() & mask;
      if (std::all_of(code.begin(), code.end(), [&](Word c) { return std::popcount(c ^ w) >= d; })) code.push_back(w);
    }
    if (static_cast<int>(code.size()) == m) return code;
  }
  return {};
}

enum class Policy { Sequential, Balance, Berlekamp };

struct Outcome {
  int tests;
  bool correct;
};

// Run the elimination key for one specimen. `answers` is its possibly corrupted word.
Outcome eliminate(const std::vector<Word>& code, int n, int t, Word answers, int truth, Policy policy) {
  const int m = static_cast<int>(code.size());
  std::vector<int> errors(static_cast<std::size_t>(m), 0);
  Word asked = 0;
  int alive = m, tests = 0;
  while (alive > 1) {
    int pick = -1;
    if (policy == Policy::Sequential) {
      pick = std::countr_zero(~asked);
    } else {
      const int remaining = n - tests - 1;
      double best = -1.0;
      for (int p = 0; p < n; ++p) {
        if ((asked >> p) & 1) continue;
        double w0 = 0.0, w1 = 0.0;
        for (int i = 0; i < m; ++i) {
          const int e = errors[static_cast<std::size_t>(i)];
          if (e > t) continue;
          const bool bit = (code[static_cast<std::size_t>(i)] >> p) & 1;
          if (policy == Policy::Balance) {
            (bit ? w1 : w0) += 1.0;
            continue;
          }
          // Berlekamp's weight after answering p: V(r, t-e) words remain on the agreeing
          // side and V(r, t-e-1) on the disagreeing side. These are exactly the words of the
          // species' ball that are consistent with the answers so far.
          const double agree = volume(remaining, t - e);
          const double disagree = e < t ? volume(remaining, t - e - 1) : 0.0;
          (bit ? w1 : w0) += agree;
          (bit ? w0 : w1) += disagree;
        }
        const double score = std::min(w0, w1);
        if (score > best) { best = score; pick = p; }
      }
    }
    if (pick < 0 || pick >= n) return {tests, false};  // cannot happen when distance >= 2t+1
    asked |= Word{1} << pick;
    ++tests;
    const Word bit = (answers >> pick) & 1;
    for (int i = 0; i < m; ++i) {
      auto& e = errors[static_cast<std::size_t>(i)];
      if (e <= t && ((code[static_cast<std::size_t>(i)] >> pick) & 1) != bit && ++e > t) --alive;
    }
  }
  int survivor = -1;
  for (int i = 0; i < m; ++i)
    if (errors[static_cast<std::size_t>(i)] <= t) survivor = i;
  return {tests, survivor == truth};
}

// ---------- 1998 sphere method ----------

struct Ball {
  Word word;
  int center;
};

// Node of the tree on ball words, built by balance on the words themselves.
struct Node {
  int position = -1;
  int left = -1, right = -1;
  int center = -1;  // set at a leaf
};

class SphereTree {
 public:
  SphereTree(const std::vector<Word>& code, int n, int t, bool stop_at_center) : n_(n), stop_(stop_at_center) {
    std::vector<Ball> balls;
    for (int c = 0; c < static_cast<int>(code.size()); ++c) add_ball(code[static_cast<std::size_t>(c)], c, t, 0, balls);
    root_ = build(balls, 0);
  }

  Outcome query(Word answers, int truth) const {
    int v = root_, depth = 0;
    while (nodes_[static_cast<std::size_t>(v)].position >= 0) {
      const auto& node = nodes_[static_cast<std::size_t>(v)];
      v = ((answers >> node.position) & 1) ? node.right : node.left;
      ++depth;
    }
    return {depth, nodes_[static_cast<std::size_t>(v)].center == truth};
  }

 private:
  void add_ball(Word w, int c, int t, int from, std::vector<Ball>& out) {
    out.push_back({w, c});
    if (t == 0) return;
    for (int p = from; p < n_; ++p) add_ball(w ^ (Word{1} << p), c, t - 1, p + 1, out);
  }

  int build(std::vector<Ball>& items, Word used) {
    const int id = static_cast<int>(nodes_.size());
    nodes_.emplace_back();
    const bool same_center = std::all_of(items.begin(), items.end(), [&](const Ball& b) { return b.center == items[0].center; });
    if (items.size() == 1 || (stop_ && same_center)) {
      nodes_[static_cast<std::size_t>(id)].center = items[0].center;
      return id;
    }
    int best = -1;
    std::size_t best_score = 0;
    for (int p = 0; p < n_; ++p) {
      if ((used >> p) & 1) continue;
      std::size_t ones = 0;
      for (const auto& b : items) ones += (b.word >> p) & 1;
      const std::size_t score = std::min(ones, items.size() - ones);
      if (score > best_score) { best_score = score; best = p; }
    }
    if (best < 0) std::abort();  // ball words are distinct, so some position splits them
    std::vector<Ball> zero, one;
    for (const auto& b : items) ((b.word >> best) & 1 ? one : zero).push_back(b);
    items.clear();
    items.shrink_to_fit();
    const int l = build(zero, used | (Word{1} << best));
    const int r = build(one, used | (Word{1} << best));
    nodes_[static_cast<std::size_t>(id)].position = best;
    nodes_[static_cast<std::size_t>(id)].left = l;
    nodes_[static_cast<std::size_t>(id)].right = r;
    return id;
  }

  int n_;
  bool stop_;
  int root_ = 0;
  std::vector<Node> nodes_;
};

Word corrupt(Word w, int n, int e, std::mt19937_64& rng) {
  std::uniform_int_distribution<int> pos(0, n - 1);
  Word flips = 0;
  while (std::popcount(flips) < e) flips |= Word{1} << pos(rng);
  return w ^ flips;
}

int sweep(std::uint64_t seed, int runs, const std::string& path) {
  std::ofstream out(path);
  out.precision(8);
  out << "m,n,t,errors,method,mean_tests,max_tests,failures,codes,volume_bound\n";
  std::mt19937_64 rng(seed);
  long long total_failures_within_t = 0, sphere_comparisons = 0, sphere_mismatches = 0;

  struct Config { int m, n; };
  const Config configs[] = {{8, 24}, {16, 24}, {32, 24}, {64, 32}, {128, 40}, {256, 48}, {512, 56}};
  for (const auto [m, n] : configs) {
    for (int t = 0; t <= 3; ++t) {
      const int d = 2 * t + 1;
      // Worst-case bound for adaptive search with t lies: smallest q with 2^q >= m V(q,t).
      int q = 1;
      while (std::log2(static_cast<double>(m)) + std::log2(volume(q, t)) > q) ++q;

      struct Tally { double sum = 0; int max = 0; long long fail = 0; long long count = 0; };
      Tally seq[4], bal[4], ber[4], sph_word[4], sph_center[4];
      const bool sphere_ok = (volume(n, t) * m < 400000.0);
      const int codes = 8;
      for (int c = 0; c < codes; ++c) {
        const auto code = random_code(m, n, d, rng);
        if (code.empty()) {
          std::println(stderr, "no code for m={} n={} d={}", m, n, d);
          return 1;
        }
        std::vector<SphereTree> trees;
        if (sphere_ok) {
          trees.emplace_back(code, n, t, false);
          trees.emplace_back(code, n, t, true);
        }
        auto record = [](Tally& tl, const Outcome& o) {
          tl.sum += o.tests;
          tl.max = std::max(tl.max, o.tests);
          tl.fail += o.correct ? 0 : 1;
          ++tl.count;
        };
        for (int run = 0; run < runs; ++run) {
          const int truth = std::uniform_int_distribution<int>(0, m - 1)(rng);
          const Word w = code[static_cast<std::size_t>(truth)];
          for (int e = 0; e <= t; ++e) {
            const Word a = corrupt(w, n, e, rng);
            record(seq[e], eliminate(code, n, t, a, truth, Policy::Sequential));
            record(bal[e], eliminate(code, n, t, a, truth, Policy::Balance));
            const Outcome by_weight = eliminate(code, n, t, a, truth, Policy::Berlekamp);
            record(ber[e], by_weight);
            if (sphere_ok) {
              record(sph_word[e], trees[0].query(a, truth));
              const Outcome by_center = trees[1].query(a, truth);
              record(sph_center[e], by_center);
              ++sphere_comparisons;
              if (by_center.tests != by_weight.tests) ++sphere_mismatches;
            }
          }
        }
      }
      auto emit = [&](const char* name, const Tally* tally) {
        for (int e = 0; e <= t; ++e) {
          const auto& tl = tally[e];
          if (!tl.count) continue;
          total_failures_within_t += tl.fail;
          out << m << ',' << n << ',' << t << ',' << e << ',' << name << ',' << tl.sum / static_cast<double>(tl.count)
              << ',' << tl.max << ',' << tl.fail << ',' << codes << ',' << q << '\n';
        }
      };
      emit("elim_sequential", seq);
      emit("elim_balance", bal);
      emit("elim_berlekamp", ber);
      emit("sphere_word", sph_word);
      emit("sphere_center", sph_center);
    }
  }
  std::println("wrong identifications with at most t errors, all methods: {}", total_failures_within_t);
  std::println("sphere key vs weighted elimination: {} of {} queries differ in test count", sphere_mismatches,
               sphere_comparisons);
  return total_failures_within_t == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "sweep" && argc == 5) return sweep(std::strtoull(argv[2], nullptr, 10), std::atoi(argv[3]), argv[4]);
  std::println(stderr, "usage: ecc sweep seed runs out.csv");
  return 2;
}
