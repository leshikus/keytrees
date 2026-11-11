// elimination: the elimination key on uniformly random codes, for the article in
// articles/elimination-keys/ (new in 2026; not a reconstruction of older work).
//
// A code is m independent uniform words of length n; it is NOT conditioned on its
// minimum distance. A specimen of species i answers c_i with some positions flipped
// (persistent errors). The elimination key keeps for every species the number e_j
// of disagreements, drops it once e_j > t, and stops when one species remains or
// all n tests have been asked ("exhausted"; possible only without the distance
// condition). With at most t errors the true species is never dropped.
//
// Rules for choosing the next test:
//   order    : positions 0, 1, 2, ... (the rule of Theorem 1 of the article)
//   weighted : Berlekamp weights V(r, t - e_j), r = tests left after this one; the
//              test making the two answers' total weights most even (prop:equivalence)
//   leader   : if the least count e is shared, balance that group; otherwise
//              ask the test on which most survivors disagree with the unique leader
//   oracle   : knows the true species i and asks the test on which most survivors
//              disagree with c_i (greedy set multicover; an upper bound on the
//              oracle optimum, whose lower bound is Theorem 4 of the article)
// Errors: none; t at uniformly random positions; or the first t asked tests.
//
//   elimination sweep <seed> <codes> <specimens> <out.csv>
//   elimination order <seed> <m> <n> <t> <codes> <specimens>   (check of Theorem 1)
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace {

using Block = std::uint64_t;

enum class Rule { Order, Weighted, Leader, Oracle };
enum class Errors { None, Random, First };

constexpr const char* rule_name(Rule r) {
  switch (r) {
    case Rule::Order: return "order";
    case Rule::Weighted: return "weighted";
    case Rule::Leader: return "leader";
    case Rule::Oracle: return "oracle";
  }
  return "";
}

constexpr const char* errors_name(Errors e) {
  switch (e) {
    case Errors::None: return "none";
    case Errors::Random: return "random";
    case Errors::First: return "first";
  }
  return "";
}

// V(r, s) = sum_{i <= s} binom(r, i), with V(r, s) = 0 for s < 0.
double volume(int r, int s) {
  if (s < 0 || r < 0) return 0.0;
  double v = 0.0, c = 1.0;
  for (int i = 0; i <= s && i <= r; ++i) {
    v += c;
    c = c * (r - i) / (i + 1);
  }
  return v;
}

// A code stored by columns: column p is a bitset over the species.
struct Code {
  int m = 0, n = 0;
  std::vector<std::vector<Block>> col;  // n columns of ceil(m/64) blocks
  bool bit(int p, int j) const { return (col[p][j >> 6] >> (j & 63)) & 1; }
};

Code random_code(int m, int n, std::mt19937_64& rng) {
  Code c;
  c.m = m;
  c.n = n;
  const int blocks = (m + 63) / 64;
  c.col.assign(n, std::vector<Block>(blocks));
  for (auto& column : c.col) {
    for (auto& b : column) b = rng();
    if (m % 64) column.back() &= (Block{1} << (m % 64)) - 1;
  }
  return c;
}

struct Result {
  int tests;
  bool exhausted;
  bool correct;
};

// One identification. Columns are restricted to the survivors and rebuilt when
// the survivors halve, so the cost per test is about n * survivors / 64.
Result identify(const Code& code, int t, int truth, Rule rule, Errors errors, std::mt19937_64& rng) {
  const int m = code.m, n = code.n;
  std::vector<char> flipped(n, 0);
  if (errors == Errors::Random) {
    int placed = 0;
    std::uniform_int_distribution<int> pos(0, n - 1);
    while (placed < std::min(t, n)) {
      const int p = pos(rng);
      if (!flipped[p]) flipped[p] = 1, ++placed;
    }
  }
  std::vector<char> asked(n, 0);
  int tests = 0;
  if (rule == Rule::Order) {  // no columns needed: track survivors directly
    std::vector<int> ids(m), e(m, 0);
    for (int j = 0; j < m; ++j) ids[j] = j;
    while (ids.size() > 1 && tests < n) {
      const int p = tests++;
      bool answer = code.bit(p, truth);
      if ((errors == Errors::Random && flipped[p]) || (errors == Errors::First && tests <= t)) answer = !answer;
      std::size_t w = 0;
      for (std::size_t a = 0; a < ids.size(); ++a) {
        const int ea = e[a] + (code.bit(p, ids[a]) != answer);
        if (ea <= t) ids[w] = ids[a], e[w] = ea, ++w;
      }
      ids.resize(w);
      e.resize(w);
    }
    const bool correct = std::find(ids.begin(), ids.end(), truth) != ids.end();
    return {tests, ids.size() > 1, correct};
  }
  // compact index space, fixed between rebuilds
  std::vector<int> ids(m);
  for (int j = 0; j < m; ++j) ids[j] = j;
  std::vector<int> e(m, 0);
  std::vector<char> alive(m, 1);
  int k = m, live = m;
  std::vector<std::vector<Block>> col(n);
  auto rebuild = [&] {
    std::vector<int> nid, ne;
    for (int a = 0; a < k; ++a)
      if (alive[a]) nid.push_back(ids[a]), ne.push_back(e[a]);
    ids = std::move(nid);
    e = std::move(ne);
    k = live = static_cast<int>(ids.size());
    alive.assign(k, 1);
    const int blocks = (k + 63) / 64;
    for (int p = 0; p < n; ++p) {
      if (asked[p]) { col[p].clear(); continue; }
      auto& c = col[p];
      c.assign(blocks, 0);
      for (int a = 0; a < k; ++a)
        if (code.bit(p, ids[a])) c[a >> 6] |= Block{1} << (a & 63);
    }
  };
  rebuild();
  while (live > 1 && tests < n) {
    const int blocks = (k + 63) / 64;
    std::vector<std::vector<Block>> group(t + 1, std::vector<Block>(blocks, 0));
    std::vector<Block> live_mask(blocks, 0);
    std::vector<int> group_size(t + 1, 0);
    int least = t + 1;
    for (int a = 0; a < k; ++a) {
      if (!alive[a]) continue;
      group[e[a]][a >> 6] |= Block{1} << (a & 63);
      live_mask[a >> 6] |= Block{1} << (a & 63);
      ++group_size[e[a]];
      least = std::min(least, e[a]);
    }
    auto count = [&](const std::vector<Block>& c, const std::vector<Block>& g) {
      int s = 0;
      for (int b = 0; b < blocks; ++b) s += std::popcount(c[b] & g[b]);
      return s;
    };
    int ref = -1;
    bool balance_least = false;
    if (rule == Rule::Oracle) {
      for (int a = 0; a < k; ++a)
        if (ids[a] == truth) ref = a;
    } else if (rule == Rule::Leader) {
      if (group_size[least] > 1) balance_least = true;
      else
        for (int a = 0; a < k; ++a)
          if (alive[a] && e[a] == least) ref = a;
    }
    const int r = n - tests - 1;
    double best = -1.0;
    int pick = -1;
    for (int p = 0; p < n; ++p) {
      if (asked[p]) continue;
      const auto& c = col[p];
      double score;
      if (rule == Rule::Weighted) {
        double w1 = 0.0, total = 0.0;
        for (int s = 0; s <= t; ++s) {
          if (!group_size[s]) continue;
          const int ones = count(c, group[s]);
          const double agree = volume(r, t - s), disagree = volume(r, t - s - 1);
          w1 += ones * agree + (group_size[s] - ones) * disagree;
          total += group_size[s] * (agree + disagree);
        }
        score = std::min(w1, total - w1);
      } else if (balance_least) {
        const int ones = count(c, group[least]);
        score = std::min(ones, group_size[least] - ones);
      } else {
        const int ones = count(c, live_mask);
        const bool rb = (c[ref >> 6] >> (ref & 63)) & 1;
        score = rb ? live - ones : ones;  // survivors disagreeing with the reference
      }
      if (score > best) best = score, pick = p;
    }
    asked[pick] = 1;
    ++tests;
    bool answer = code.bit(pick, truth);
    if ((errors == Errors::Random && flipped[pick]) || (errors == Errors::First && tests <= t)) answer = !answer;
    for (int a = 0; a < k; ++a) {
      if (!alive[a]) continue;
      const bool b = (col[pick][a >> 6] >> (a & 63)) & 1;
      if (b != answer && ++e[a] > t) alive[a] = 0, --live;
    }
    if (live > 1 && 2 * live <= k) rebuild();
  }
  bool correct = false;
  for (int a = 0; a < k; ++a)
    if (alive[a] && ids[a] == truth) correct = true;
  return {tests, live > 1, correct};
}

struct Stats {
  double sum = 0, sumsq_code = 0;
  long long count = 0, exhausted = 0, wrong = 0;
  int max = 0;
  std::vector<double> code_means;
};

int sweep(std::uint64_t seed, int codes, int specimens, const std::string& path) {
  std::ofstream out(path);
  out.precision(8);
  out << "regime,m,n,t,errors,rule,codes,specimens,mean_tests,stderr,max_tests,exhausted,wrong\n";
  std::mt19937_64 rng(seed);
  long long wrong_total = 0;
  struct Regime { const char* name; int lo, hi; };
  const Regime regimes[] = {{"4log", 4, 16}, {"linear", 4, 11}};
  const Rule rules[] = {Rule::Order, Rule::Weighted, Rule::Leader, Rule::Oracle};
  const Errors errs[] = {Errors::None, Errors::Random, Errors::First};
  for (const auto& reg : regimes) {
    for (int L = reg.lo; L <= reg.hi; ++L) {
      const int m = 1 << L;
      const int n = std::string(reg.name) == "4log" ? 4 * L : m;
      for (int t = 0; t <= 3; ++t) {
        Stats st[4][3];
        for (int c = 0; c < codes; ++c) {
          const Code code = random_code(m, n, rng);
          double code_sum[4][3] = {};
          for (int s = 0; s < specimens; ++s) {
            const int truth = std::uniform_int_distribution<int>(0, m - 1)(rng);
            for (int ri = 0; ri < 4; ++ri)
              for (int ei = 0; ei < 3; ++ei) {
                if (t == 0 && ei > 0) continue;
                const Result res = identify(code, t, truth, rules[ri], errs[ei], rng);
                auto& x = st[ri][ei];
                x.sum += res.tests;
                ++x.count;
                x.max = std::max(x.max, res.tests);
                x.exhausted += res.exhausted;
                x.wrong += !res.correct;
                code_sum[ri][ei] += res.tests;
              }
          }
          for (int ri = 0; ri < 4; ++ri)
            for (int ei = 0; ei < 3; ++ei) st[ri][ei].code_means.push_back(code_sum[ri][ei] / specimens);
        }
        for (int ri = 0; ri < 4; ++ri)
          for (int ei = 0; ei < 3; ++ei) {
            const auto& x = st[ri][ei];
            if (!x.count) continue;
            const double mean = x.sum / static_cast<double>(x.count);
            double var = 0;
            for (double v : x.code_means) var += (v - mean) * (v - mean);
            const double se = codes > 1 ? std::sqrt(var / (codes - 1) / codes) : 0.0;
            wrong_total += x.wrong;
            out << reg.name << ',' << m << ',' << n << ',' << t << ',' << errors_name(errs[ei]) << ','
                << rule_name(rules[ri]) << ',' << codes << ',' << specimens << ',' << mean << ',' << se << ','
                << x.max << ',' << x.exhausted << ',' << x.wrong << '\n';
          }
        out.flush();
        std::println("{} m={} n={} t={} done", reg.name, m, n, t);
      }
    }
  }
  std::println("specimens with at most t errors whose species was dropped: {}", wrong_total);
  return wrong_total == 0 ? 0 : 1;
}

int order_check(std::uint64_t seed, int m, int n, int t, int codes, int specimens) {
  std::mt19937_64 rng(seed);
  double sum = 0;
  long long count = 0;
  for (int c = 0; c < codes; ++c) {
    const Code code = random_code(m, n, rng);
    for (int s = 0; s < specimens; ++s) {
      const int truth = std::uniform_int_distribution<int>(0, m - 1)(rng);
      sum += identify(code, t, truth, Rule::Order, Errors::None, rng).tests;
      ++count;
    }
  }
  std::println("{:.6f}", sum / static_cast<double>(count));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "sweep" && argc == 6)
    return sweep(std::strtoull(argv[2], nullptr, 10), std::atoi(argv[3]), std::atoi(argv[4]), argv[5]);
  if (cmd == "order" && argc == 8)
    return order_check(std::strtoull(argv[2], nullptr, 10), std::atoi(argv[3]), std::atoi(argv[4]),
                       std::atoi(argv[5]), std::atoi(argv[6]), std::atoi(argv[7]));
  std::println(stderr, "usage: elimination sweep seed codes specimens out.csv | order seed m n t codes specimens");
  return 2;
}
