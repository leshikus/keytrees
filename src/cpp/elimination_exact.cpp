// elimination_exact: exact mean cost of the elimination key over ALL codes, for the
// article in articles/elimination-keys/ (new in 2026). No sampling: integer sums.
//
// Averages over all tables of m words of length n (model "all"), or over all tables
// of minimum distance >= 2t+1 (model "dist"), over the true species (uniform) and
// over the error sets of each weight (uniform). Rules are those of elimination.cpp;
// ties go to the lowest column index.
//
// Symmetries used (each is a bijection on tables preserving the cost):
//  - complementing a column in all rows and in the answer: so the true word is 0;
//  - permuting rows: so the true species is row 0, and the other rows are a
//    multiset, counted with its multinomial weight.
// Permuting columns is NOT a symmetry of a rule that breaks ties by column index,
// so it is not used; see the article.
//
//   elimination_exact <all|dist> <m> <n> <t> <optimal:0|1>    (CSV rows on stdout)
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <print>
#include <string>
#include <thread>
#include <vector>

namespace {

using u32 = std::uint32_t;
using i64 = std::int64_t;
using u128 = unsigned __int128;

constexpr int kMaxM = 16;
enum Rule { Order = 0, Weighted = 1, Leader = 2, kRules = 3 };
const char* const kRuleName[kRules] = {"order", "weighted", "leader"};

i64 vol[40][8];  // V(r, s) for s >= 0

void init_volume() {
  for (int r = 0; r < 40; ++r)
    for (int s = 0; s < 8; ++s) {
      i64 v = 0, c = 1;
      for (int i = 0; i <= s && i <= r; ++i) {
        v += c;
        c = c * (r - i) / (i + 1);
      }
      vol[r][s] = v;
    }
}
inline i64 V(int r, int s) { return s < 0 ? 0 : vol[r][s]; }

struct Problem {
  int m, n, t;
};

// Number of tests for a specimen with answer word a; row 0 is the true species.
int walk(const Problem& P, const u32* c, u32 a, Rule rule) {
  const int m = P.m, n = P.n, t = P.t;
  int e[kMaxM] = {};
  int live = m, tests = 0;
  u32 asked = 0;
  while (live > 1 && tests < n) {
    int pick = -1;
    if (rule == Order) {
      pick = tests;
    } else {
      int size[8] = {};
      int least = t + 1;
      for (int j = 0; j < m; ++j)
        if (e[j] <= t) ++size[e[j]], least = std::min(least, e[j]);
      int ref = -1;
      const bool balance = rule == Leader && size[least] > 1;
      if (rule == Leader && !balance)
        for (int j = 0; j < m; ++j)
          if (e[j] == least) ref = j;
      const int r = n - tests - 1;
      i64 best = -1;
      for (int p = 0; p < n; ++p) {
        if ((asked >> p) & 1) continue;
        i64 score;
        if (rule == Weighted) {
          i64 w1 = 0, total = 0;
          for (int j = 0; j < m; ++j) {
            if (e[j] > t) continue;
            const i64 agree = V(r, t - e[j]), disagree = V(r, t - e[j] - 1);
            w1 += ((c[j] >> p) & 1) ? agree : disagree;
            total += agree + disagree;
          }
          score = std::min(w1, total - w1);
        } else if (balance) {
          int ones = 0;
          for (int j = 0; j < m; ++j)
            if (e[j] == least) ones += (c[j] >> p) & 1;
          score = std::min(ones, size[least] - ones);
        } else {
          int differ = 0;
          const u32 rb = (c[ref] >> p) & 1;
          for (int j = 0; j < m; ++j)
            if (e[j] <= t) differ += ((c[j] >> p) & 1) != rb;
          score = differ;
        }
        if (score > best) best = score, pick = p;
      }
    }
    asked |= u32{1} << pick;
    ++tests;
    const u32 bit = (a >> pick) & 1;
    for (int j = 0; j < m; ++j)
      if (e[j] <= t && ((c[j] >> pick) & 1) != bit && ++e[j] > t) --live;
  }
  return tests;
}

struct Totals {
  // per weight w = 0..t and rule: sum of tests; and of the worst over weights <= t
  u128 cost[8][kRules] = {};
  u128 worst[kRules] = {};
  u128 oracle = 0, optimal = 0;
  u128 tables = 0;  // sum of table weights
  void add(const Totals& o, int t) {
    for (int w = 0; w <= t; ++w)
      for (int r = 0; r < kRules; ++r) cost[w][r] += o.cost[w][r];
    for (int r = 0; r < kRules; ++r) worst[r] += o.worst[r];
    oracle += o.oracle;
    optimal += o.optimal;
    tables += o.tables;
  }
};

// Least number of tests of a no-error key, summed over species, for one table.
class Optimal {
 public:
  Optimal(const Problem& P) : P_(P), memo_(std::size_t{1} << (P.n + P.m), 0), stamp_(memo_.size(), 0) {}
  int run(const u32* c) {
    c_ = c;
    ++generation_;
    return f(0, (u32{1} << P_.m) - 1);
  }

 private:
  int f(u32 S, u32 Z) {
    const std::size_t key = (std::size_t{S} << P_.m) | Z;
    if (stamp_[key] == generation_) return memo_[key];
    const int z = std::countr_zero(Z);
    int alive = 0;
    for (int j = 0; j < P_.m; ++j) alive += std::popcount((c_[j] ^ c_[z]) & S) <= P_.t;
    int best = 0;
    const u32 full = (u32{1} << P_.n) - 1;
    if (alive > 1 && S != full) {
      best = 1 << 28;
      for (int p = 0; p < P_.n; ++p) {
        if ((S >> p) & 1) continue;
        u32 Z1 = 0;
        for (int j = 0; j < P_.m; ++j)
          if (((Z >> j) & 1) && ((c_[j] >> p) & 1)) Z1 |= u32{1} << j;
        const u32 Z0 = Z & ~Z1, T = S | (u32{1} << p);
        int v = std::popcount(Z);
        if (Z0) v += f(T, Z0);
        if (Z1) v += f(T, Z1);
        best = std::min(best, v);
      }
    }
    stamp_[key] = generation_;
    memo_[key] = best;
    return best;
  }
  Problem P_;
  const u32* c_ = nullptr;
  std::vector<int> memo_;
  std::vector<u32> stamp_;
  u32 generation_ = 0;
};

struct Worker {
  Problem P;
  bool use_optimal;
  const std::vector<std::vector<u32>>* errors;  // error sets by weight
  const std::vector<u32>* by_size;              // all column sets by increasing size
  Totals tot;
  Optimal opt;
  Worker(const Problem& p, bool o, const std::vector<std::vector<u32>>* e, const std::vector<u32>* s)
      : P(p), use_optimal(o), errors(e), by_size(s), opt(o ? p : Problem{1, 1, 0}) {}

  void table(const u32* c, u128 weight) {
    tot.tables += weight;
    int worst[kRules] = {};
    for (int w = 0; w <= P.t && w <= P.n; ++w)
      for (u32 E : (*errors)[w])
        for (int r = 0; r < kRules; ++r) {
          const int k = walk(P, c, E, static_cast<Rule>(r));
          tot.cost[w][r] += weight * static_cast<unsigned>(k);
          worst[r] = std::max(worst[r], k);
        }
    for (int r = 0; r < kRules; ++r) tot.worst[r] += weight * static_cast<unsigned>(worst[r]);
    int star = P.n;
    for (u32 S : *by_size) {
      if (std::popcount(S) >= star) break;
      bool ok = true;
      for (int j = 1; j < P.m && ok; ++j) ok = std::popcount(c[j] & S) > P.t;
      if (ok) { star = std::popcount(S); break; }
    }
    tot.oracle += weight * static_cast<unsigned>(star);
    if (use_optimal) tot.optimal += weight * static_cast<unsigned>(opt.run(c));
  }
};

u128 factorial(int k) {
  u128 f = 1;
  for (int i = 2; i <= k; ++i) f *= static_cast<unsigned>(i);
  return f;
}

// Enumerate the multisets of the m-1 wrong words (model all).
void all_rec(Worker& W, u32* c, int j, u32 from, u128 weight, int run) {
  const u32 words = u32{1} << W.P.n;
  if (j == W.P.m) { W.table(c, weight); return; }
  for (u32 x = from; x < words; ++x) {
    const int r = (j > 1 && x == c[j - 1]) ? run + 1 : 1;
    c[j] = x;
    all_rec(W, c, j + 1, x, weight / static_cast<unsigned>(r), r);
  }
}

// Enumerate the sets of m-1 wrong words at distance >= d from 0 and each other.
void dist_rec(Worker& W, u32* c, int j, u32 from, int d, u128 weight) {
  const u32 words = u32{1} << W.P.n;
  if (j == W.P.m) { W.table(c, weight); return; }
  for (u32 x = from; x < words; ++x) {
    bool ok = true;
    for (int i = 0; i < j && ok; ++i) ok = std::popcount(c[i] ^ x) >= d;
    if (!ok) continue;
    c[j] = x;
    dist_rec(W, c, j + 1, x + 1, d, weight);
  }
}

std::string str(u128 v) {
  if (!v) return "0";
  std::string s;
  while (v) s.push_back(static_cast<char>('0' + static_cast<int>(v % 10))), v /= 10;
  std::reverse(s.begin(), s.end());
  return s;
}
u128 gcd(u128 a, u128 b) {
  while (b) { const u128 r = a % b; a = b; b = r; }
  return a;
}

void emit(const std::string& model, const Problem& P, const std::string& errors, const char* rule, u128 num, u128 den,
          u128 tables) {
  const u128 g = gcd(num, den);
  num /= g;
  den /= g;
  std::println("{},{},{},{},{},{},{},{},{:.10f},{}", model, P.m, P.n, P.t, errors, rule, str(num), str(den),
               static_cast<double>(num) / static_cast<double>(den), str(tables));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::println(stderr, "usage: elimination_exact <all|dist> m n t optimal");
    return 2;
  }
  const std::string model = argv[1];
  const Problem P{std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4])};
  const bool use_optimal = std::atoi(argv[5]) != 0;
  if (P.m < 2 || P.m > kMaxM || P.n > 30 || P.t > 6) return 2;
  init_volume();

  std::vector<std::vector<u32>> errors(P.t + 1);
  std::vector<u32> by_size;
  for (u32 S = 0; S < (u32{1} << P.n); ++S) {
    by_size.push_back(S);
    if (std::popcount(S) <= P.t) errors[std::popcount(S)].push_back(S);
  }
  std::stable_sort(by_size.begin(), by_size.end(), [](u32 a, u32 b) { return std::popcount(a) < std::popcount(b); });

  const int threads = std::max(1u, std::thread::hardware_concurrency());
  std::atomic<u32> next{0};
  std::mutex lock;
  Totals total;
  const u32 words = u32{1} << P.n;
  const u128 base = factorial(P.m - 1);
  const int d = 2 * P.t + 1;
  auto job = [&] {
    Worker W(P, use_optimal, &errors, &by_size);
    u32 c[kMaxM] = {};
    for (u32 x = next++; x < words; x = next++) {
      if (model == "dist" && std::popcount(x) < d) continue;
      c[1] = x;
      if (P.m == 2) { W.table(c, 1); continue; }
      if (model == "all") all_rec(W, c, 2, x, base, 1);
      else dist_rec(W, c, 2, x + 1, d, base);
    }
    std::lock_guard g(lock);
    total.add(W.tot, P.t);
  };
  std::vector<std::thread> pool;
  for (int i = 0; i < threads; ++i) pool.emplace_back(job);
  for (auto& th : pool) th.join();
  if (total.tables == 0) return 3;  // no such code

  for (int w = 0; w <= P.t && w <= P.n; ++w) {
    const u128 count = errors[w].size();
    for (int r = 0; r < kRules; ++r)
      emit(model, P, std::to_string(w), kRuleName[r], total.cost[w][r], total.tables * count, total.tables);
  }
  for (int r = 0; r < kRules; ++r) emit(model, P, "worst", kRuleName[r], total.worst[r], total.tables, total.tables);
  emit(model, P, "0", "oracle", total.oracle, total.tables, total.tables);
  if (use_optimal) emit(model, P, "0", "optimal", total.optimal, total.tables * static_cast<unsigned>(P.m), total.tables);
  return 0;
}
