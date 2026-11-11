// omn: exact worst-case and error-free average cost of (o, m, n) trees.
//
// o objects, any subset may be asked about, answers may be wrong. An (o, m, n)
// tree (m <= n) names the object whenever at most m answers are wrong and, when
// at most n are wrong, either names it or reports errors, never naming a wrong
// object. New in this repository (2026), see articles/omn-trees/PLAN.md.
//
// State: a[i] = number of objects that are contradicted by exactly i answers so
// far, i = 0..n (objects contradicted more than n times are dropped). A question
// puts b[i] objects of class i on the "yes" side; the children are
//   yes[i] = b[i] + (a[i-1] - b[i-1]),   no[i] = (a[i] - b[i]) + b[i-1].
// A state is terminal (a leaf is allowed) iff at most one object is left or no
// object has count <= m.
//
// W(s): least worst-case number of further questions (iterative deepening with
// memoised lower/upper bounds). Lower bounds used for pruning, both proved in
// the article: the Hamming (Berlekamp volume) bound on classes <= m,
//   sum_{i<=m} a[i] V(q, m-i) <= 2^q,  V(q,k) = sum_{j<=k} C(q,j),
// and the pair bound q >= m + n + 1 - c1 - c2 for the two smallest counts
// c1 <= c2 when c1 <= m.
//
// A(s): least sum, over objects of class 0, of the number of further questions
// on their error-free path (exact integers; the average is A / o). Exhaustive
// branch and bound with the lower bound max(L(a0), a0 * d), where L is the
// complete-tree cost and d = n + 1 - (least count among the other objects).
//
//   omn worst   <m> <n> <omax> <seconds> <out.csv>
//   omn average <m> <n> <omax> <seconds> <out.csv>
//   omn minmax  <m> <n> <omax> <seconds> <out.csv>
//   omn wone    <m> <n> <o>                    (prints the "worst" row of one o)
//   omn state   <m> <n> <a0> ... <an>          (prints W and A of one state)
//
// minmax: the least error-free total among the trees of least worst-case depth
// W, by branch and bound over the questions allowed by the Hamming window.
//
// The run over o = 1, 2, ... stops after the first o whose computation takes
// longer than <seconds> (that row is still written, with its time), or at omax.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <print>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

__extension__ typedef unsigned __int128 u128;
__extension__ typedef __int128 i128;

constexpr int kMaxClasses = 5;  // n <= 4
using State = std::array<int, kMaxClasses>;

int M = 0, N = 0;  // the game parameters

std::uint64_t key(const State& a) {
  std::uint64_t k = 0;
  for (int i = 0; i <= N; ++i) k = (k << 12) | static_cast<std::uint64_t>(a[i]);
  return k;
}

u128 binom(int q, int k) {
  if (k < 0 || k > q) return 0;
  u128 v = 1;
  for (int i = 0; i < k; ++i) v = v * static_cast<u128>(q - i) / static_cast<u128>(i + 1);
  return v;
}

u128 ball_direct(int q, int k) {  // V(q, k)
  u128 s = 0;
  for (int j = 0; j <= k; ++j) s += binom(q, j);
  return s;
}

constexpr int kMaxQ = 120;  // V(q, k) < 2^128 for q <= 120, k <= 4
const auto ball_table = [] {
  std::array<std::array<u128, kMaxClasses>, kMaxQ + 1> t{};
  for (int q = 0; q <= kMaxQ; ++q)
    for (int k = 0; k < kMaxClasses; ++k) t[q][k] = ball_direct(q, k);
  return t;
}();

u128 ball(int q, int k) { return k < 0 ? 0 : ball_table[q][k]; }

u128 pow2(int q) { return static_cast<u128>(1) << q; }

int alive(const State& a) {
  int s = 0;
  for (int i = 0; i <= N; ++i) s += a[i];
  return s;
}

bool terminal(const State& a) {
  if (alive(a) <= 1) return true;
  for (int i = 0; i <= M; ++i)
    if (a[i] > 0) return false;
  return true;
}

u128 hamming(const State& a, int q) {
  u128 s = 0;
  for (int i = 0; i <= M; ++i) s += static_cast<u128>(a[i]) * ball(q, M - i);
  return s;
}

// Two smallest counts (c1 <= c2) among the objects; -1 if absent.
std::pair<int, int> two_smallest(const State& a) {
  int c1 = -1, c2 = -1;
  for (int i = 0; i <= N && c2 < 0; ++i)
    for (int j = 0; j < a[i] && c2 < 0; ++j) (c1 < 0 ? c1 : c2) = i;
  return {c1, c2};
}

int static_lower_bound(const State& a) {
  if (terminal(a)) return 0;
  int q = 0;
  while (hamming(a, q) > pow2(q)) ++q;
  auto [c1, c2] = two_smallest(a);
  if (c2 >= 0 && c1 <= M) q = std::max(q, M + N + 1 - c1 - c2);
  // Detection bound: ceil(log2 #{objects with count <= c}) + n - c, c <= m.
  int below = 0;
  for (int c = 0; c <= M; ++c) {
    below += a[c];
    if (below >= 2) q = std::max(q, std::bit_width(static_cast<unsigned>(below - 1)) + N - c);
  }
  return q;
}

void children(const State& a, const State& b, State& y, State& z) {
  for (int i = 0; i <= N; ++i) {
    y[i] = b[i] + (i > 0 ? a[i - 1] - b[i - 1] : 0);
    z[i] = (a[i] - b[i]) + (i > 0 ? b[i - 1] : 0);
  }
  for (int i = N + 1; i < kMaxClasses; ++i) y[i] = z[i] = 0;
}

// Enumerates the questions b with b <= a - b lexicographically (a question and
// its complement give the same pair of children) and both sides non-empty.
template <class F>
void for_each_question(const State& a, F&& f) {
  State b{};
  auto rec = [&](auto&& self, int i, bool tied) -> void {
    if (i > N) {
      int s = 0;
      for (int j = 0; j <= N; ++j) s += b[j];
      if (s == 0 || s == alive(a)) return;
      f(b);
      return;
    }
    const int hi = tied ? a[i] / 2 : a[i];
    for (int v = 0; v <= hi; ++v) {
      b[i] = v;
      self(self, i + 1, tied && 2 * v == a[i]);
    }
    b[i] = 0;
  };
  rec(rec, 0, true);
}

// ---------------------------------------------------------------- worst case

struct Bounds {
  std::uint8_t lb, ub;
};
// Open-addressing hash table from state keys (never 0: terminal states are not
// stored, and a non-terminal state has an object) to bounds.
class FlatMap {
 public:
  Bounds* find(std::uint64_t k) {
    if (keys_.empty()) return nullptr;
    for (std::size_t i = slot(k);; i = (i + 1) & mask_) {
      if (keys_[i] == k) return &vals_[i];
      if (keys_[i] == 0) return nullptr;
    }
  }
  Bounds* insert(std::uint64_t k, Bounds v) {  // k must be absent
    if (2 * (size_ + 1) > keys_.size()) grow();
    std::size_t i = slot(k);
    while (keys_[i] != 0) i = (i + 1) & mask_;
    keys_[i] = k;
    vals_[i] = v;
    ++size_;
    return &vals_[i];
  }
  std::size_t size() const { return size_; }

 private:
  std::size_t slot(std::uint64_t k) const {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    return static_cast<std::size_t>(k) & mask_;
  }
  void grow() {
    std::vector<std::uint64_t> ok = std::move(keys_);
    std::vector<Bounds> ov = std::move(vals_);
    const std::size_t cap = ok.empty() ? 1024 : 2 * ok.size();
    keys_.assign(cap, 0);
    vals_.assign(cap, Bounds{});
    mask_ = cap - 1;
    size_ = 0;
    for (std::size_t i = 0; i < ok.size(); ++i)
      if (ok[i] != 0) insert(ok[i], ov[i]);
  }
  std::vector<std::uint64_t> keys_;
  std::vector<Bounds> vals_;
  std::size_t mask_ = 0, size_ = 0;
};
FlatMap wmemo;
std::uint64_t stat_searches = 0, stat_candidates = 0, stat_tried = 0;

// Questions from state a with q questions left whose two children both satisfy
// the Hamming bound with q - 1 left, with |H(yes) - H(no)| for ordering.
// H(yes) = base + sum_i d[i] b[i] must lie in [lo, hi]; the loops over b[i] are
// cut to that window, using the largest possible rest of the sum.
std::vector<std::pair<u128, State>> window_questions(const State& a, int q) {
  const i128 half = static_cast<i128>(pow2(q - 1));
  const i128 total = static_cast<i128>(hamming(a, q));
  const i128 lo = total - half, hi = half;
  std::array<i128, kMaxClasses + 1> w{}, d{}, rest{};
  for (int i = 0; i <= M; ++i) w[i] = static_cast<i128>(ball(q - 1, M - i));
  i128 base = 0;
  for (int i = 0; i <= M; ++i) base += a[i] * w[i + 1];
  for (int i = 0; i <= N; ++i) d[i] = w[i] - w[i + 1];
  for (int i = N; i >= 0; --i) rest[i] = rest[i + 1] + d[i] * a[i];
  std::vector<std::pair<u128, State>> cand;
  State bq{};
  auto rec = [&](auto&& self, int i, bool tied, i128 part) -> void {
    if (i > N) {
      if (part < lo || part > hi) return;
      int s = 0;
      for (int j = 0; j <= N; ++j) s += bq[j];
      if (s == 0 || s == alive(a)) return;
      const i128 diff = 2 * part - total;
      cand.emplace_back(static_cast<u128>(diff < 0 ? -diff : diff), bq);
      return;
    }
    int from = 0, to = tied ? a[i] / 2 : a[i];
    if (d[i] > 0) {
      const i128 need = lo - part - rest[i + 1];
      if (need > 0) from = std::max<int>(from, static_cast<int>((need + d[i] - 1) / d[i]));
      const i128 room = hi - part;
      if (room < 0) return;
      to = std::min<int>(to, static_cast<int>(std::min<i128>(room / d[i], a[i])));
    } else if (part > hi || part + rest[i + 1] < lo) {
      return;
    }
    for (int v = from; v <= to; ++v) {
      bq[i] = v;
      self(self, i + 1, tied && 2 * v == a[i], part + d[i] * v);
    }
    bq[i] = 0;
  };
  rec(rec, 0, true, base);
  return cand;
}

// True if the memo or the static bound already shows that q questions do not suffice.
bool known_infeasible(const State& a, int q) {
  if (terminal(a)) return false;
  if (const Bounds* e = wmemo.find(key(a))) return q < e->lb;
  return q < static_lower_bound(a);
}

bool feasible(const State& a, int q) {
  if (terminal(a)) return true;
  const auto k = key(a);
  const Bounds* e0 = wmemo.find(k);
  if (!e0) e0 = wmemo.insert(k, Bounds{static_cast<std::uint8_t>(static_lower_bound(a)), 255});
  if (q >= e0->ub) return true;
  if (q < e0->lb) return false;
  auto cand = window_questions(a, q);
  ++stat_searches;
  stat_candidates += cand.size();
  std::stable_sort(cand.begin(), cand.end(),
                   [](const auto& x, const auto& y) { return x.first < y.first; });
  for (const auto& [gap, b] : cand) {
    State y, z;
    children(a, b, y, z);
    if (known_infeasible(y, q - 1) || known_infeasible(z, q - 1)) continue;
    ++stat_tried;
    if (feasible(y, q - 1) && feasible(z, q - 1)) {
      Bounds& e = *wmemo.find(k);
      e.ub = static_cast<std::uint8_t>(std::min<int>(e.ub, q));
      return true;
    }
  }
  Bounds& e = *wmemo.find(k);
  e.lb = static_cast<std::uint8_t>(std::max<int>(e.lb, q + 1));
  return false;
}

int worst(const State& a) {
  int q = static_lower_bound(a);
  while (!feasible(a, q)) ++q;
  return q;
}

// --------------------------------------------------------- error-free average

std::unordered_map<std::uint64_t, std::uint32_t> amemo;

std::uint64_t complete_tree_cost(std::uint64_t k) {  // least sum of leaf depths
  if (k <= 1) return 0;
  std::uint64_t d = 0;
  while ((std::uint64_t{1} << (d + 1)) <= k) ++d;
  return k * d + 2 * (k - (std::uint64_t{1} << d));
}

std::uint64_t average_lower_bound(const State& a) {
  if (terminal(a) || a[0] == 0) return 0;
  // least count among the objects other than one object of class 0
  int c = -1;
  if (a[0] >= 2) c = 0;
  else
    for (int i = 1; i <= N && c < 0; ++i)
      if (a[i] > 0) c = i;
  const std::uint64_t d = static_cast<std::uint64_t>(N + 1 - c);
  const std::uint64_t a0 = static_cast<std::uint64_t>(a[0]);
  return std::max(complete_tree_cost(a0), a0 * d);
}

std::uint64_t average(const State& a) {
  if (terminal(a) || a[0] == 0) return 0;
  const auto k = key(a);
  if (auto it = amemo.find(k); it != amemo.end()) return it->second;
  // Heuristic order: most even split of class 0, then of the whole state.
  std::vector<std::pair<std::uint64_t, State>> cand;
  for_each_question(a, [&](const State& b) {
    State y, z;
    children(a, b, y, z);
    const std::uint64_t lb = average_lower_bound(y) + average_lower_bound(z);
    cand.emplace_back(lb, b);
  });
  std::stable_sort(cand.begin(), cand.end(),
                   [](const auto& x, const auto& y) { return x.first < y.first; });
  std::uint64_t best = UINT64_MAX;
  const std::uint64_t a0 = static_cast<std::uint64_t>(a[0]);
  for (const auto& [lb, b] : cand) {
    if (best != UINT64_MAX && a0 + lb >= best) break;  // sorted by lb
    State y, z;
    children(a, b, y, z);
    const std::uint64_t vy = average(y);
    if (best != UINT64_MAX && a0 + vy + average_lower_bound(z) >= best) continue;
    const std::uint64_t v = a0 + vy + average(z);
    best = std::min(best, v);
  }
  amemo.emplace(k, static_cast<std::uint32_t>(best));
  return best;
}

// Least error-free total among the trees of depth <= q from state a, or
// UINT64_MAX if there is none. Branch and bound with the same lower bound.
std::unordered_map<std::uint64_t, std::uint64_t> bmemo;

std::uint64_t bounded_average(const State& a, int q) {
  if (terminal(a)) return 0;
  if (!feasible(a, q)) return UINT64_MAX;
  if (a[0] == 0) return 0;
  const std::uint64_t k = key(a) * 128 + static_cast<std::uint64_t>(q);
  if (auto it = bmemo.find(k); it != bmemo.end()) return it->second;
  std::vector<std::pair<std::uint64_t, State>> cand;
  for (const auto& [gap, b] : window_questions(a, q)) {
    State y, z;
    children(a, b, y, z);
    if (known_infeasible(y, q - 1) || known_infeasible(z, q - 1)) continue;
    cand.emplace_back(average_lower_bound(y) + average_lower_bound(z), b);
  }
  std::stable_sort(cand.begin(), cand.end(),
                   [](const auto& x, const auto& y) { return x.first < y.first; });
  std::uint64_t best = UINT64_MAX;
  const std::uint64_t a0 = static_cast<std::uint64_t>(a[0]);
  for (const auto& [lb, b] : cand) {
    if (best != UINT64_MAX && a0 + lb >= best) break;
    State y, z;
    children(a, b, y, z);
    const std::uint64_t vy = bounded_average(y, q - 1);
    if (vy == UINT64_MAX) continue;
    if (best != UINT64_MAX && a0 + vy + average_lower_bound(z) >= best) continue;
    const std::uint64_t vz = bounded_average(z, q - 1);
    if (vz == UINT64_MAX) continue;
    best = std::min(best, a0 + vy + vz);
  }
  bmemo.emplace(k, best);
  return best;
}

State root(int o) {
  State a{};
  a[0] = o;
  return a;
}

int run(const std::string& mode, int omax, double limit, const std::string& out) {
  std::ofstream f(out);
  if (mode == "worst") f << "o,m,n,W,hamming,pair,seconds\n";
  else if (mode == "minmax") f << "o,m,n,W,total,seconds\n";
  else f << "o,m,n,total,seconds\n";
  for (int o = 1; o <= omax; ++o) {
    const auto t0 = std::chrono::steady_clock::now();
    const State a = root(o);
    std::string row;
    if (mode == "worst") {
      const int w = worst(a);
      int h = 0;
      while (hamming(a, h) > pow2(h)) ++h;
      const int p = o >= 2 ? M + N + 1 : 0;
      row = std::format("{},{},{},{},{},{}", o, M, N, w, h, p);
    } else if (mode == "minmax") {
      const int w = worst(a);
      row = std::format("{},{},{},{},{}", o, M, N, w, bounded_average(a, w));
    } else {
      row = std::format("{},{},{},{}", o, M, N, average(a));
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    f << row << std::format(",{:.3f}\n", s);
    if (std::getenv("OMN_STATS"))
      std::println(stderr, "o={} states={} searches={} candidates={} tried={}", o, wmemo.size(),
                   stat_searches, stat_candidates, stat_tried);
    f.flush();
    if (s > limit) {
      std::println(stderr, "{} m={} n={}: stopped after o={} ({:.1f} s)", mode, M, N, o, s);
      break;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::println(stderr, "usage: omn worst|average <m> <n> <omax> <seconds> <out.csv>\n"
                         "       omn state <m> <n> <a0> ... <an>");
    return 2;
  }
  const std::string mode = argv[1];
  M = std::atoi(argv[2]);
  N = std::atoi(argv[3]);
  if (M < 0 || N < M || N >= kMaxClasses || (mode == "minmax" && N > 3)) {
    std::println(stderr, "need 0 <= m <= n <= {}", kMaxClasses - 1);
    return 2;
  }
  if (mode == "wone") {  // one root, worst case only, row as in "worst" without header
    const int o = std::atoi(argv[4]);
    const auto t0 = std::chrono::steady_clock::now();
    const State a = root(o);
    const int w = worst(a);
    int h = 0;
    while (hamming(a, h) > pow2(h)) ++h;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::println("{},{},{},{},{},{},{:.3f}", o, M, N, w, h, o >= 2 ? M + N + 1 : 0, sec);
    return 0;
  }
  if (mode == "state") {
    State a{};
    for (int i = 0; i <= N && 4 + i < argc; ++i) a[i] = std::atoi(argv[4 + i]);
    std::println("W={} A={}", worst(a), average(a));
    return 0;
  }
  if (argc < 7) return 2;
  return run(mode, std::atoi(argv[4]), std::atof(argv[5]), argv[6]);
}
