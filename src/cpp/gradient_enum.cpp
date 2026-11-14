// gradient_enum: exact enumeration of all n x m binary matrices for Step 9 of the
// analytic estimate of the gradient key (codes: articles/gradient-bound/analytic/).
//
// Under the uniform measure on all n x m matrices (iid fair bits; columns need not be
// distinct) the gradient key G is run with the rule of Step 1: at a set S with |S| >= 2
// the first row of largest quality min(k, |S| - k) is tested if that quality is >= 1,
// otherwise the construction stops at S. Follow the path S_0 = [m] > S_1 > ... of
// column 0 ("column 1" in the text). A class is (i; s_0, ..., s_i; r_0, ..., r_{i-1}):
// the depth, the sizes and the rows chosen on the path. For every class with s_i >= 2
// the engine prints the number of matrices in it (W), the histogram of the largest
// quality at S_i, and for every row r not among r_0..r_{i-1} the histograms of its
// qualities q_r(S_i) = min(k_r(S_i), s_i - k_r(S_i)) and q_r(S_{i-1}). Nothing is sampled; counts are exact 128-bit integers.
//
//   gradient_enum brute <n> <m> <threads>   all 2^(nm) matrices
//   gradient_enum sym <n> <m> <threads>     the same numbers, by symmetry: each row is
//        complemented so that column 0 is 0 (factor 2^n), and columns 1..m-1 are taken
//        as a multiset with multinomial weight (G and every printed quantity are
//        invariant under both operations)
//
// Output lines:
//   C i s_0 .. s_i ; r_0 .. r_{i-1} ; W ; q-hist (0..s_i/2)
//   R r ; q_r(S_i) hist (0..s_i/2) ; q_r(S_{i-1}) hist (0..s_{i-1}/2, empty if i = 0)
// Classes are printed in lexicographic order of (i, sizes, rows).
//
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <print>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using u64 = std::uint64_t;
using u128 = unsigned __int128;

std::string str(u128 v) {
  if (v == 0) return "0";
  std::string s;
  while (v) {
    s.push_back(static_cast<char>('0' + static_cast<int>(v % 10)));
    v /= 10;
  }
  std::reverse(s.begin(), s.end());
  return s;
}

struct Node {
  int depth = 0, size = 0, parent_size = 0;
  std::vector<int> key;  // sizes then rows, as printed
  std::vector<bool> chosen;
  u128 w = 0;
  std::vector<u128> qh;                // size/2 + 1
  std::vector<std::vector<u128>> kh;   // [row][quality on S_i, 0..size/2]
  std::vector<std::vector<u128>> kp;   // [row][quality on S_{i-1}, 0..parent_size/2]
};

struct Tally {
  int n = 0, m = 0;
  std::vector<Node> nodes;
  std::unordered_map<u64, int> child;  // (node, row, next size) -> node

  explicit Tally(int n_, int m_) : n(n_), m(m_) {
    Node root;
    root.size = m;
    root.key = {m};
    root.chosen.assign(static_cast<std::size_t>(n), false);
    init(root);
    nodes.push_back(std::move(root));
  }

  void init(Node& nd) const {
    nd.qh.assign(static_cast<std::size_t>(nd.size / 2 + 1), 0);
    nd.kh.assign(static_cast<std::size_t>(n), std::vector<u128>(static_cast<std::size_t>(nd.size / 2) + 1, 0));
    nd.kp.assign(static_cast<std::size_t>(n), std::vector<u128>(static_cast<std::size_t>(nd.parent_size / 2) + 1, 0));
  }

  int get_child(int id, int row, int next) {
    const u64 key = (static_cast<u64>(id) << 16) | (static_cast<u64>(row) << 8) | static_cast<u64>(next);
    auto it = child.find(key);
    if (it != child.end()) return it->second;
    Node nd;
    const Node& par = nodes[static_cast<std::size_t>(id)];
    nd.depth = par.depth + 1;
    nd.size = next;
    nd.parent_size = par.size;
    const int d = par.depth;
    nd.key.assign(par.key.begin(), par.key.begin() + d + 1);
    nd.key.push_back(next);
    nd.key.insert(nd.key.end(), par.key.begin() + d + 1, par.key.end());
    nd.key.push_back(row);
    nd.chosen = par.chosen;
    nd.chosen[static_cast<std::size_t>(row)] = true;
    init(nd);
    nodes.push_back(std::move(nd));
    const int nid = static_cast<int>(nodes.size()) - 1;
    child.emplace(key, nid);
    return nid;
  }

  // One matrix (rows as bitmasks over the m columns), with weight w.
  void add(const std::vector<u64>& row, u128 w) {
    u64 S = (m == 64) ? ~u64{0} : (u64{1} << m) - 1;
    int id = 0;
    u64 prev = 0;
    int k[64];
    for (;;) {
      Node* nd = &nodes[static_cast<std::size_t>(id)];
      const int s = nd->size;
      if (s < 2) return;
      int best = -1, bestq = 0;
      for (int r = 0; r < n; ++r) {
        k[r] = std::popcount(row[static_cast<std::size_t>(r)] & S);
        const int q = std::min(k[r], s - k[r]);
        if (q > bestq) bestq = q, best = r;
      }
      nd->w += w;
      nd->qh[static_cast<std::size_t>(bestq)] += w;
      for (int r = 0; r < n; ++r) {
        if (nd->chosen[static_cast<std::size_t>(r)]) continue;
        nd->kh[static_cast<std::size_t>(r)][static_cast<std::size_t>(std::min(k[r], s - k[r]))] += w;
        if (nd->depth > 0) {
          const int kq = std::popcount(row[static_cast<std::size_t>(r)] & prev);
          nd->kp[static_cast<std::size_t>(r)][static_cast<std::size_t>(std::min(kq, nd->parent_size - kq))] += w;
        }
      }
      if (best < 0) return;
      const u64 rb = row[static_cast<std::size_t>(best)];
      const u64 next = (rb & 1) ? (S & rb) : (S & ~rb);
      prev = S;
      S = next;
      id = get_child(id, best, std::popcount(S));
    }
  }
};

void merge(std::map<std::vector<int>, Node>& out, Tally& t) {
  for (Node& nd : t.nodes) {
    if (nd.w == 0) continue;
    auto [it, fresh] = out.try_emplace(nd.key);
    if (fresh) {
      it->second = std::move(nd);
      continue;
    }
    Node& o = it->second;
    o.w += nd.w;
    for (std::size_t i = 0; i < o.qh.size(); ++i) o.qh[i] += nd.qh[i];
    for (std::size_t r = 0; r < o.kh.size(); ++r) {
      for (std::size_t i = 0; i < o.kh[r].size(); ++i) o.kh[r][i] += nd.kh[r][i];
      for (std::size_t i = 0; i < o.kp[r].size(); ++i) o.kp[r][i] += nd.kp[r][i];
    }
  }
}

void print(const std::map<std::vector<int>, Node>& out, int n) {
  std::vector<std::pair<std::vector<int>, const Node*>> v;
  for (auto& [k, nd] : out) v.emplace_back(k, &nd);
  std::sort(v.begin(), v.end(), [](auto& a, auto& b) {
    if (a.second->depth != b.second->depth) return a.second->depth < b.second->depth;
    return a.first < b.first;
  });
  for (auto& [key, nd] : v) {
    std::string line = "C " + std::to_string(nd->depth);
    for (int j = 0; j <= nd->depth; ++j) line += " " + std::to_string(key[static_cast<std::size_t>(j)]);
    line += " ;";
    for (int j = nd->depth + 1; j < static_cast<int>(key.size()); ++j) line += " " + std::to_string(key[static_cast<std::size_t>(j)]);
    line += " ; " + str(nd->w) + " ;";
    for (u128 x : nd->qh) line += " " + str(x);
    std::println("{}", line);
    for (int r = 0; r < n; ++r) {
      if (nd->chosen[static_cast<std::size_t>(r)]) continue;
      std::string l = "R " + std::to_string(r) + " ;";
      for (u128 x : nd->kh[static_cast<std::size_t>(r)]) l += " " + str(x);
      l += " ;";
      if (nd->depth > 0)
        for (u128 x : nd->kp[static_cast<std::size_t>(r)]) l += " " + str(x);
      std::println("{}", l);
    }
  }
}

// Columns as words: bit r of word c is row r. Builds row masks.
void set_column(std::vector<u64>& row, int n, int col, u64 word) {
  for (int r = 0; r < n; ++r) {
    const u64 bit = u64{1} << col;
    if ((word >> r) & 1) row[static_cast<std::size_t>(r)] |= bit;
    else row[static_cast<std::size_t>(r)] &= ~bit;
  }
}

int run(bool sym, int n, int m, int threads) {
  if (n < 1 || n > 20 || m < 2 || m > 40 || threads < 1) {
    std::println(stderr, "need 1 <= n <= 20, 2 <= m <= 40, threads >= 1");
    return 2;
  }
  const u64 words = u64{1} << n;
  // Tasks: the first one or two free columns.
  std::vector<std::pair<u64, u64>> tasks;
  if (sym) {
    if (m >= 3)
      for (u64 a = 0; a < words; ++a)
        for (u64 b = a; b < words; ++b) tasks.emplace_back(a, b);
    else
      for (u64 a = 0; a < words; ++a) tasks.emplace_back(a, 0);
  } else {
    for (u64 a = 0; a < words; ++a)
      for (u64 b = 0; b < words; ++b) tasks.emplace_back(a, b);
  }
  u128 fact = 1;
  for (int i = 2; i <= m - 1; ++i) fact *= static_cast<u128>(i);
  const u128 scale = sym ? (u128{1} << n) : 1;
  std::atomic<std::size_t> next{0};
  std::vector<Tally> tallies;
  tallies.reserve(static_cast<std::size_t>(threads));
  for (int t = 0; t < threads; ++t) tallies.emplace_back(n, m);
  {
    std::vector<std::jthread> pool;
    for (int t = 0; t < threads; ++t)
      pool.emplace_back([&, t] {
        Tally& T = tallies[static_cast<std::size_t>(t)];
        std::vector<u64> row(static_cast<std::size_t>(n), 0);
        std::vector<u64> col(static_cast<std::size_t>(m), 0);
        for (std::size_t ti = next++; ti < tasks.size(); ti = next++) {
          const auto [a, b] = tasks[ti];
          if (sym) {
            // col[0] = 0; col[1..m-1] nondecreasing; weight fact / prod(mult!) * 2^n.
            std::fill(row.begin(), row.end(), 0);
            col[1] = a;
            set_column(row, n, 1, a);
            int first = 2;
            u128 den = 1;
            int run_len = 1;
            if (m >= 3) {
              col[2] = b;
              set_column(row, n, 2, b);
              if (b == a) run_len = 2, den = 2;
              first = 3;
            }
            auto rec = [&](auto&& self, int c, u128 d, int rl) -> void {
              if (c == m) {
                T.add(row, fact / d * scale);
                return;
              }
              for (u64 x = col[static_cast<std::size_t>(c) - 1]; x < words; ++x) {
                col[static_cast<std::size_t>(c)] = x;
                set_column(row, n, c, x);
                if (x == col[static_cast<std::size_t>(c) - 1]) self(self, c + 1, d * static_cast<u128>(rl + 1), rl + 1);
                else self(self, c + 1, d, 1);
              }
            };
            rec(rec, first, den, run_len);
          } else {
            // column 0 = a, column 1 = b, the rest free.
            std::fill(row.begin(), row.end(), 0);
            set_column(row, n, 0, a);
            set_column(row, n, 1, b);
            auto rec = [&](auto&& self, int c) -> void {
              if (c == m) {
                T.add(row, 1);
                return;
              }
              for (u64 x = 0; x < words; ++x) {
                set_column(row, n, c, x);
                self(self, c + 1);
              }
            };
            rec(rec, 2);
          }
        }
      });
  }
  std::map<std::vector<int>, Node> out;
  for (Tally& t : tallies) merge(out, t);
  print(out, n);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if ((cmd == "brute" || cmd == "sym") && argc == 5)
    return run(cmd == "sym", std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]));
  std::println(stderr, "usage: gradient_enum brute|sym n m threads");
  return 2;
}
