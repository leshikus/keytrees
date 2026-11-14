// gradient_tplus: the table function t^+_{n,m} of articles/gradient-bound, an
// upper bound on the average cost t_{n,m}(G) of the gradient key.
//
// t^+_{n,m} is the least of the bounds the article proves for the cell:
//   exact     the exact value of data/gradient-bound/exact_formula.csv
//   trivial   min(n, C_max(m)): on D a path never repeats a row
//   fixed     Theorem fixed-m: C(m) + (C_max - C) sum_{s>=4} C(m,s)(1-beta_s)^n / Q(D)
//   union     Proposition union, minimised over the finite family of
//             parameters below
//   explicit  Corollary explicit, K in {20, 30, 40, 60}, where its hypotheses hold
// with t = 0 for m = 1. The cell must have m <= 2^n.
//
// Parameters of "union": a requirement kappa_j(s), for s <= s_1, is the largest
// k <= floor(s/2) with C(m,s) f_k(s)^n <= 2^-j / m (1 if there is none), for
// j in TAUS; s_1 = m with x = 0.4 if m <= S1MAX (no fork is large), and
// s_1 in S1S with s_1 < m, x in XS with x < s_1/4.
//
// Rigour: the whole computation runs in the upward rounding mode
// (fesetround(FE_UPWARD), with FENV_ACCESS on), so +, -, *, /, sqrt of
// upper bounds give upper bounds; lower bounds are formed as -up(-x). log,
// log1p and exp are not correctly rounded, so each result is widened by
// MARGIN ulp-sized relative steps plus the least subnormal. lgamma is not
// used: binomials are log-domain running sums of log of integers. Every
// printed number is written with %.17g in the upward mode, so the decimal
// string is at least the double. Discarding an ingredient (as when a sum is
// already >= 1) never invalidates the minimum.
//
// Subcommands:
//   cell <n> <m> [--no-exact]   all ingredients of one cell, readable
//   cells [--no-exact]          "n m" pairs on stdin, CSV rows on stdout
//   grid [<out.csv>]            the grid of the article (default
//                               data/gradient-bound/tplus.csv)
// Option --data <dir> (default data/gradient-bound) locates
// exact_formula.csv and gradient_constants.csv.
//
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#pragma STDC FENV_ACCESS ON

namespace {

using u64 = std::uint64_t;
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr int S1MAX = 2048;
constexpr int TAUS[] = {2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40};
constexpr int S1S[] = {16, 32, 64, 128, 256, 512, 1024, 2048};
constexpr double XS[] = {0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8};
constexpr double X_SMALL = 0.4;  // x when s_1 = m; it plays no role then
constexpr double MARGIN = 16 * DBL_EPSILON;
constexpr long SCANCAP = 1L << 24;  // largest s scanned in the fixed-m sum

// ---------------------------------------------------------------- directed rounding

double neg(double a) { return -a; }
double dn_add(double a, double b) { return -((-a) - b); }
double dn_sub(double a, double b) { return -(b - a); }
double dn_mul(double a, double b) { return -((-a) * b); }
double dn_div(double a, double b) { return -((-a) / b); }

double widen_up(double r) { return r + (std::fabs(r) * MARGIN + DBL_TRUE_MIN); }
double widen_dn(double r) { return -((-r) + (std::fabs(r) * MARGIN + DBL_TRUE_MIN)); }

double log_up(double x) { return widen_up(std::log(x)); }
double log_dn(double x) { return widen_dn(std::log(x)); }
double log1p_up(double x) { return widen_up(std::log1p(x)); }
double log1p_dn(double x) { return widen_dn(std::log1p(x)); }
double exp_up(double x) {
  if (x == -INF) return 0;
  double r = std::exp(x);
  return r * (1 + MARGIN) + DBL_TRUE_MIN;
}
double exp_dn(double x) {
  if (x == -INF) return 0;
  double r = std::exp(x);
  return std::max(0.0, dn_sub(dn_mul(r, dn_sub(1, MARGIN)), DBL_TRUE_MIN));
}

double LN2_UP, LN2_DN;

// upper bound of log(e^a + e^b) from upper bounds a, b
double lse_up(double a, double b) {
  if (a == -INF) return b;
  if (b == -INF) return a;
  double hi = std::max(a, b), lo = std::min(a, b);
  return hi + log1p_up(exp_up(lo - hi));
}

bool is_pow2(u64 m) { return m && !(m & (m - 1)); }
int floor_log2(u64 m) { return 63 - __builtin_clzll(m); }
double log2_up(u64 m) { return is_pow2(m) ? floor_log2(m) : log_up(double(m)) / LN2_DN; }
double log2_dn(u64 m) { return is_pow2(m) ? floor_log2(m) : dn_div(log_dn(double(m)), LN2_UP); }

std::string fmt(double x) {  // the decimal string is >= x (glibc honours the rounding mode)
  if (x == INF) return "inf";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.17g", x);
  return buf;
}

// ---------------------------------------------------------------- tables for s <= S1MAX

std::vector<double> lnU, lnD, lfU, lfD;  // log i and log i!, up and down
std::vector<std::vector<double>> LOGF;   // LOGF[s][k] >= log f_k(s), 1 <= k <= s/2
std::vector<std::vector<double>> OMH;    // OMH[s][k] >= 1 - h(k/s)

void build_tables() {
  int N = S1MAX + 1;
  lnU.assign(N, 0); lnD.assign(N, 0); lfU.assign(N, 0); lfD.assign(N, 0);
  for (int i = 2; i < N; ++i) {
    lnU[i] = log_up(i); lnD[i] = log_dn(i);
    lfU[i] = lfU[i - 1] + lnU[i];
    lfD[i] = dn_add(lfD[i - 1], lnD[i]);
  }
  LOGF.assign(N, {}); OMH.assign(N, {});
  for (int s = 2; s < N; ++s) {
    int half = s / 2;
    LOGF[s].assign(half + 1, 0);
    OMH[s].assign(half + 1, 0);
    double acc = 0;  // log sum_{j<k} C(s,j), starting with k = 1
    double base = double(1 - s) * LN2_DN;  // >= (1 - s) ln 2, as 1 - s < 0
    for (int k = 1; k <= half; ++k) {
      if (k > 1) acc = lse_up(acc, lfU[s] - lfD[k - 1] - lfD[s - k + 1]);
      LOGF[s][k] = base + acc;
      // h(k/s) = (s ln s - k ln k - (s-k) ln(s-k)) / (s ln 2), bounded below
      double a = double(k) * lnU[k] + double(s - k) * lnU[s - k];
      double h = dn_div(dn_sub(dn_mul(double(s), lnD[s]), a), double(s) * LN2_UP);
      OMH[s][k] = 1 - h;
    }
  }
}

// ---------------------------------------------------------------- data files

std::map<std::pair<long, u64>, std::pair<std::string, std::string>> EXACT;
std::map<int, double> CK;  // c_K upper bounds

double parse_dir(const std::string& s, int mode) {
  std::fesetround(mode);
  double v = std::strtod(s.c_str(), nullptr);
  std::fesetround(FE_UPWARD);
  return v;
}

std::vector<std::vector<std::string>> read_csv(const std::string& path) {
  std::ifstream f(path);
  std::vector<std::vector<std::string>> rows;
  std::string line;
  if (!f) return rows;
  std::getline(f, line);
  while (std::getline(f, line)) {
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, ',')) cells.push_back(c);
    rows.push_back(cells);
  }
  return rows;
}

void read_data(const std::string& dir) {
  auto ex = read_csv(dir + "/exact_formula.csv");
  if (ex.empty()) std::cerr << "warning: no " << dir << "/exact_formula.csv\n";
  for (auto& r : ex) EXACT[{std::stol(r[0]), std::stoull(r[1])}] = {r[2], r[3]};
  // columns K,s1,x,max_rho_upper,argmax,cylinder_tail_upper,c_K_upper, 8 decimals
  // rounded to nearest: add 1e-8
  for (auto& r : read_csv(dir + "/gradient_constants.csv")) CK[std::stoi(r[0])] = parse_dir(r[6], FE_UPWARD) + 1e-8;
}

// ---------------------------------------------------------------- one cell

struct Ingredient {
  double value = INF;  // INF: not applicable or no better than C_max
  std::string params;
};

struct Cell {
  long n;
  u64 m;
  double log2m_up, log2m_dn, C_up, C_dn, Cmax_up, Rmax_up, invQD_up;
  Ingredient exact, trivial, fixed, uni, expl;
  double tplus;
  std::string winner, params;
};

// upper bound of 1/Q(D), Q(D) = prod_{i<m} (1 - i 2^-n)
double inv_qd_up(long n, u64 m) {
  if (m <= 1) return 1;
  // -ln Q(D) <= sum x_i/(1-x_i) <= m(m-1) 2^-(n+1) / (1 - (m-1) 2^-n)
  double u;
  double mm = double(m) * double(m - 1);  // rounded up
  if (n > 900) {
    u = std::ldexp(1.0, -800);  // m <= 2^32, so the bound is below 2^(65-n) * 2
  } else {
    double num = std::ldexp(mm, -int(n + 1));
    double den = dn_sub(1.0, std::ldexp(double(m - 1), -int(n)));
    u = num / den;
    if (m <= (u64(1) << 20) && u > 1e-6) {  // direct sum, tighter when m is near 2^n
      double v = 0;
      for (u64 i = 1; i < m; ++i) v += -log1p_dn(-std::ldexp(double(i), -int(n)));
      u = std::min(u, v);
    }
  }
  return exp_up(u);
}

void ingredient_fixed(Cell& c) {
  u64 m = c.m;
  long n = c.n;
  if (m <= 3) {
    c.fixed = {c.C_up, "sum=0"};
    return;
  }
  if (c.invQD_up == INF) return;
  // scan s = 4..m: log C(m,s) by running sums, beta_s from a_k = C(2k,k) 4^-k
  double lbin = 0;   // >= log C(m, s)
  double lak = 0;    // <= log a_k
  double lsum = -INF;
  for (long s = 1; s <= long(m); ++s) {
    if (s > SCANCAP) return;
    lbin = lbin + log_up(double(m - s + 1)) - (s <= S1MAX ? lnD[s] : log_dn(double(s)));
    if (s % 2 == 0) lak = dn_add(lak, log1p_dn(-(1.0 / double(s))));  // a_k = a_{k-1}(2k-1)/(2k), s = 2k
    if (s < 4) continue;
    double l1mb;  // >= log(1 - beta_s)
    if (s <= S1MAX) {
      l1mb = LOGF[s][s / 2];
    } else {
      double lbeta = lak;  // even s
      if (s % 2) lbeta = dn_add(lak, log_dn(dn_div(double(s), double(s / 2 + 1))));  // a_k (2k+1)/(k+1)
      l1mb = log1p_up(-exp_dn(lbeta));
    }
    lsum = lse_up(lsum, lbin + double(n) * l1mb);
    if (lsum > 0) return;  // sum >= 1: no better than C_max
  }
  double v = c.C_up + (c.Cmax_up - c.C_dn) * exp_up(lsum) * c.invQD_up;
  c.fixed = {v, "sum=" + fmt(exp_up(lsum))};
}

void ingredient_union(Cell& c) {
  u64 m = c.m;
  long n = c.n;
  if (m < 2 || c.invQD_up == INF) return;
  int S = int(std::min<u64>(m, S1MAX));
  // log C(m,s) for s <= S
  std::vector<double> Lm(S + 1, 0);
  double num = 0;
  for (int s = 1; s <= S; ++s) {
    num += log_up(double(m - s + 1));
    Lm[s] = num - lfD[s];
  }
  double lnm_up = log_up(double(m));
  Ingredient best;
  std::vector<int> kap(S + 1, 1);
  std::vector<double> rho(S + 1, 0), pm(S + 1, 0), pc(S + 1, 0);
  for (int j : TAUS) {
    double thr = dn_sub(dn_mul(-double(j), LN2_UP), lnm_up);  // <= log(2^-j / m)
    for (int s = 2; s <= S; ++s) {
      int lo = 1, hi = s / 2;  // largest k with Lm + n LOGF <= thr; LOGF increases in k
      if (Lm[s] + double(n) * LOGF[s][1] > thr) { kap[s] = 1; continue; }
      while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (Lm[s] + double(n) * LOGF[s][mid] <= thr) lo = mid; else hi = mid - 1;
      }
      kap[s] = lo;
    }
    std::fill(rho.begin(), rho.end(), 0.0);
    std::fill(pm.begin(), pm.end(), 0.0);
    std::fill(pc.begin(), pc.end(), 0.0);
    for (int s = 2; s <= S; ++s) {
      double r = 0;
      for (int k = kap[s]; k <= s / 2; ++k) {
        double v = OMH[s][k] + (double(k) / s) * rho[k] + (double(s - k) / s) * rho[s - k];
        r = std::max(r, v);
      }
      rho[s] = r;
      pm[s] = std::max(pm[s - 1], r);
      if (s >= 4) pc[s] = pc[s - 1] + (kap[s] >= 2 ? exp_up(Lm[s] + double(n) * LOGF[s][kap[s]]) : 0.0);
    }
    auto consider = [&](int s1, double x, double tail, double pcyl, std::string what) {
      double v = c.log2m_up + pm[s1] + tail + c.Rmax_up * c.invQD_up * (pc[s1] + pcyl);
      if (v < best.value) best = {v, "j=" + std::to_string(j) + " s1=" + std::to_string(s1) + " x=" + fmt(x) + what};
    };
    if (m <= u64(S1MAX)) consider(int(m), X_SMALL, 0.0, 0.0, " (s1=m)");
    for (int s1 : S1S) {
      if (u64(s1) >= m) continue;
      for (double x : XS) {
        if (x >= s1 / 4.0) continue;
        double theta = 0.5 + std::sqrt(x / s1);
        if (theta >= 1) continue;
        double omt = dn_sub(1.0, theta);
        double tail = 4 * x / dn_mul(double(s1), omt);
        // d* <= floor(log(m/s1) / log(1/theta))
        double ratio = (lnm_up - lnD[s1]) / neg(log_up(theta));
        double dstar = std::floor(ratio);
        long dmax = long(std::min(dstar, double(n - 1)));
        double l2n = log_up(2.0 * double(n));
        double lg = LN2_UP - 2 * x;  // log g, g = 2 e^{-2x}
        double ls = -INF;
        for (long d = 0; d <= dmax; ++d) ls = lse_up(ls, double(d) * l2n + double(n - d) * lg);
        consider(s1, x, tail, exp_up(ls), "");
      }
    }
  }
  c.uni = best;
}

void ingredient_explicit(Cell& c) {
  u64 m = c.m;
  long n = c.n;
  if (m < 2 || c.invQD_up == INF) return;
  const double c061 = std::nextafter(0.61, 1.0), c026 = std::nextafter(0.026, 1.0);
  double side = (2 + c061 * (1 + log2_up(u64(n)))) * c.log2m_up + 6;
  if (double(n) < side) return;
  Ingredient best;
  for (auto [K, cK] : CK) {
    if (double(n) < double(K) * c.log2m_up) continue;
    double err = c026 / dn_mul(double(m), double(m));  // >= 0.026 m^-2
    if (m > 500) err += std::ldexp(1.0, -10);
    double v = c.log2m_up + cK + err * c.invQD_up;
    if (v < best.value) best = {v, "K=" + std::to_string(K)};
  }
  c.expl = best;
}

Cell evaluate(long n, u64 m, bool use_exact) {
  Cell c{};
  c.n = n;
  c.m = m;
  if (m == 1) {
    c.tplus = 0;
    c.winner = "m=1";
    return c;
  }
  int l = floor_log2(m);
  c.log2m_up = log2_up(m);
  c.log2m_dn = log2_dn(m);
  double p = std::ldexp(1.0, l + 1);
  c.C_up = double(l + 2) - dn_div(p, double(m));
  c.C_dn = dn_sub(double(l + 2), p / double(m));
  c.Cmax_up = double(m + 1) / 2 - dn_div(1.0, double(m));
  c.Rmax_up = c.Cmax_up - c.log2m_dn;
  c.invQD_up = inv_qd_up(n, m);

  if (use_exact) {
    auto it = EXACT.find({n, m});
    if (it != EXACT.end()) {
      double v = parse_dir(it->second.first, FE_UPWARD) / parse_dir(it->second.second, FE_DOWNWARD);
      c.exact = {v, ""};
    }
  }
  c.trivial = {std::min(double(n), c.Cmax_up), double(n) <= c.Cmax_up ? "n" : "Cmax"};
  ingredient_fixed(c);
  ingredient_union(c);
  ingredient_explicit(c);

  std::pair<const char*, Ingredient*> all[] = {
      {"exact", &c.exact}, {"fixed", &c.fixed}, {"union", &c.uni}, {"explicit", &c.expl}, {"trivial", &c.trivial}};
  c.tplus = INF;
  for (auto [name, ing] : all) {
    if (ing->value < c.tplus) {
      c.tplus = ing->value;
      c.winner = name;
      c.params = ing->params;
    }
  }
  return c;
}

std::string csv_header() {
  return "n,m,tplus,tplus_minus_log2m,ingredient,params,exact,trivial,fixed,union,union_params,explicit,explicit_params";
}

std::string opt(const Ingredient& i) { return i.value == INF ? "" : fmt(i.value); }

std::string csv_row(const Cell& c) {
  std::string r = std::to_string(c.n) + "," + std::to_string(c.m) + "," + fmt(c.tplus) + ",";
  r += (c.m == 1 ? std::string("0") : fmt(c.tplus - c.log2m_dn)) + "," + c.winner + "," + c.params + ",";
  if (c.m == 1) return r + ",,,,,,";
  r += opt(c.exact) + "," + opt(c.trivial) + "," + opt(c.fixed) + "," + opt(c.uni) + "," + c.uni.params + ",";
  r += opt(c.expl) + "," + c.expl.params;
  return r;
}

bool defined(long n, u64 m) { return m >= 1 && (n >= 64 || m <= (u64(1) << n)); }

void self_check() {
  volatile double one = 1, three = 3;
  double up = one / three;
  std::fesetround(FE_DOWNWARD);
  double dn = one / three;
  std::fesetround(FE_UPWARD);
  if (!(up > dn)) {
    std::cerr << "rounding modes are not honoured; refusing to run\n";
    std::exit(3);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::fesetround(FE_UPWARD);
  self_check();
  LN2_UP = log_up(2.0);
  LN2_DN = log_dn(2.0);
  std::vector<std::string> args;
  std::string dir = "data/gradient-bound";
  bool use_exact = true;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--data" && i + 1 < argc) dir = argv[++i];
    else if (a == "--no-exact") use_exact = false;
    else args.push_back(a);
  }
  if (args.empty()) {
    std::cerr << "usage: gradient_tplus cell <n> <m> | cells | grid [out.csv]  [--no-exact] [--data dir]\n";
    return 2;
  }
  read_data(dir);
  build_tables();
  if (args[0] == "cell" && args.size() == 3) {
    long n = std::stol(args[1]);
    u64 m = std::stoull(args[2]);
    if (n < 1 || !defined(n, m)) {
      std::cout << "n=" << n << " m=" << m << ": m > 2^n, the set of tables is empty; t^+ is defined only vacuously\n";
      return 0;
    }
    Cell c = evaluate(n, m, use_exact);
    std::cout << "n=" << n << " m=" << m << "\n";
    std::cout << "t+        = " << fmt(c.tplus) << "  (" << c.winner << (c.params.empty() ? "" : ", " + c.params) << ")\n";
    if (m == 1) return 0;
    std::cout << "t+ - log m <= " << fmt(c.tplus - c.log2m_dn) << "\n";
    std::cout << "C(m)      >= " << fmt(c.C_dn) << "\n";
    auto show = [](const char* name, const Ingredient& i) {
      std::cout << "  " << name << (i.value == INF ? std::string("n/a") : fmt(i.value))
                << (i.params.empty() ? "" : "  [" + i.params + "]") << "\n";
    };
    show("(a) exact    ", c.exact);
    show("(b) trivial  ", c.trivial);
    show("(c) fixed-m  ", c.fixed);
    show("(d) union    ", c.uni);
    show("(e) explicit ", c.expl);
    return 0;
  }
  if (args[0] == "cells") {
    std::cout << csv_header() << "\n";
    long n;
    u64 m;
    while (std::cin >> n >> m) {
      if (n >= 1 && defined(n, m)) std::cout << csv_row(evaluate(n, m, use_exact)) << "\n";
    }
    return 0;
  }
  if (args[0] == "grid") {
    std::string out = args.size() > 1 ? args[1] : dir + "/tplus.csv";
    std::vector<long> ns;
    for (long n = 1; n <= 12; ++n) ns.push_back(n);
    for (long n : {16L, 20L, 24L, 32L, 48L, 64L, 96L, 128L, 256L, 512L, 1024L, 4096L, 1L << 16, 1L << 20}) ns.push_back(n);
    std::vector<u64> ms;
    for (u64 m = 1; m <= 16; ++m) ms.push_back(m);
    for (u64 m : {24ULL, 32ULL, 48ULL, 64ULL, 128ULL, 256ULL, 512ULL, 1024ULL, 1ULL << 12, 1ULL << 16, 1ULL << 20,
                  1ULL << 24, 1ULL << 32})
      ms.push_back(m);
    std::ofstream f(out);
    f << csv_header() << "\n";
    for (long n : ns)
      for (u64 m : ms)
        if (defined(n, m)) f << csv_row(evaluate(n, m, use_exact)) << "\n" << std::flush;
    return 0;
  }
  std::cerr << "unknown command\n";
  return 2;
}
