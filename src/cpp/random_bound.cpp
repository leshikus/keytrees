// random_bound: computations for articles/random-algorithm-bound.
//
// The random algorithm R_s on a uniform set of m distinct binary words of
// length n has average cost t(n,m). For R_3 and m >= 4,
//   t(n,m) = 1 + sum_k c(n,m,k) (k t(n-1,k) + (m-k) t(n-1,m-k)) / m,
// with t(n,2) = 1 and t(n,3) = 5/3. R_inf (PATRICIA) never charges a test that
// does not split. The limits t_m = lim_n t(n,m) obey the binomial recurrence.
//
// Subcommands (exact arithmetic uses Boost.Multiprecision, header-only):
//   finite <Nmax> <out.csv>      exact t(n,m) for R_3, n <= Nmax, all m <= 2^n:
//                                monotonicity in n and the convexity facts of
//                                u_n(k) = k t(n,k) used in the proof
//   limit <Mmax> <out.csv>       limits for R_3 and PATRICIA in long double:
//                                redundancy r_m = t_m - log2 m and r_m + alpha/m
//   enclose <Mmax> <bits> <out.csv>
//                                rigorous fixed-point enclosures lo/2^bits <= t_m <= hi/2^bits
//                                for R_3 and PATRICIA (positive recurrence, outward rounding)
//   patricia <Nmax> <Mmax>       monotonicity in n of the exact PATRICIA solution of
//                                Prodinger and Szpankowski, in long double
#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <print>
#include <string>
#include <vector>

namespace {

using boost::multiprecision::cpp_int;
using boost::multiprecision::cpp_rational;
using Real = long double;

cpp_int binom(int n, int k) {
  if (k < 0 || k > n) return 0;
  cpp_int c = 1;
  for (int i = 0; i < k; ++i) c = c * (n - i) / (i + 1);
  return c;
}

double to_double(const cpp_rational& q) { return static_cast<double>(q); }

// Exact R_3 values; checks monotonicity in n and the convexity facts.
int cmd_finite(int nmax, const std::string& out_path) {
  std::ofstream out(out_path);
  out << "n,monotone_violations,min_increment,negative_second_differences,d2u_3,d2u_4,F1,F2,"
         "min_d2u_other,zero_second_differences\n";
  std::vector<cpp_rational> prev{0, 0};
  for (int n = 1; n <= nmax; ++n) {
    const int top = 1 << n;
    const long long h = 1LL << (n - 1);
    std::vector<cpp_rational> cur(static_cast<std::size_t>(top) + 1, cpp_rational(0));
    for (int m = 2; m <= top; ++m) {
      if (m == 2) { cur[2] = 1; continue; }
      if (m == 3) { cur[3] = cpp_rational(5, 3); continue; }
      const int lo = static_cast<int>(std::max<long long>(0, m - h));
      const int hi = static_cast<int>(std::min<long long>(m, h));
      // c(n,m,k) = C(h,k) C(h,m-k) / C(2h,m), computed by ratios from k = lo.
      cpp_int wk = binom(static_cast<int>(h), lo) * binom(static_cast<int>(h), m - lo);
      cpp_rational acc = 0;
      for (int k = lo; k <= hi; ++k) {
        if (k > lo) wk = wk * (h - k + 1) * (m - k + 1) / (cpp_int(k) * (h - m + k));
        if (k > 0) acc += cpp_rational(wk * 2 * k) * prev[static_cast<std::size_t>(k)];
      }
      cur[static_cast<std::size_t>(m)] = 1 + acc / (cpp_rational(binom(static_cast<int>(2 * h), m)) * m);
    }
    int violations = 0;
    cpp_rational min_inc = 1000;
    if (n > 1)
      for (int m = 2; m <= top / 2; ++m) {
        const cpp_rational inc = cur[static_cast<std::size_t>(m)] - prev[static_cast<std::size_t>(m)];
        if (inc < 0) ++violations;
        if (m >= 4 && inc < min_inc) min_inc = inc;
      }
    std::vector<cpp_rational> d2(static_cast<std::size_t>(top) + 1, cpp_rational(0));
    auto u = [&](int k) { return cpp_rational(k) * cur[static_cast<std::size_t>(k)]; };
    std::string negatives, zeros;
    cpp_rational min_other = 1000;
    for (int j = 1; j < top; ++j) {
      d2[static_cast<std::size_t>(j)] = u(j - 1) - 2 * u(j) + u(j + 1);
      if (d2[static_cast<std::size_t>(j)] < 0) negatives += (negatives.empty() ? "" : " ") + std::to_string(j);
      if (d2[static_cast<std::size_t>(j)] == 0) zeros += (zeros.empty() ? "" : " ") + std::to_string(j);
      if (j != 4 && d2[static_cast<std::size_t>(j)] > 0 && d2[static_cast<std::size_t>(j)] < min_other)
        min_other = d2[static_cast<std::size_t>(j)];
    }
    const double f1 = top > 5 ? to_double(d2[3] + 2 * d2[4]) : 0.0;
    const double f2 = top > 6 ? to_double(d2[4] + d2[5]) : 0.0;
    out << n << ',' << violations << ',' << (n > 2 ? to_double(min_inc) : 0.0) << ",\"" << negatives << "\","
        << (top > 4 ? to_double(d2[3]) : 0.0) << ',' << (top > 5 ? to_double(d2[4]) : 0.0) << ',' << f1 << ','
        << f2 << ',' << to_double(min_other) << ",\"" << zeros << "\"\n";
    std::println("n={:2}: decreases {} ; negative d2u at [{}] ; F1 {:.6f} ; F2 {:.6f}", n, violations, negatives, f1,
                 f2);
    prev.swap(cur);
  }
  return 0;
}

// Limits in long double, for R_3 (patricia = false) and PATRICIA.
std::vector<Real> limits(int mmax, bool patricia) {
  std::vector<Real> t(static_cast<std::size_t>(mmax) + 1, 0.0L);
  if (mmax >= 2) t[2] = 1.0L;
  if (mmax >= 3) t[3] = 5.0L / 3.0L;
  for (int m = 4; m <= mmax; ++m) {
    // sum_{k=2}^{m-1} C(m-1,k-1) 2^{1-m} t_k, with log-space binomial weights.
    const int nn = m - 1;
    Real acc = 0.0L;
    const Real base = std::lgamma(static_cast<Real>(nn) + 1) - nn * std::log(2.0L);
    for (int i = 1; i < nn; ++i) {
      const Real lw = base - std::lgamma(static_cast<Real>(i) + 1) - std::lgamma(static_cast<Real>(nn - i) + 1);
      if (lw < -12000.0L) continue;
      acc += std::exp(lw) * t[static_cast<std::size_t>(i) + 1];
    }
    const Real q = std::ldexp(1.0L, 1 - m);
    t[static_cast<std::size_t>(m)] = patricia ? (1.0L - q + acc) / (1.0L - q) : (1.0L + acc) / (1.0L - q);
  }
  return t;
}

int cmd_limit(int mmax, const std::string& out_path) {
  const auto t3 = limits(mmax, false);
  const auto tp = limits(mmax, true);
  std::ofstream out(out_path);
  out.precision(15);
  out << "m,t_s3,r_s3,t_patricia,r_patricia\n";
  constexpr Real alpha = 0.723L;
  Real best3 = -1, bestp = -1, sup3 = -1, supp = -1;
  int arg3 = 0, argp = 0, argsup3 = 0, argsupp = 0;
  for (int m = 2; m <= mmax; ++m) {
    const Real l = std::log2(static_cast<Real>(m));
    const Real r3 = t3[static_cast<std::size_t>(m)] - l, rp = tp[static_cast<std::size_t>(m)] - l;
    if (r3 > sup3) { sup3 = r3; argsup3 = m; }
    if (rp > supp) { supp = rp; argsupp = m; }
    if (m >= 8 && r3 + alpha / m > best3) { best3 = r3 + alpha / m; arg3 = m; }
    if (m >= 8 && rp + alpha / m > bestp) { bestp = rp + alpha / m; argp = m; }
    const bool keep = m <= 256 || m % 16 == 0;
    if (keep)
      out << m << ',' << static_cast<double>(t3[static_cast<std::size_t>(m)]) << ',' << static_cast<double>(r3)
          << ',' << static_cast<double>(tp[static_cast<std::size_t>(m)]) << ',' << static_cast<double>(rp) << '\n';
  }
  std::println("R_3:      max r_m = {:.10f} at m = {}; max_(m>=8) r_m + 0.723/m = {:.10f} at m = {}",
               static_cast<double>(sup3), argsup3, static_cast<double>(best3), arg3);
  std::println("PATRICIA: max r_m = {:.10f} at m = {}; max_(m>=8) r_m + 0.723/m = {:.10f} at m = {}",
               static_cast<double>(supp), argsupp, static_cast<double>(bestp), argp);
  return 0;
}

// Fixed-point enclosures with outward rounding; writes lo and hi as decimals.
int cmd_enclose(int mmax, int bits, const std::string& out_path) {
  std::ofstream out(out_path);
  out << "variant,m,lo,hi\n";
  const cpp_int one = cpp_int(1) << bits;
  for (int variant = 0; variant < 2; ++variant) {
    const bool patricia = variant == 1;
    std::vector<cpp_int> lo(static_cast<std::size_t>(mmax) + 1, 0), hi = lo;
    for (int m = 2; m <= mmax; ++m) {
      const cpp_int pw = cpp_int(1) << (m - 1);
      const cpp_int den = pw - 1;
      const cpp_int toll = (!patricia && m > 3) ? pw : den;
      cpp_int slo = 0, shi = 0, c = 1;  // c = C(m-1, k-1)
      for (int k = 1; k < m; ++k) {
        if (k >= 2) {
          slo += c * lo[static_cast<std::size_t>(k)];
          shi += c * hi[static_cast<std::size_t>(k)];
        }
        c = c * (m - k) / k;
      }
      const cpp_int nlo = toll * one + slo, nhi = toll * one + shi;
      lo[static_cast<std::size_t>(m)] = nlo / den;                 // floor
      hi[static_cast<std::size_t>(m)] = (nhi + den - 1) / den;     // ceiling
      // Print with 30 decimals, rounded outward: floor(lo*10^30/2^bits), ceil(hi*...).
      const cpp_int ten30 = boost::multiprecision::pow(cpp_int(10), 30);
      const cpp_int dlo = lo[static_cast<std::size_t>(m)] * ten30 >> bits;
      const cpp_int dhi = ((hi[static_cast<std::size_t>(m)] * ten30) + one - 1) >> bits;
      out << (patricia ? "patricia" : "s3") << ',' << m << ',' << dlo << ',' << dhi << '\n';
    }
  }
  std::println("enclosures for m <= {} written to {} (decimals scaled by 10^30)", mmax, out_path);
  return 0;
}

// Exact PATRICIA solution t(n,m) = sum_{l<n} (1 - Q_l), Q_l = prod_{i<m-1} (1 - 2^l/(2^n-1-i)).
int cmd_patricia(int nmax, int mmax) {
  Real worst = 1e9L;
  int wn = 0, wm = 0;
  std::vector<Real> prev;
  for (int n = 1; n <= nmax; ++n) {
    const long long top = n < 40 ? std::min<long long>(mmax, 1LL << n) : mmax;
    std::vector<Real> q(static_cast<std::size_t>(n), 1.0L), cur(static_cast<std::size_t>(top) + 1, 0.0L);
    const Real big = std::ldexp(1.0L, n) - 1.0L;
    for (long long m = 1; m <= top; ++m) {
      if (m >= 2)
        for (int l = 0; l < n; ++l) q[static_cast<std::size_t>(l)] *= 1.0L - std::ldexp(1.0L, l) / (big - static_cast<Real>(m - 2));
      Real t = 0.0L;
      for (int l = 0; l < n; ++l) t += 1.0L - q[static_cast<std::size_t>(l)];
      cur[static_cast<std::size_t>(m)] = t;
      if (n > 1 && m >= 4 && m < static_cast<long long>(prev.size())) {
        const Real inc = (t - prev[static_cast<std::size_t>(m)]) * std::ldexp(1.0L, n) / static_cast<Real>(m - 1);
        if (inc < worst) { worst = inc; wn = n; wm = static_cast<int>(m); }
      }
    }
    prev.swap(cur);
  }
  std::println("PATRICIA, n <= {}, m <= {}: min over m >= 4 of (t(n,m) - t(n-1,m)) 2^n/(m-1) = {:.6f} at (n,m) = ({},{})",
               nmax, mmax, static_cast<double>(worst), wn, wm);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "finite" && argc == 4) return cmd_finite(std::atoi(argv[2]), argv[3]);
  if (cmd == "limit" && argc == 4) return cmd_limit(std::atoi(argv[2]), argv[3]);
  if (cmd == "enclose" && argc == 5) return cmd_enclose(std::atoi(argv[2]), std::atoi(argv[3]), argv[4]);
  if (cmd == "patricia" && argc == 4) return cmd_patricia(std::atoi(argv[2]), std::atoi(argv[3]));
  std::println(stderr,
               "usage: random_bound finite Nmax out.csv | limit Mmax out.csv | enclose Mmax bits out.csv | "
               "patricia Nmax Mmax");
  return 2;
}
