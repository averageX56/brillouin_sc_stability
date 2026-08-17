// estimators.hpp — observables: linewidth (phase MSD and |g1| decay) and g2(0).
//
// All estimators skip non-finite samples (diverged paths) instead of poisoning the
// whole sweep with NaN.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

#include "model.hpp"
#include "solver.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace brillouin {

struct FitResult {
  double value = std::numeric_limits<double>::quiet_NaN();
  double r2 = std::numeric_limits<double>::quiet_NaN();
};

// Ordinary least squares y = slope*x + intercept, plus R^2.
inline bool ols(const std::vector<double>& x, const std::vector<double>& y,
                double& slope, double& intercept, double& r2) {
  const std::size_t n = x.size();
  if (n < 2) return false;
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (std::size_t i = 0; i < n; ++i) {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    sxy += x[i] * y[i];
  }
  const double den = n * sxx - sx * sx;
  if (std::abs(den) < 1e-300) return false;
  slope = (n * sxy - sx * sy) / den;
  intercept = (sy - slope * sx) / n;
  const double ybar = sy / n;
  double ss_tot = 0, ss_res = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double pred = slope * x[i] + intercept;
    ss_tot += (y[i] - ybar) * (y[i] - ybar);
    ss_res += (y[i] - pred) * (y[i] - pred);
  }
  r2 = (ss_tot > 0) ? 1.0 - ss_res / ss_tot : std::numeric_limits<double>::quiet_NaN();
  return true;
}

// Unwrapped phase of mode j for one path (numpy.unwrap semantics, period 2*pi).
inline std::vector<double> unwrapped_phase(const Paths& P, int path, int j) {
  std::vector<double> ph(P.n_keep);
  double prev = 0.0, offset = 0.0;
  for (long k = 0; k < P.n_keep; ++k) {
    const cdouble z = P.at(k, path, j);
    double a = std::arg(z);  // (-pi, pi]
    if (k == 0) {
      ph[k] = a;
      prev = a;
      continue;
    }
    double d = a - prev;
    // numpy.unwrap: shift jumps greater than pi into (-pi, pi]
    double dmod = std::fmod(d + M_PI, 2.0 * M_PI);
    if (dmod < 0) dmod += 2.0 * M_PI;
    dmod -= M_PI;
    if (dmod == -M_PI && d > 0) dmod = M_PI;
    offset += dmod - d;
    ph[k] = a + offset;
    prev = a;
  }
  return ph;
}

// Geometrically spaced unique lags in [lo, hi].
inline std::vector<long> geom_lags(long lo, long hi, int nlags) {
  std::vector<long> lags;
  if (hi <= lo) return lags;
  for (int i = 0; i < nlags; ++i) {
    const double f = (nlags == 1) ? 0.0 : static_cast<double>(i) / (nlags - 1);
    const long L = static_cast<long>(std::llround(
        std::exp(std::log(static_cast<double>(lo)) +
                 f * (std::log(static_cast<double>(hi)) - std::log(static_cast<double>(lo))))));
    if (lags.empty() || L > lags.back()) lags.push_back(L);
  }
  return lags;
}

// D_eff from the slope of Var[dphi(tau)] = D*tau, pooled over paths and origins.
inline FitResult D_from_msd(const Paths& P, int j, double lag_min_frac = 0.001,
                            double lag_max_frac = 0.15, int nlags = 60) {
  FitResult res;
  const long n_t = P.n_keep;
  if (n_t < 20 || P.T.size() < 2) return res;
  const double dt = P.T[1] - P.T[0];

  const long lo = std::max(2L, static_cast<long>(lag_min_frac * n_t));
  const long hi = std::max(lo + 5, static_cast<long>(lag_max_frac * n_t));
  const std::vector<long> lags = geom_lags(lo, std::min(hi, n_t - 1), nlags);
  if (lags.size() < 5) return res;

  // Precompute unwrapped phases once per path.
  std::vector<std::vector<double>> ph(P.n_paths);
  for (int p = 0; p < P.n_paths; ++p) ph[p] = unwrapped_phase(P, p, j);

  std::vector<double> tau, var;
  tau.reserve(lags.size());
  var.reserve(lags.size());
  for (long L : lags) {
    // Variance over all paths and all origins (matches numpy's ph[L:] - ph[:-L]).
    double s = 0.0, s2 = 0.0;
    long cnt = 0;
    for (int p = 0; p < P.n_paths; ++p) {
      const auto& v = ph[p];
      for (long k = 0; k + L < n_t; ++k) {
        const double d = v[k + L] - v[k];
        if (!std::isfinite(d)) continue;
        s += d;
        s2 += d * d;
        ++cnt;
      }
    }
    if (cnt < 2) continue;
    const double mean = s / cnt;
    const double vv = s2 / cnt - mean * mean;
    tau.push_back(static_cast<double>(L) * dt);
    var.push_back(std::max(vv, 0.0));
  }
  if (tau.size() < 6) return res;

  // Fit the central 20%-90% window, as in the original.
  const std::size_t n = tau.size();
  const std::size_t a = static_cast<std::size_t>(0.2 * n);
  const std::size_t b = static_cast<std::size_t>(0.9 * n);
  if (b <= a + 1) return res;
  std::vector<double> tx(tau.begin() + a, tau.begin() + b);
  std::vector<double> vy(var.begin() + a, var.begin() + b);

  double slope, icept, r2;
  if (!ols(tx, vy, slope, icept, r2)) return res;
  res.value = std::max(slope, 0.0);
  res.r2 = r2;
  return res;
}

// --- minimal radix-2 iterative FFT (in-place) -------------------------------
inline void fft_inplace(std::vector<cdouble>& a, bool inverse) {
  const std::size_t n = a.size();
  for (std::size_t i = 1, jj = 0; i < n; ++i) {
    std::size_t bit = n >> 1;
    for (; jj & bit; bit >>= 1) jj ^= bit;
    jj ^= bit;
    if (i < jj) std::swap(a[i], a[jj]);
  }
  for (std::size_t len = 2; len <= n; len <<= 1) {
    const double ang = 2.0 * M_PI / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
    const cdouble wl(std::cos(ang), std::sin(ang));
    for (std::size_t i = 0; i < n; i += len) {
      cdouble w(1.0, 0.0);
      for (std::size_t k = 0; k < len / 2; ++k) {
        const cdouble u = a[i + k];
        const cdouble v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
        w *= wl;
      }
    }
  }
  if (inverse)
    for (auto& z : a) z /= static_cast<double>(n);
}

// |<a*(t) a(t+tau)>| via FFT, averaged over paths (biased/linear correlation).
inline void g1_abs(const Paths& P, int j, double max_frac, std::vector<double>& tau,
                   std::vector<double>& g) {
  tau.clear();
  g.clear();
  const long n = P.n_keep;
  if (n < 10 || P.T.size() < 2) return;
  const double dt = P.T[1] - P.T[0];
  const long L = std::max(2L, static_cast<long>(max_frac * n));

  std::size_t nfft = 1;
  while (nfft < static_cast<std::size_t>(2 * n)) nfft <<= 1;

  std::vector<cdouble> acc(nfft, cdouble(0.0, 0.0));
  int used = 0;
  std::vector<cdouble> buf(nfft);
  for (int p = 0; p < P.n_paths; ++p) {
    bool ok = true;
    for (long k = 0; k < n; ++k) {
      const cdouble z = P.at(k, p, j);
      if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) {
        ok = false;
        break;
      }
      buf[k] = z;
    }
    if (!ok) continue;  // skip diverged path
    std::fill(buf.begin() + n, buf.end(), cdouble(0.0, 0.0));
    fft_inplace(buf, false);
    for (std::size_t i = 0; i < nfft; ++i) buf[i] = buf[i] * std::conj(buf[i]);
    fft_inplace(buf, true);
    for (long k = 0; k < L; ++k) acc[k] += buf[k];
    ++used;
  }
  if (used == 0) return;

  tau.resize(L);
  g.resize(L);
  for (long k = 0; k < L; ++k) {
    const cdouble c = acc[k] / static_cast<double>(used) / static_cast<double>(n - k);
    tau[k] = static_cast<double>(k) * dt;
    g[k] = std::abs(c);
  }
}

// D_eff from the |g1| decay rate: |g1| ~ exp(-D*tau/2).
inline FitResult D_from_g1(const Paths& P, int j, double floor = 0.25,
                           double max_frac = 0.15) {
  FitResult res;
  std::vector<double> tau, g;
  g1_abs(P, j, max_frac, tau, g);
  if (g.empty() || !(g[0] > 0)) return res;

  std::vector<double> tx, ly;
  auto collect = [&](double thr) {
    tx.clear();
    ly.clear();
    for (std::size_t i = 0; i < tau.size(); ++i) {
      const double gn = g[i] / g[0];
      if (tau[i] > 0 && gn > thr && std::isfinite(gn)) {
        tx.push_back(tau[i]);
        ly.push_back(std::log(gn));
      }
    }
  };
  collect(floor);
  if (tx.size() < 10) collect(0.0);
  if (tx.size() < 3) return res;

  double slope, icept, r2;
  if (!ols(tx, ly, slope, icept, r2)) return res;
  res.value = std::max(-2.0 * slope, 0.0);
  res.r2 = r2;
  return res;
}

// Exact nonlinear g2(0) = <I^2>/<I>^2 with I = |a_j|^2, pooled over all samples.
inline double g2_zero(const Paths& P, int j) {
  double s1 = 0.0, s2 = 0.0;
  long cnt = 0;
  for (long k = 0; k < P.n_keep; ++k)
    for (int p = 0; p < P.n_paths; ++p) {
      const cdouble z = P.at(k, p, j);
      const double I = std::norm(z);  // |z|^2
      if (!std::isfinite(I)) continue;
      s1 += I;
      s2 += I * I;
      ++cnt;
    }
  if (cnt == 0 || s1 <= 0) return std::numeric_limits<double>::quiet_NaN();
  const double mI = s1 / cnt;
  return (s2 / cnt) / (mI * mI);
}

// Mean modulus <|y_j|> over all samples.
inline double mean_abs(const Paths& P, int j) {
  double s = 0.0;
  long cnt = 0;
  for (long k = 0; k < P.n_keep; ++k)
    for (int p = 0; p < P.n_paths; ++p) {
      const double v = std::abs(P.at(k, p, j));
      if (!std::isfinite(v)) continue;
      s += v;
      ++cnt;
    }
  return cnt ? s / cnt : std::numeric_limits<double>::quiet_NaN();
}

}  // namespace brillouin
