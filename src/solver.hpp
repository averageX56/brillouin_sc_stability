// solver.hpp — deterministic steady state + strong SDE integrators.
//
// The drift is exactly quadratic in x, so the directional derivatives used by the
// Kloeden & Platen order-1.5 additive-noise scheme are computed by central
// differences that are EXACT for any step h (no truncation error).
//
// All loops run over the RUNTIME active sizes p.dim() / p.nvar() / p.m_noise(),
// so the same code handles any photon count; at order 3 it reproduces the
// original results bit-for-bit.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "model.hpp"
#include "rng.hpp"

namespace brillouin {

// Forward declaration: exact directional derivative of drift() (defined
// further below), needed by jacobian() to assemble the full Jacobian.
inline State dir_deriv(const State& X, const State& V, double E, const Params& p,
                       double h);

// Build the exact Jacobian dF/dx (dim x dim, row-major) via dir_deriv, which
// is exact for this quadratic drift (no finite-difference truncation error).
inline void jacobian(const State& x, double E, const Params& p,
                     std::vector<double>& J) {
  const int dim = p.dim();
  J.assign(static_cast<std::size_t>(dim) * dim, 0.0);
  State e{};
  for (int col = 0; col < dim; ++col) {
    e[col] = 1.0;
    const State d = dir_deriv(x, e, E, p, 1.0);
    for (int row = 0; row < dim; ++row) J[static_cast<std::size_t>(row) * dim + col] = d[row];
    e[col] = 0.0;
  }
}

// Solve J dx = -F by Gaussian elimination with partial pivoting (dim is small:
// 2*(order+2), typically 10). Returns false if J is (numerically) singular,
// which legitimately happens exactly at a bifurcation point.
inline bool newton_step(std::vector<double> J, State F, int dim, State& dx) {
  std::vector<double> rhs(dim);
  for (int i = 0; i < dim; ++i) rhs[i] = -F[i];
  for (int col = 0; col < dim; ++col) {
    int piv = col;
    double best = std::abs(J[static_cast<std::size_t>(col) * dim + col]);
    for (int r = col + 1; r < dim; ++r) {
      const double v = std::abs(J[static_cast<std::size_t>(r) * dim + col]);
      if (v > best) { best = v; piv = r; }
    }
    if (!(best > 1e-300)) return false;
    if (piv != col) {
      for (int c = 0; c < dim; ++c)
        std::swap(J[static_cast<std::size_t>(col) * dim + c], J[static_cast<std::size_t>(piv) * dim + c]);
      std::swap(rhs[col], rhs[piv]);
    }
    const double d = J[static_cast<std::size_t>(col) * dim + col];
    for (int r = col + 1; r < dim; ++r) {
      const double f = J[static_cast<std::size_t>(r) * dim + col] / d;
      if (f == 0.0) continue;
      for (int c = col; c < dim; ++c)
        J[static_cast<std::size_t>(r) * dim + c] -= f * J[static_cast<std::size_t>(col) * dim + c];
      rhs[r] -= f * rhs[col];
    }
  }
  for (int r = dim - 1; r >= 0; --r) {
    double s = rhs[r];
    for (int c = r + 1; c < dim; ++c) s -= J[static_cast<std::size_t>(r) * dim + c] * dx[c];
    dx[r] = s / J[static_cast<std::size_t>(r) * dim + r];
  }
  return true;
}

// ---------------------------------------------------------------------------
// Deterministic steady state: adaptive Dormand-Prince RK45 with PI step control.
//
// Near the hard-excitation threshold one Jacobian eigenvalue passes close to
// zero (that IS the bifurcation), so relaxation to the fixed point can take
// far longer than the nominal 400/gamma_min horizon, and the trajectory can
// still be slowly spiralling in when RK45 stops. To stay robust there without
// paying for it away from thresholds, this function (1) extends the RK45
// integration in bounded doublings whenever it runs out of time still short
// of the residual target, and (2) finishes with a damped Newton polish using
// the exact Jacobian — quadratic convergence for the last few decades of
// residual that pure RK45 would otherwise creep through step by step.
// ---------------------------------------------------------------------------
struct SteadyResult {
  State x{};
  bool converged = false;
  double t_reached = 0.0;
  double residual = 0.0;  // ||drift|| at the end, scaled
  long n_steps = 0;
  int extensions = 0;    // how many extra RK45 horizons were needed
  int newton_iters = 0;  // Newton polish iterations actually taken
};

inline double scaled_residual(const State& d, const State& x, int dim, double gscale) {
  return max_abs(d, dim) / std::max(1.0, max_abs(x, dim)) / gscale;
}

inline SteadyResult rk45_run(double E, const Params& p, State x, double t_end,
                             double rtol, double atol, double gscale) {
  const int dim = p.dim();
  static constexpr double a21 = 1.0 / 5;
  static constexpr double a31 = 3.0 / 40, a32 = 9.0 / 40;
  static constexpr double a41 = 44.0 / 45, a42 = -56.0 / 15, a43 = 32.0 / 9;
  static constexpr double a51 = 19372.0 / 6561, a52 = -25360.0 / 2187,
                          a53 = 64448.0 / 6561, a54 = -212.0 / 729;
  static constexpr double a61 = 9017.0 / 3168, a62 = -355.0 / 33,
                          a63 = 46732.0 / 5247, a64 = 49.0 / 176,
                          a65 = -5103.0 / 18656;
  static constexpr double b1 = 35.0 / 384, b3 = 500.0 / 1113, b4 = 125.0 / 192,
                          b5 = -2187.0 / 6784, b6 = 11.0 / 84;
  static constexpr double e1 = 71.0 / 57600, e3 = -71.0 / 16695, e4 = 71.0 / 1920,
                          e5 = -17253.0 / 339200, e6 = 22.0 / 525, e7 = -1.0 / 40;

  double t = 0.0;
  double h = std::min(1e-3 / gscale, t_end / 100.0);
  const double h_min = 1e-14 * std::max(1.0, t_end);
  SteadyResult out;
  State k1 = drift(x, E, p);
  long nstep = 0;
  const long max_steps = 20000000;

  auto lin = [&](const State& base,
                 std::initializer_list<std::pair<double, const State*>> terms) {
    State r = base;
    for (auto& [c, v] : terms)
      for (int i = 0; i < dim; ++i) r[i] += c * (*v)[i];
    return r;
  };

  bool early_break = false;
  while (t < t_end && nstep < max_steps) {
    if (h > t_end - t) h = t_end - t;
    if (h < h_min) break;

    State y2 = lin(x, {{h * a21, &k1}});
    State k2 = drift(y2, E, p);
    State y3 = lin(x, {{h * a31, &k1}, {h * a32, &k2}});
    State k3 = drift(y3, E, p);
    State y4 = lin(x, {{h * a41, &k1}, {h * a42, &k2}, {h * a43, &k3}});
    State k4 = drift(y4, E, p);
    State y5 = lin(x, {{h * a51, &k1}, {h * a52, &k2}, {h * a53, &k3}, {h * a54, &k4}});
    State k5 = drift(y5, E, p);
    State y6 = lin(x, {{h * a61, &k1}, {h * a62, &k2}, {h * a63, &k3}, {h * a64, &k4}, {h * a65, &k5}});
    State k6 = drift(y6, E, p);
    State x_new = lin(x, {{h * b1, &k1}, {h * b3, &k3}, {h * b4, &k4}, {h * b5, &k5}, {h * b6, &k6}});
    State k7 = drift(x_new, E, p);

    double err = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double e = h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] +
                            e6 * k6[i] + e7 * k7[i]);
      const double sc = atol + rtol * std::max(std::abs(x[i]), std::abs(x_new[i]));
      const double r = e / sc;
      err += r * r;
    }
    err = std::sqrt(err / dim);

    if (!finite(x_new, dim)) {
      h *= 0.1;
      continue;
    }

    if (err <= 1.0) {
      t += h;
      x = x_new;
      k1 = k7;  // FSAL
      ++nstep;
      if ((nstep & 511) == 0) {
        const double res = scaled_residual(k1, x, dim, gscale);
        if (res < 1e-8) { early_break = true; break; }
      }
      const double fac = (err == 0.0) ? 5.0 : 0.9 * std::pow(err, -0.2);
      h *= std::min(5.0, std::max(0.2, fac));
    } else {
      h *= std::min(1.0, std::max(0.1, 0.9 * std::pow(err, -0.25)));
    }
  }

  const State d = drift(x, E, p);
  out.x = x;
  out.t_reached = t;
  out.n_steps = nstep;
  out.residual = scaled_residual(d, x, dim, gscale);
  out.converged = finite(x, dim) && (out.residual < 1e-6 || early_break);
  return out;
}

inline SteadyResult steady_state(double E, const Params& p, double t_end = -1.0,
                                 double rtol = 1e-10, double atol = 1e-10,
                                 double x0_fill = 10.0) {
  const int dim = p.dim();
  double gmin = p.gammas[0].real();
  for (int i = 0; i < p.nvar(); ++i) gmin = std::min(gmin, p.gammas[i].real());
  gmin = std::max(gmin, 1e-300);
  if (t_end <= 0.0) t_end = 400.0 / gmin;

  State x{};
  for (int i = 0; i < dim; ++i) x[i] = x0_fill;

  // 1) RK45 to the nominal horizon, then bounded doublings if the residual
  // target (1e-6, matched by the final out.converged check below) hasn't
  // been reached yet — critical slowing down near a threshold can need much
  // more than 400/gamma_min. Cap total extra time at 64x the nominal horizon
  // so a genuinely non-converging point (e.g. a limit cycle) still returns
  // promptly instead of spinning for 20M steps repeatedly.
  SteadyResult out = rk45_run(E, p, x, t_end, rtol, atol, gmin);
  int extensions = 0;
  double horizon = t_end;
  while (!out.converged && finite(out.x, dim) && extensions < 6 && horizon < 64.0 * t_end) {
    horizon *= 2.0;
    out = rk45_run(E, p, out.x, horizon, rtol, atol, gmin);
    ++extensions;
  }
  out.extensions = extensions;

  // 2) Damped Newton polish: quadratic convergence for the last stretch,
  // where RK45 alone would need many more (cheap but serial) small steps.
  // Best-effort — only accepted while it strictly reduces the residual, and
  // abandoned (falling back to the RK45 state) if the Jacobian is singular
  // (which happens exactly AT a bifurcation, where Newton is not meaningful)
  // or if damping bottoms out without progress.
  if (finite(out.x, dim)) {
    State x_cur = out.x;
    State F = drift(x_cur, E, p);
    double res = scaled_residual(F, x_cur, dim, gmin);
    std::vector<double> J;
    int iters = 0;
    for (; iters < 30 && res > 1e-12 && res < 1e6; ++iters) {
      jacobian(x_cur, E, p, J);
      State dx{};
      if (!newton_step(J, F, dim, dx)) break;
      double lambda = 1.0;
      bool improved = false;
      for (int tries = 0; tries < 20; ++tries) {
        State x_try = axpy(x_cur, lambda, dx, dim);
        if (finite(x_try, dim)) {
          const State F_try = drift(x_try, E, p);
          const double res_try = scaled_residual(F_try, x_try, dim, gmin);
          if (res_try < res) {
            x_cur = x_try;
            F = F_try;
            res = res_try;
            improved = true;
            break;
          }
        }
        lambda *= 0.5;
      }
      if (!improved) break;
    }
    if (res < out.residual) {
      out.x = x_cur;
      out.residual = res;
      out.newton_iters = iters;
      out.converged = finite(x_cur, dim) && res < 1e-6;
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Exact directional derivatives for a quadratic drift.
// ---------------------------------------------------------------------------
inline State dir_deriv(const State& X, const State& V, double E, const Params& p,
                       double h = 1.0) {
  const int dim = p.dim();
  const State Xp = axpy(X, h, V, dim);
  const State Xm = axpy(X, -h, V, dim);
  const State ap = drift(Xp, E, p);
  const State am = drift(Xm, E, p);
  State r{};
  for (int i = 0; i < dim; ++i) r[i] = (ap[i] - am[i]) / (2.0 * h);
  return r;
}

inline State dir_second(const State& X, const State& V, const State& aX, double E,
                        const Params& p, double h = 1.0) {
  const int dim = p.dim();
  const State ap = drift(axpy(X, h, V, dim), E, p);
  const State am = drift(axpy(X, -h, V, dim), E, p);
  State r{};
  for (int i = 0; i < dim; ++i) r[i] = (ap[i] - 2.0 * aX[i] + am[i]) / (h * h);
  return r;
}

enum class Scheme { Euler, Taylor15, Splitting };

struct IntegrateConfig {
  double dt = 0.01;
  long n_steps = 100000;
  int n_paths = 200;
  long burn = 20000;
  long thin = 20;
  std::uint64_t seed = 0;
  Scheme scheme = Scheme::Splitting;
  // Distribution of the Wiener increments (K&P 1994):
  //   Gauss     — dW ~ N(0, dt); required by the strong schemes of Ch. 4.
  //   Telegraph — two-point dW = ±sqrt(dt), eq. (5.1.5): the simplified
  //               (weak order 1.0) scheme (5.1.6). With Taylor15 the dZ
  //               integral is replaced by dt*dW/2 as in Sec. 5.1; the result
  //               is only weakly convergent (p. 182: two-point variables are
  //               not sufficient even for weak order 2.0).
  NoiseKind noise = NoiseKind::Gauss;
  double blowup_abs = 1e12;  // sanity bound on |x| components
};

// Trajectory store: complex Y of shape (n_keep, n_paths, nvar), flattened.
struct Paths {
  long n_keep = 0;
  int n_paths = 0;
  int nvar = NVAR;         // active number of complex variables
  int order = ORDER;       // active number of photon modes
  std::vector<double> T;   // (n_keep)
  std::vector<cdouble> Y;  // (n_keep * n_paths * nvar), index ((k*P)+p)*nvar + j
  long n_diverged = 0;

  cdouble at(long k, int p, int j) const {
    return Y[(static_cast<std::size_t>(k) * n_paths + p) * nvar + j];
  }
};

// Integrate the FULL nonlinear SDE for n_paths independent trajectories.
//
// INITIAL CONDITION. Photons start at the deterministic steady state x0; the
// phonons start at their deterministic value PLUS an independent thermal
// fluctuation, Re/Im b_k ~ N(0, D0_k/Gamma_k) per path. Rationale: below
// threshold the deterministic fixed point has b_k = 0, so a cold start makes
// the phonon variance grow as sigma^2(t) = sigma_inf^2 (1 - exp(-Gamma t)).
// Averaging intensity statistics over that ramp samples a MIXTURE of Gaussians
// with different variances, for which
//     g2(0) = 2 <sigma^4> / <sigma^2>^2  >  2   strictly,
// i.e. a spuriously superthermal phonon g2 whenever burn and the recording
// window are not >> 1/Gamma. Seeding from the stationary law removes that bias
// at the source; for a decoupled phonon (e.g. b2 at --N-photons 2) the start is
// then EXACTLY stationary and g2 = 2 up to sampling error alone.
// The fluctuation is always drawn from a Gaussian, also under
// NoiseKind::Telegraph, because the target stationary law of the linear part is
// Gaussian regardless of how the increments are discretised.
inline Paths integrate_paths(double E, const State& x0, const Params& p,
                             const IntegrateConfig& cfg) {
  if (cfg.n_steps <= cfg.burn) throw std::invalid_argument("n_steps must exceed burn");
  if (cfg.thin <= 0) throw std::invalid_argument("thin must be positive");
  if (cfg.n_paths <= 0) throw std::invalid_argument("n_paths must be positive");

  const int dim = p.dim();
  const int nvar = p.nvar();
  const int m_noise = p.m_noise();
  const NoiseCols cols = noise_columns(p);
  const long n_keep = (cfg.n_steps - cfg.burn) / cfg.thin;

  Paths out;
  out.n_keep = n_keep;
  out.n_paths = cfg.n_paths;
  out.nvar = nvar;
  out.order = p.order;
  out.T.assign(n_keep, 0.0);
  out.Y.assign(static_cast<std::size_t>(n_keep) * cfg.n_paths * nvar, cdouble(0.0, 0.0));

  for (long k = 0; k < n_keep; ++k)
    out.T[k] = static_cast<double>(cfg.burn + k * cfg.thin + 1) * cfg.dt;

  // 1/2 * sum_j (b_j.grad)^2 a — constant for a quadratic drift, computed once.
  State L0_diff{};
  L0_diff.fill(0.0);
  {
    const State aX0 = drift(x0, E, p);
    for (int j = 0; j < m_noise; ++j) {
      const State s = dir_second(x0, cols[j], aX0, E, p);
      for (int i = 0; i < dim; ++i) L0_diff[i] += 0.5 * s[i];
    }
  }

  const double dt = cfg.dt;
  const double sqdt = std::sqrt(dt);
  const double inv_sqrt3 = 1.0 / std::sqrt(3.0);
  long n_diverged = 0;

  // --- Strang splitting precomputation: exact linear + OU flow over dt/2. ---
  // Each complex variable obeys dy/dt = -c y + f in the linear part:
  //   photons k:  c = gammas[k]/2,                    f = -iE (k = 0 only)
  //   first b:    c = gammas[b1]/2 - i*corr,          f = 0
  //   other b_k:  c = gammas[bk]/2,                   f = 0
  // Exact flow over h: y -> ec*y + fi, ec = exp(-c h), fi = f*(1-ec)/c.
  // Phonon noise over h is the exact OU transition: independent Gaussian on
  // each quadrature with variance D0*(1 - exp(-Gamma*h))/Gamma (no smallness
  // of Gamma*h or D0*h required).
  const double h_half = 0.5 * dt;
  std::array<cdouble, MAX_NVAR> lin_ec{};   // decay factors
  std::array<cdouble, MAX_NVAR> lin_fi{};   // inhomogeneous (pump) increments
  std::array<double, MAX_PHONONS> ou_sig{}; // per-quadrature noise std devs
  if (cfg.scheme == Scheme::Splitting) {
    for (int j = 0; j < nvar; ++j) {
      cdouble c = p.gammas[j] / 2.0;
      if (j == p.phon_index(0)) c -= cdouble(0.0, p.corr());
      const cdouble ec = std::exp(-c * h_half);
      lin_ec[j] = ec;
      cdouble f(0.0, 0.0);
      if (j == 0) f = cdouble(0.0, -E);
      lin_fi[j] = (std::abs(c) > 1e-300) ? f * (1.0 - ec) / c : f * h_half;
    }
    for (int k = 0; k < p.n_phon(); ++k) {
      const double G = p.gammas[p.phon_index(k)].real();
      const double var = (G > 1e-300)
                             ? p.D0[k] * (1.0 - std::exp(-G * h_half)) / G
                             : p.D0[k] * h_half;
      ou_sig[k] = std::sqrt(std::max(var, 0.0));
    }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : n_diverged)
#endif
  for (int ip = 0; ip < cfg.n_paths; ++ip) {
    // Independent, thread-safe stream per path; identical for any thread count.
    Xoshiro256pp rng(cfg.seed, static_cast<std::uint64_t>(ip));

    State X = x0;
    for (int k = 0; k < p.n_phon(); ++k) {
      const double sg = thermal_sigma(p, k);
      if (sg > 0.0) {
        const int jb = p.phon_index(k);
        set_cvar(X, nvar, jb,
                 cvar(X, nvar, jb) + cdouble(rng.normal() * sg, rng.normal() * sg));
      }
    }
    bool dead = false;
    long ki = 0;

    // Exact linear + OU half-step for the splitting scheme.
    auto lin_half = [&](State& x) {
      for (int j = 0; j < nvar; ++j)
        set_cvar(x, nvar, j, lin_ec[j] * cvar(x, nvar, j) + lin_fi[j]);
      for (int k = 0; k < p.n_phon(); ++k) {
        const int jb = p.phon_index(k);
        const cdouble eta(rng.draw(cfg.noise) * ou_sig[k],
                          rng.draw(cfg.noise) * ou_sig[k]);
        set_cvar(x, nvar, jb, cvar(x, nvar, jb) + eta);
      }
    };

    for (long n = 0; n < cfg.n_steps; ++n) {
      if (!dead) {
       if (cfg.scheme == Scheme::Splitting) {
        // Strang: exact(dt/2) o RK4_coupling(dt) o exact(dt/2). The stiff
        // linear decay and the additive noise are handled without any step
        // restriction; only the slow bilinear coupling limits dt.
        lin_half(X);
        {
          const State k1 = drift_coupling(X, p);
          const State k2 = drift_coupling(axpy(X, 0.5 * dt, k1, dim), p);
          const State k3 = drift_coupling(axpy(X, 0.5 * dt, k2, dim), p);
          const State k4 = drift_coupling(axpy(X, dt, k3, dim), p);
          for (int i = 0; i < dim; ++i)
            X[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        lin_half(X);
        if (!finite(X, dim) || max_abs(X, dim) > cfg.blowup_abs) dead = true;
       } else {
        const State aX = drift(X, E, p);
        State inc{};
        for (int i = 0; i < dim; ++i) inc[i] = aX[i] * dt;

        if (cfg.scheme == Scheme::Euler) {
          for (int j = 0; j < m_noise; ++j) {
            const double dW = rng.draw(cfg.noise) * sqdt;
            for (int i = 0; i < dim; ++i) inc[i] += cols[j][i] * dW;
          }
        } else {
          std::array<double, MAX_M_NOISE> dW{}, dZ{};
          if (cfg.noise == NoiseKind::Gauss) {
            // Correlated pair (dW, dZ): dZ = dt^{3/2}(u1 + u2/sqrt(3))/2, so
            // E dZ^2 = dt^3/3 and E dW dZ = dt^2/2 (K&P, Problem 1.3.5).
            for (int j = 0; j < m_noise; ++j) {
              const double u1 = rng.normal();
              const double u2 = rng.normal();
              dW[j] = u1 * sqdt;
              dZ[j] = 0.5 * dt * sqdt * (u1 + u2 * inv_sqrt3);
            }
          } else {
            // Simplified (weak) variant, Sec. 5.1: two-point dW = ±sqrt(dt),
            // multiple integral I(j,0) replaced by dt*dW/2.
            for (int j = 0; j < m_noise; ++j) {
              dW[j] = rng.telegraph() * sqdt;
              dZ[j] = 0.5 * dt * dW[j];
            }
          }
          for (int j = 0; j < m_noise; ++j)
            for (int i = 0; i < dim; ++i) inc[i] += cols[j][i] * dW[j];

          const State L0a = dir_deriv(X, aX, E, p);
          for (int i = 0; i < dim; ++i) inc[i] += 0.5 * dt * dt * (L0a[i] + L0_diff[i]);

          for (int j = 0; j < m_noise; ++j) {
            const State bg = dir_deriv(X, cols[j], E, p);
            for (int i = 0; i < dim; ++i) inc[i] += bg[i] * dZ[j];
          }
        }

        for (int i = 0; i < dim; ++i) X[i] += inc[i];

        if (!finite(X, dim) || max_abs(X, dim) > cfg.blowup_abs) dead = true;
       }
      }

      if (n >= cfg.burn && (n - cfg.burn) % cfg.thin == 0 && ki < n_keep) {
        const std::size_t base = (static_cast<std::size_t>(ki) * cfg.n_paths + ip) * nvar;
        for (int j = 0; j < nvar; ++j)
          out.Y[base + j] = dead ? cdouble(std::numeric_limits<double>::quiet_NaN(),
                                           std::numeric_limits<double>::quiet_NaN())
                                 : cvar(X, nvar, j);
        ++ki;
      }
    }
    if (dead) n_diverged += 1;
  }

  out.n_diverged = n_diverged;
  return out;
}

}  // namespace brillouin
