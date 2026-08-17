// model.hpp — Brillouin cascade: parameters, state layout, deterministic drift.
//
// The number of photon modes (ORDER) is now a RUNTIME parameter (Params::order),
// generalising the equations of motion of Cascade_Brillouin_scattering.tex,
// eq. (124)-(168), to any N. The number of phonon modes stays 2 (the paper's
// two-phonon model: b1 drives odd->even transitions, b2 drives even->odd).
//
// To keep the hot loops allocation-free, the real state vector has a fixed
// CAPACITY (MAX_DIM) but only the first `p.dim()` entries are active. At the
// default order 3 the arithmetic is bit-for-bit identical to the original code.
//
// Real state layout (dim = 2*nvar), mirroring the original Python code:
//   x = [Re y_0 .. Re y_{nvar-1}, Im y_0 .. Im y_{nvar-1}]
// with y = [a_1 .. a_ORDER, b_1, b_2]; x[k]=Re y[k], x[nvar+k]=Im y[k].
#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <vector>

namespace brillouin {

using cdouble = std::complex<double>;

// Compile-time capacities. ORDER/N_PHON below are the DEFAULTS (kept as named
// constants so existing call sites and the N=3 notebook keep working); the
// active sizes come from Params at run time.
inline constexpr int ORDER = 3;              // default photon modes a1..a3
inline constexpr int N_PHON = 2;             // phonon modes b1..b2 (fixed by the model)
inline constexpr int NVAR = ORDER + N_PHON;  // default 5 complex variables
inline constexpr int DIM = 2 * NVAR;         // default 10 real variables
inline constexpr int M_NOISE = 2 * N_PHON;   // 4 independent Wiener processes

// Runtime capacity: supports up to MAX_ORDER photon modes without heap use.
inline constexpr int MAX_ORDER = 64;
inline constexpr int MAX_NVAR = MAX_ORDER + N_PHON;
inline constexpr int MAX_DIM = 2 * MAX_NVAR;

// Fixed-capacity real state. Only the first `dim` entries are meaningful, but
// the whole array is value-initialised so the unused tail stays a harmless 0.
using State = std::array<double, MAX_DIM>;

// Physical parameters. Defaults reproduce the notebook's hard-coded N=3 values.
struct Params {
  int order = ORDER;  // number of photon modes (runtime)

  double omega_0 = 1e5;
  double alpha = 1e-7 * 1e5;        // ALPHA  (g1)
  double beta = 1e-7 * 1e5;         // BETA   (g2)
  double omega_shift = 8e-5 * 1e5;  // OMEGA (detuning correction scale)

  // Photon decay rates gamma_j (complex; Im part feeds the detuning correction)
  // followed by the two phonon decay rates Gamma_1, Gamma_2 at indices
  // [order], [order+1]. Sized to the capacity; only the active head is used.
  std::array<cdouble, MAX_NVAR> gammas{};

  std::array<double, N_PHON> D0{};  // phonon diffusion intensities

  Params() { init_defaults(order); }

  int nvar() const { return order + N_PHON; }
  int dim() const { return 2 * nvar(); }
  int m_noise() const { return 2 * N_PHON; }

  // Index of phonon k (0-based) inside the complex vector y.
  int phon_index(int k) const { return order + k; }

  // (Re)initialise gammas/D0 for a given photon count, preserving the original
  // per-mode default values (photons 1e-6*1e5, phonons 1e-7*1e5).
  void init_defaults(int new_order) { 
    order = new_order; 
    
    for (auto& g : gammas) 
        g = cdouble(0.0, 0.0); 
        
    // Первый цикл: каждый следующий элемент на 10% больше предыдущего
    for (int k = 0; k < order; ++k) {
        double base_val = 1e-6 * 1e5;
        gammas[k] = cdouble(base_val * std::pow(1.1, k), 0.0);
    }
    
    // Второй цикл: поправка +10% отсчитывается заново от 0 до N_PHON
    for (int k = 0; k < N_PHON; ++k) {
        double base_val = 1e-7 * 1e5;
        gammas[order + k] = cdouble(base_val * std::pow(1.1, k), 0.0);
    }
    
    set_default_D0(); 
}


  // D0_PHONON = ones(2) * Re(Gamma) * 1e-3 * OMEGA_0 / OMEGA * 3 * 2   (legacy).
  void set_default_D0() {
    const double v =
        gammas[phon_index(0)].real() * 1e-3 * omega_0 / omega_shift * 3.0 * 2;
    D0[0] = v;
    D0[1] = v;
  }

  // Detuning correction on rho_1: (2*Omega*g0i + g0i^2) / (2*(Omega + g0i)).
  double corr() const {
    const double g0i = gammas[0].imag();
    return (2.0 * omega_shift * g0i + g0i * g0i) / (2.0 * (omega_shift + g0i));
  }
};

inline cdouble cvar(const State& x, int nvar, int k) {
  return cdouble(x[k], x[nvar + k]);
}
inline void set_cvar(State& x, int nvar, int k, cdouble v) {
  x[k] = v.real();
  x[nvar + k] = v.imag();
}

// Backwards-compatible fixed-order overloads (default N=3 layout, nvar=NVAR).
inline cdouble cvar(const State& x, int k) { return cvar(x, NVAR, k); }
inline void set_cvar(State& x, int k, cdouble v) { set_cvar(x, NVAR, k, v); }

// Deterministic drift a(x) for arbitrary photon count. At order 3 this is
// bit-for-bit the original algebra (verified numerically).
//
// Photon k is 0-based; the paper's mode index is j = k+1.
//   a_1 : i g1 b1 a_2                      - g_1/2 a_1 - i E
//   even j (0-based odd k): i g2 b2 a_{j+1} + i g1 b1* a_{j-1}  - g_j/2 a_j
//   odd  j>1 (0-based even k): i g2 b2* a_{j-1} + i g1 b1 a_{j+1} - g_j/2 a_j
//   last mode drops whichever neighbour does not exist.
// Phonons (code convention a_j * conj(a_{j+1}), matching the original N=3 code):
//   b1 : i g1 sum_{odd j}  a_j conj(a_{j+1}) - G1/2 b1 + i corr b1
//   b2 : i g2 sum_{even j} a_j conj(a_{j+1}) - G2/2 b2
inline State drift(const State& x, double E, const Params& p) {
  const cdouble I(0.0, 1.0);
  const int O = p.order;
  const int nv = p.nvar();
  const auto& g = p.gammas;
  const int i_b1 = p.phon_index(0), i_b2 = p.phon_index(1);
  const cdouble r1 = cvar(x, nv, i_b1), r2 = cvar(x, nv, i_b2);

  State d{};  // zero-initialised (unused tail stays 0)

  for (int k = 0; k < O; ++k) {
    const int j = k + 1;  // 1-based paper index
    cdouble acc(0.0, 0.0);
    if (j == 1) {
      if (O >= 2) acc += I * p.alpha * r1 * cvar(x, nv, 1);
      acc += -I * E;
    } else if (j % 2 == 0) {  // even j: forward Stokes fed by b2, drained to b1
      if (k + 1 < O) acc += I * p.alpha * r2 * cvar(x, nv, k + 1);
      acc += I * p.alpha * std::conj(r1) * cvar(x, nv, k - 1);
    } else {  // odd j > 1
      acc += I * p.alpha * std::conj(r2) * cvar(x, nv, k - 1);
      if (k + 1 < O) acc += I * p.alpha * r1 * cvar(x, nv, k + 1);
    }
    acc += -g[k] * cvar(x, nv, k) / 2.0;
    set_cvar(d, nv, k, acc);
  }

  // Phonon sources. b1 over odd j (0-based even k), b2 over even j (0-based odd k).
  cdouble s1(0.0, 0.0), s2(0.0, 0.0);
  for (int k = 0; k + 1 < O; k += 2)
    s1 += cvar(x, nv, k) * std::conj(cvar(x, nv, k + 1));
  for (int k = 1; k + 1 < O; k += 2)
    s2 += cvar(x, nv, k) * std::conj(cvar(x, nv, k + 1));

  const cdouble dr1 = I * p.beta * s1 - g[i_b1] * r1 / 2.0 + r1 * I * p.corr();
  const cdouble dr2 = I * p.beta * s2 - g[i_b2] * r2 / 2.0;
  set_cvar(d, nv, i_b1, dr1);
  set_cvar(d, nv, i_b2, dr2);
  return d;
}

// Coupling-only drift: the bilinear interaction terms of drift() with the
// linear part (decay, pump, detuning correction) removed. Used by the
// exact-OU splitting integrator, where the linear+noise flow is applied in
// closed form and only this non-stiff part is stepped numerically.
inline State drift_coupling(const State& x, const Params& p) {
  const cdouble I(0.0, 1.0);
  const int O = p.order;
  const int nv = p.nvar();
  const int i_b1 = p.phon_index(0), i_b2 = p.phon_index(1);
  const cdouble r1 = cvar(x, nv, i_b1), r2 = cvar(x, nv, i_b2);

  State d{};
  for (int k = 0; k < O; ++k) {
    const int j = k + 1;
    cdouble acc(0.0, 0.0);
    if (j == 1) {
      if (O >= 2) acc += I * p.alpha * r1 * cvar(x, nv, 1);
    } else if (j % 2 == 0) {
      if (k + 1 < O) acc += I * p.alpha * r2 * cvar(x, nv, k + 1);
      acc += I * p.alpha * std::conj(r1) * cvar(x, nv, k - 1);
    } else {
      acc += I * p.alpha * std::conj(r2) * cvar(x, nv, k - 1);
      if (k + 1 < O) acc += I * p.alpha * r1 * cvar(x, nv, k + 1);
    }
    set_cvar(d, nv, k, acc);
  }
  cdouble s1(0.0, 0.0), s2(0.0, 0.0);
  for (int k = 0; k + 1 < O; k += 2)
    s1 += cvar(x, nv, k) * std::conj(cvar(x, nv, k + 1));
  for (int k = 1; k + 1 < O; k += 2)
    s2 += cvar(x, nv, k) * std::conj(cvar(x, nv, k + 1));
  set_cvar(d, nv, i_b1, I * p.beta * s1);
  set_cvar(d, nv, i_b2, I * p.beta * s2);
  return d;
}

// Constant additive diffusion matrix B (dim x m_noise), columns = Wiener procs.
// Noise enters phonons only: (Re b1, Im b1, Re b2, Im b2) with intensity sqrt(D0k).
using NoiseCols = std::array<State, M_NOISE>;

inline NoiseCols noise_columns(const Params& p) {
  const int nv = p.nvar();
  NoiseCols cols{};
  for (auto& c : cols) c.fill(0.0);
  for (int k = 0; k < N_PHON; ++k) {
    const int re_idx = p.phon_index(k);        // Re b_k
    const int im_idx = nv + p.phon_index(k);   // Im b_k
    const double s = std::sqrt(p.D0[k]);
    cols[2 * k][re_idx] = s;
    cols[2 * k + 1][im_idx] = s;
  }
  return cols;
}

// Stationary per-quadrature standard deviation of phonon k under its own linear
// OU dynamics: d(Re b) = -(Gamma/2) Re b dt + sqrt(D0) dW  =>  Var = D0/Gamma,
// so <|b_k|^2> = 2*D0/Gamma. With the CLI mapping D0 = Gamma*nth/2 this is
// <|b_k|^2> = nth. Used to start trajectories from a THERMALISED phonon bath
// instead of the cold deterministic fixed point (see integrate_paths).
inline double thermal_sigma(const Params& p, int k) {
  const double G = p.gammas[p.phon_index(k)].real();
  if (!(G > 0) || !(p.D0[k] > 0)) return 0.0;
  return std::sqrt(p.D0[k] / G);
}

// --- small vector helpers (operate on the active head [0,dim)) ---------------
inline State axpy(const State& x, double c, const State& v, int dim) {
  State r{};
  for (int i = 0; i < dim; ++i) r[i] = x[i] + c * v[i];
  return r;
}
inline bool finite(const State& x, int dim) {
  for (int i = 0; i < dim; ++i)
    if (!std::isfinite(x[i])) return false;
  return true;
}
inline double max_abs(const State& x, int dim) {
  double m = 0.0;
  for (int i = 0; i < dim; ++i) m = std::max(m, std::abs(x[i]));
  return m;
}

// Backwards-compatible fixed-DIM overloads (default N=3).
inline State axpy(const State& x, double c, const State& v) {
  return axpy(x, c, v, DIM);
}
inline bool finite(const State& x) { return finite(x, DIM); }
inline double max_abs(const State& x) { return max_abs(x, DIM); }

}  // namespace brillouin
