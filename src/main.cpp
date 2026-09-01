// main.cpp — CLI driver: pump sweep of the full nonlinear SDE + linear reference,
// writing a single JSON file for the Python notebook to visualise.
//
// Usage example:
//   ./sde_solver --E-min 0 --E-max 10 --nE 10 --dt 0.01 --n-steps 100000
//                --n-paths 200 --burn 20000 --thin 20 --seed 0 --out sde_sweep.json
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "estimators.hpp"
#include "linear_theory.hpp"
#include "model.hpp"
#include "solver.hpp"

using namespace brillouin;

namespace {

struct Options {
  double E_min = 0.0, E_max = 10.0;
  int nE = 10;
  std::vector<double> E_list;  // overrides the grid if non-empty
  IntegrateConfig cfg;
  std::string out = "sde_sweep.json";
  bool linear_only = false;
  bool quiet = false;
  bool verbose = false;   // per-phase timing markers on stderr
  int threads = 0;  // 0 = library default

  int N_photons = ORDER;    // number of photon modes (default 3)

  // Spectrum output (Lorentzian lines from D_eff, eq. (29),(60)-(62)).
  bool spectrum = false;    // emit per-E field spectrum built from FWHM + amplitudes
  int spec_points = 2000;   // frequency samples per spectrum
  double spec_span = -1.0;  // half-width in units of Omega; <0 => auto ((N+1)*Omega)

  // Physical-parameter overrides (code units). Negative = keep the defaults.
  double g = -1.0;          // sets alpha = beta = g
  double Gamma = -1.0;      // sets Re(gammas[phonons]) = Gamma
  double gamma_opt = -1.0;  // sets Re(gammas[photons]) = gamma_opt
  double nth = -1.0;        // thermal occupancy: D0[k] = Gamma_k * nth / 2,
                            // chosen so that <|b_k|^2> = nth exactly (see below)
};

[[noreturn]] void usage(int code) {
  std::cout <<
      R"(sde_solver — Brillouin cascade nonlinear SDE solver.

Options:
  --E-min <f>        lowest pump value                 (default 0)
  --E-max <f>        highest pump value                (default 10)
  --nE <int>         number of pump grid points        (default 10)
  --E-list a,b,c     explicit pump values (overrides the grid above)
  --N-photons <int>  number of photon modes N          (default 3)
  --dt <f>           integration step                  (default 0.01)
  --n-steps <int>    number of steps                   (default 100000)
  --n-paths <int>    independent trajectories          (default 200)
  --burn <int>       discarded transient, in steps     (default 20000)
  --thin <int>       record every k-th step            (default 20)
  --seed <int>       base RNG seed                     (default 0)
  --scheme <s>       splitting | taylor15 | euler      (default splitting)
                     splitting = Strang: exact OU flow for decay+noise+pump,
                     RK4 for the bilinear coupling; no Gamma*dt / D*dt
                     step restriction — use for strongly rescaled units
  --noise <s>        gauss | telegraph                 (default gauss)
                     telegraph = two-point dW = ±sqrt(dt) (K&P 1994, eq. 5.1.5);
                     weak order 1.0 only — pair it with --scheme euler
  --threads <int>    OpenMP threads (0 = default)
  --g <f>            coupling: alpha = beta = g        (default 1e-2)
  --Gamma <f>        common decay of every phonon mode (default 1e-2)
  --gamma-opt <f>    photon decay gamma_j (all modes)  (default 1e-1)
  --nth <f>          thermal phonon occupancy: <|b_k|^2> = nth exactly,
                     i.e. D0 = Gamma*nth/2           (default: legacy D0)
  --spectrum         emit Lorentzian field spectra S(w) from D_eff, eq.(29)
  --spec-points <int> frequency samples per spectrum   (default 2000)
  --spec-span <f>    spectrum half-width in units OMEGA (default auto)
  --linear-only      skip the SDE, compute the Lyapunov reference only (N=3)
  --out <path>       output JSON file                  (default sde_sweep.json)
  --verbose          per-phase timing on stderr (steady state / integration /
                     each estimator). Use this when a run seems to hang: the
                     markers show which phase owns the time.
  --quiet            suppress progress output
  -h, --help         this message

The linear (Lyapunov) reference is only defined for N=3; for other N the
lw_lin / g2_lin fields are null and the FWHM used for spectra comes from the
nonlinear SDE (phase-MSD estimator).
)";
  std::exit(code);
}

double need_double(const char* s, const char* flag) {
  char* end = nullptr;
  const double v = std::strtod(s, &end);
  if (end == s || *end != '\0') {
    std::cerr << "error: " << flag << " expects a number, got '" << s << "'\n";
    std::exit(2);
  }
  return v;
}

long need_long(const char* s, const char* flag) {
  char* end = nullptr;
  const long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0') {
    std::cerr << "error: " << flag << " expects an integer, got '" << s << "'\n";
    std::exit(2);
  }
  return v;
}

Options parse(int argc, char** argv) {
  Options o;
  auto next = [&](int& i, const char* flag) -> const char* {
    if (i + 1 >= argc) {
      std::cerr << "error: " << flag << " requires a value\n";
      std::exit(2);
    }
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") usage(0);
    else if (a == "--E-min") o.E_min = need_double(next(i, "--E-min"), "--E-min");
    else if (a == "--E-max") o.E_max = need_double(next(i, "--E-max"), "--E-max");
    else if (a == "--nE") o.nE = static_cast<int>(need_long(next(i, "--nE"), "--nE"));
    else if (a == "--E-list") {
      std::stringstream ss(next(i, "--E-list"));
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) o.E_list.push_back(need_double(tok.c_str(), "--E-list"));
      }
    }
    else if (a == "--dt") o.cfg.dt = need_double(next(i, "--dt"), "--dt");
    else if (a == "--n-steps") o.cfg.n_steps = need_long(next(i, "--n-steps"), "--n-steps");
    else if (a == "--n-paths") o.cfg.n_paths = static_cast<int>(need_long(next(i, "--n-paths"), "--n-paths"));
    else if (a == "--burn") o.cfg.burn = need_long(next(i, "--burn"), "--burn");
    else if (a == "--thin") o.cfg.thin = need_long(next(i, "--thin"), "--thin");
    else if (a == "--seed") o.cfg.seed = static_cast<std::uint64_t>(need_long(next(i, "--seed"), "--seed"));
    else if (a == "--scheme") {
      const std::string s = next(i, "--scheme");
      if (s == "euler") o.cfg.scheme = Scheme::Euler;
      else if (s == "taylor15") o.cfg.scheme = Scheme::Taylor15;
      else if (s == "splitting") o.cfg.scheme = Scheme::Splitting;
      else { std::cerr << "error: unknown scheme '" << s << "'\n"; std::exit(2); }
    }
    else if (a == "--noise") {
      const std::string s = next(i, "--noise");
      if (s == "gauss") o.cfg.noise = NoiseKind::Gauss;
      else if (s == "telegraph") o.cfg.noise = NoiseKind::Telegraph;
      else { std::cerr << "error: unknown noise kind '" << s << "'\n"; std::exit(2); }
    }
    else if (a == "--N-photons") o.N_photons = static_cast<int>(need_long(next(i, "--N-photons"), "--N-photons"));
    else if (a == "--spectrum") o.spectrum = true;
    else if (a == "--spec-points") o.spec_points = static_cast<int>(need_long(next(i, "--spec-points"), "--spec-points"));
    else if (a == "--spec-span") o.spec_span = need_double(next(i, "--spec-span"), "--spec-span");
    else if (a == "--threads") o.threads = static_cast<int>(need_long(next(i, "--threads"), "--threads"));
    else if (a == "--g") o.g = need_double(next(i, "--g"), "--g");
    else if (a == "--Gamma") o.Gamma = need_double(next(i, "--Gamma"), "--Gamma");
    else if (a == "--gamma-opt") o.gamma_opt = need_double(next(i, "--gamma-opt"), "--gamma-opt");
    else if (a == "--nth") o.nth = need_double(next(i, "--nth"), "--nth");
    else if (a == "--linear-only") o.linear_only = true;
    else if (a == "--out") o.out = next(i, "--out");
    else if (a == "--quiet") o.quiet = true;
    else if (a == "--verbose") o.verbose = true;
    else { std::cerr << "error: unknown option '" << a << "'\n"; usage(2); }
  }

  // Validation — fail fast and loudly rather than producing garbage.
  if (o.E_list.empty()) {
    if (o.nE < 1) { std::cerr << "error: --nE must be >= 1\n"; std::exit(2); }
    o.E_list.resize(o.nE);
    for (int i = 0; i < o.nE; ++i)
      o.E_list[i] = (o.nE == 1) ? o.E_min
                                : o.E_min + (o.E_max - o.E_min) * i / (o.nE - 1);
  }
  if (o.N_photons < 1 || o.N_photons > MAX_ORDER) {
    std::cerr << "error: --N-photons must be in [1, " << MAX_ORDER << "]\n";
    std::exit(2);
  }
  if (USE_PAIRWISE_PHONONS && o.N_photons < 2) {
    std::cerr << "error: pairwise-phonon model requires --N-photons >= 2\n";
    std::exit(2);
  }
  if (o.spec_points < 2) { std::cerr << "error: --spec-points must be >= 2\n"; std::exit(2); }
  if (!(o.cfg.dt > 0)) { std::cerr << "error: --dt must be > 0\n"; std::exit(2); }
  if (o.cfg.n_steps <= o.cfg.burn) { std::cerr << "error: --n-steps must exceed --burn\n"; std::exit(2); }
  if (o.cfg.thin <= 0) { std::cerr << "error: --thin must be > 0\n"; std::exit(2); }
  if (o.cfg.n_paths <= 0) { std::cerr << "error: --n-paths must be > 0\n"; std::exit(2); }
  if ((o.cfg.n_steps - o.cfg.burn) / o.cfg.thin < 20) {
    std::cerr << "error: too few recorded samples ((n_steps-burn)/thin < 20)\n";
    std::exit(2);
  }
  if (o.cfg.noise == NoiseKind::Telegraph && o.cfg.scheme != Scheme::Euler && !o.quiet)
    std::cerr << "warning: --noise telegraph is admissible only under the WEAK\n"
                 "         convergence criterion (K&P 1994 Sec. 5.1, p. 182). With\n"
                 "         --scheme taylor15 the dZ integral degrades to dt*dW/2; with\n"
                 "         --scheme splitting the exact OU transition law is Gaussian,\n"
                 "         so two-point increments only match its first two moments\n"
                 "         and Gaussianity is recovered by the CLT over ~1/(Gamma*dt)\n"
                 "         steps. Pair telegraph with --scheme euler.\n";
  return o;
}

// --- minimal JSON emission --------------------------------------------------
std::string jnum(double v) {
  if (!std::isfinite(v)) return "null";  // JSON has no NaN/Inf; null round-trips to NaN in pandas/numpy
  std::ostringstream ss;
  ss << std::setprecision(17) << v;
  return ss.str();
}

std::string jarr(const std::vector<double>& v) {
  std::ostringstream ss;
  ss << "[";
  for (std::size_t i = 0; i < v.size(); ++i) ss << (i ? ", " : "") << jnum(v[i]);
  ss << "]";
  return ss.str();
}

template <std::size_t N>
std::string jarr(const std::array<double, N>& a) {
  return jarr(std::vector<double>(a.begin(), a.end()));
}

// Build the Stokes field spectrum S(w) = sum_j S_Ej(w) from the per-mode
// effective linewidths (FWHM = D_eff_j) and steady amplitudes, eq. (29):
//   S_Ej(w) = A_j^2 * (Deff_j/2) / [ (w - w_j)^2 + (Deff_j/4)^2 ],
// with line centres w_j = w_1 - (j-1)*Omega placed at the Brillouin shift Omega.
// A phase-locked mode (Deff -> 0, e.g. the pump) is drawn as a narrow line whose
// width floor is a small fraction of Omega so it remains visible.
struct Spectrum {
  std::vector<double> w;  // detuning axis (units: same as Omega/code units)
  std::vector<double> S;  // total spectral power density
};

inline Spectrum build_spectrum(const std::vector<double>& A,
                               const std::vector<double>& fwhm, int order,
                               double Omega, int npts, double span_in_Omega) {
  Spectrum sp;
  const int N = order;
  const double span = (span_in_Omega > 0 ? span_in_Omega : (N + 1)) * Omega;
  // Detuning measured from the pump line w_1 (so w_1 sits at 0, lower orders < 0).
  const double w_lo = -span;
  const double w_hi = 0.2 * Omega + 1e-30;
  sp.w.resize(npts);
  sp.S.assign(npts, 0.0);
  for (int i = 0; i < npts; ++i)
    sp.w[i] = w_lo + (w_hi - w_lo) * i / (npts - 1);

  // Width floor for locked lines: 0.5% of the mode spacing.
  const double gmin = 5e-3 * Omega;
  for (int j = 0; j < N; ++j) {
    const double Aj = A[j];
    if (!std::isfinite(Aj) || Aj <= 0) continue;
    double G = (j < static_cast<int>(fwhm.size())) ? fwhm[j] : 0.0;
    if (!std::isfinite(G) || G < gmin) G = gmin;  // FWHM floor
    const double wj = -static_cast<double>(j) * Omega;  // w_1 - (j)*Omega, 0-based
    const double half = G / 2.0;         // Deff/2 numerator weight
    const double hwhm2 = (G / 4.0) * (G / 4.0);
    const double weight = Aj * Aj;
    for (int i = 0; i < npts; ++i) {
      const double dw = sp.w[i] - wj;
      sp.S[i] += weight * half / (dw * dw + hwhm2);
    }
  }
  return sp;
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  Params p;
  p.init_defaults(o.N_photons);   // set active photon count + default gammas/D0
  const int OO = p.order;         // active number of photon modes
  const int NB = p.n_phon();      // 2 shared phonons or N-1 pairwise phonons
  if (o.g > 0) { p.alpha = o.g; p.beta = o.g; }
  if (o.gamma_opt > 0)
    for (int k = 0; k < OO; ++k) p.gammas[k] = cdouble(o.gamma_opt, p.gammas[k].imag());
  if (o.Gamma > 0)
    for (int k = 0; k < NB; ++k)
      p.gammas[p.phon_index(k)] = cdouble(o.Gamma, p.gammas[p.phon_index(k)].imag());
  if (o.nth >= 0) {  // >=: --nth 0 (zero temperature, D0 = 0) is legitimate
    for (int k = 0; k < NB; ++k)
      // Per-quadrature OU: d(Re b) = -(Gamma/2) Re b dt + sqrt(D0) dW, hence
      // Var(Re b) = D0/Gamma and <|b|^2> = 2*D0/Gamma. Setting <|b|^2> = nth
      // gives D0 = Gamma*nth/2. (The old mapping D0 = 2*Gamma*nth produced
      // <|b|^2> = 4*nth, i.e. an occupation four times the nominal one.)
      p.D0[k] = 0.5 * p.gammas[p.phon_index(k)].real() * o.nth;
  } else if (o.Gamma > 0) {
    p.set_default_D0();  // legacy D0 scales with the (new) phonon decay rate
  }

  // The Lyapunov linear reference (linear_theory.hpp) is hard-wired to N=3.
  const bool have_linear = !USE_PAIRWISE_PHONONS && (OO == ORDER);
  if (o.linear_only && !have_linear) {
    std::cerr << "error: --linear-only is available only for the shared-two "
                 "model with --N-photons 3\n";
    return 2;
  }

#ifdef _OPENMP
  if (o.threads > 0) omp_set_num_threads(o.threads);
#endif

  // Write to a temporary file and rename it into place only once the document
  // is complete. Two problems this solves:
  //   * a run that is killed, crashes or stalls used to leave a TRUNCATED JSON
  //     behind, because the whole document was buffered in memory and written
  //     in one go at the very end. Downstream that file looks like a valid
  //     cache entry until something tries to parse it.
  //   * with the rename, the final path is either absent or a complete
  //     document — never half of one.
  // Records are also streamed and flushed as they are computed, so the file
  // grows visibly during a long sweep: if it grows, the writer is alive.
  const std::string tmp_out = o.out + ".partial";
  std::ofstream f(tmp_out);
  if (!f) {
    std::cerr << "error: cannot open output file '" << tmp_out << "'\n";
    return 1;
  }

  const auto t_start = std::chrono::steady_clock::now();
  // "results" comes FIRST so records can be streamed; JSON object member order
  // is not significant, and the readers access sections by key.
  f << "{\n \"results\": [\n";
  f.flush();  // so the file is non-empty (and visibly opened) from the start
  std::size_t n_written = 0;

  // Phase markers. A long pump point has three very different phases (RK45
  // steady state, the parallel SDE integration, the SERIAL estimators), and
  // without markers a stall in any of them looks identical from outside. Each
  // marker carries the seconds spent in the previous phase, so a run that
  // "hangs" tells you immediately which phase owns the time. Written to stderr
  // and flushed, so nothing is lost to buffering if the process is killed.
  const bool trace = o.verbose && !o.quiet;
  auto t_phase = std::chrono::steady_clock::now();
  auto mark = [&](const char* what) {
    if (!trace) return;
    const double el =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_phase).count();
    std::cerr << "      [" << what << " " << std::fixed << std::setprecision(2) << el
              << "s]" << std::endl;  // endl: flush, this is a diagnostic
    t_phase = std::chrono::steady_clock::now();
  };

  for (std::size_t i = 0; i < o.E_list.size(); ++i) {
    const double E = o.E_list[i];
    const auto t0 = std::chrono::steady_clock::now();
    t_phase = t0;
    if (trace) std::cerr << "  E = " << E << ": steady state ..." << std::endl;

    // 1) deterministic steady state
    const SteadyResult st = steady_state(E, p);
    mark("steady");
    if (trace && !st.converged)
      std::cerr << "      warning: steady state not converged (residual "
                << st.residual << ", " << st.n_steps << " RK45 steps)" << std::endl;
    std::vector<double> A(OO, 0.0), B(NB, 0.0);
    for (int k = 0; k < OO; ++k) A[k] = std::abs(cvar(st.x, p.nvar(), k));
    for (int k = 0; k < NB; ++k)
      B[k] = std::abs(cvar(st.x, p.nvar(), p.phon_index(k)));

    // 2) linear (Lyapunov) reference — only defined for N=3.
    std::vector<double> lw_li(OO, std::nan("")), g2_li(OO, std::nan(""));
    if (have_linear) {
      Vec3 A3{A[0], A[1], A[2]};
      Vec2 B2{B[0], B[1]};
      const std::array<double, 2> D02{p.D0[0], p.D0[1]};
      const Vec3 lw = lw_linear(A3, B2, D02);
      const Vec3 gam{p.gammas[0].real(), p.gammas[1].real(), p.gammas[2].real()};
      const Vec2 Gam{p.gammas[3].real(), p.gammas[4].real()};
      Vec3 g2l = g2_linear_minus_one(A3, B2, gam, Gam, p.alpha, p.beta, D02);
      for (int k = 0; k < ORDER; ++k) { lw_li[k] = lw[k]; g2_li[k] = g2l[k] + 1.0; }
    }

    // 3) full nonlinear SDE
    std::vector<double> fwhm_msd(OO, std::nan("")), fwhm_g1(OO, std::nan("")),
        r2_msd(OO, std::nan("")), r2_g1(OO, std::nan("")),
        g2_0(OO, std::nan("")), A_mean(OO, std::nan("")),
        B_mean(NB, std::nan("")),
        g2_0_phon(NB, std::nan(""));
    long n_div = 0;
    long n_keep = 0;

    if (!o.linear_only) {
      if (trace)
        std::cerr << "      integrating " << o.cfg.n_steps << " steps x "
                  << o.cfg.n_paths << " paths ..." << std::endl;
      const Paths P = integrate_paths(E, st.x, p, o.cfg);
      mark("integrate");
      n_div = P.n_diverged;
      n_keep = P.n_keep;
      if (trace)
        std::cerr << "      estimators on " << P.n_keep << " samples/path"
                  << (P.n_diverged ? " (" + std::to_string(P.n_diverged) +
                                         " paths diverged)"
                                   : "")
                  << " ..." << std::endl;
      for (int j = 0; j < OO; ++j) {
        const FitResult m = D_from_msd(P, j);
        mark("msd");
        const FitResult g = D_from_g1(P, j);
        mark("g1");
        fwhm_msd[j] = m.value;
        r2_msd[j] = m.r2;
        fwhm_g1[j] = g.value;
        r2_g1[j] = g.r2;
        g2_0[j] = g2_zero(P, j);
        A_mean[j] = mean_abs(P, j);
      }
      for (int k = 0; k < NB; ++k) {
        B_mean[k] = mean_abs(P, p.phon_index(k));
        g2_0_phon[k] = g2_zero(P, p.phon_index(k));  // phonon intensity g2(0)
      }
      mark("phonon stats");
    }

    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::ostringstream r;
    r << "  {\n"
      << "   \"E\": " << jnum(E) << ",\n"
      << "   \"A_det\": " << jarr(A) << ",\n"
      << "   \"B_det\": " << jarr(B) << ",\n"
      << "   \"A_mean\": " << jarr(A_mean) << ",\n"
      << "   \"B_mean\": " << jarr(B_mean) << ",\n"
      << "   \"fwhm_msd\": " << jarr(fwhm_msd) << ",\n"
      << "   \"fwhm_g1\": " << jarr(fwhm_g1) << ",\n"
      << "   \"r2_msd\": " << jarr(r2_msd) << ",\n"
      << "   \"r2_g1\": " << jarr(r2_g1) << ",\n"
      << "   \"g2_0\": " << jarr(g2_0) << ",\n"
      << "   \"g2_0_phonon\": " << jarr(g2_0_phon) << ",\n"
      << "   \"lw_lin\": " << jarr(lw_li) << ",\n"
      << "   \"g2_lin\": " << jarr(g2_li) << ",\n"
      << "   \"steady_converged\": " << (st.converged ? "true" : "false") << ",\n"
      << "   \"steady_residual\": " << jnum(st.residual) << ",\n"
      << "   \"n_diverged\": " << n_div << ",\n"
      << "   \"n_keep\": " << n_keep << ",\n";

    // Optional: Lorentzian field spectrum from D_eff (eq. 29). Prefer the
    // nonlinear MSD linewidth; fall back to the linear one where MSD is absent.
    if (o.spectrum) {
      std::vector<double> width(OO, std::nan(""));
      for (int j = 0; j < OO; ++j) {
        double g = fwhm_msd[j];
        if (!std::isfinite(g)) g = (j < (int)lw_li.size()) ? lw_li[j] : std::nan("");
        width[j] = g;
      }
      const Spectrum sp = build_spectrum(A, width, OO, p.omega_shift,
                                         o.spec_points, o.spec_span);
      r << "   \"spectrum_w\": " << jarr(sp.w) << ",\n"
        << "   \"spectrum_S\": " << jarr(sp.S) << ",\n";
    }

    r << "   \"walltime\": " << jnum(wall) << "\n"
      << "  }";
    // Stream this record and flush: on a long sweep the partial file is the
    // proof that the writer is making progress.
    if (n_written++) f << ",\n";
    f << r.str();
    f.flush();
    if (!f) {
      std::cerr << "error: failed while writing '" << tmp_out << "' (disk full, "
                   "permissions, or the file is locked by another process such as "
                   "an antivirus scanner or a cloud-sync client)\n";
      return 1;
    }

    if (!o.quiet) {
      // Show up to the first three modes (compactly) regardless of N.
      auto g = [&](const std::vector<double>& v, int k) {
        return k < (int)v.size() ? v[k] : std::nan("");
      };
      std::printf("[%zu/%zu] N=%d E=%6.3f  A=(%7.3f,%6.3f,%7.3f..)  "
                  "g2=(%.5f,%.5f,%.5f..)  FWHM=(%.2e,%.2e,%.2e..)%s  [%.1fs]\n",
                  i + 1, o.E_list.size(), OO, E, g(A, 0), g(A, 1), g(A, 2),
                  g(g2_0, 0), g(g2_0, 1), g(g2_0, 2),
                  g(fwhm_msd, 0), g(fwhm_msd, 1), g(fwhm_msd, 2),
                  (n_div ? "  !DIVERGED PATHS" : ""), wall);
      std::fflush(stdout);
    }
    if (!st.converged)
      std::cerr << "warning: steady state not converged at E=" << E
                << " (residual " << st.residual << ")\n";
    if (n_div)
      std::cerr << "warning: " << n_div << "/" << o.cfg.n_paths
                << " paths diverged at E=" << E << " — reduce --dt\n";
  }

  const double total =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

  // Close the streamed results array, then the metadata sections.
  f << "\n ],\n"
    << " \"meta\": {\n"
    << "  \"solver\": \"cpp\",\n"
    << "  \"scheme\": \"" << (o.cfg.scheme == Scheme::Euler ? "euler"
                       : o.cfg.scheme == Scheme::Splitting ? "splitting" : "taylor15") << "\",\n"
    << "  \"noise\": \"" << (o.cfg.noise == NoiseKind::Telegraph ? "telegraph" : "gauss") << "\",\n"
    << "  \"dt\": " << jnum(o.cfg.dt) << ",\n"
    << "  \"n_steps\": " << o.cfg.n_steps << ",\n"
    << "  \"n_paths\": " << o.cfg.n_paths << ",\n"
    << "  \"burn\": " << o.cfg.burn << ",\n"
    << "  \"thin\": " << o.cfg.thin << ",\n"
    << "  \"seed\": " << o.cfg.seed << ",\n"
    << "  \"linear_only\": " << (o.linear_only ? "true" : "false") << ",\n"
    << "  \"has_linear\": " << (have_linear ? "true" : "false") << ",\n"
    << "  \"phonon_layout\": \"" << (USE_PAIRWISE_PHONONS ? "pairwise" : "shared_two") << "\",\n"
    << "  \"spectrum\": " << (o.spectrum ? "true" : "false") << ",\n"
    << "  \"total_walltime\": " << jnum(total) << "\n"
    << " },\n"
    << " \"params\": {\n"
    << "  \"OMEGA_0\": " << jnum(p.omega_0) << ",\n"
    << "  \"ALPHA\": " << jnum(p.alpha) << ",\n"
    << "  \"BETA\": " << jnum(p.beta) << ",\n"
    << "  \"OMEGA\": " << jnum(p.omega_shift) << ",\n"
    << "  \"ORDER\": " << OO << ",\n"
    << "  \"N_PHON\": " << NB << ",\n"
    << "  \"GAMMAS_RE\": [";
  for (int i = 0; i < p.nvar(); ++i) f << (i ? ", " : "") << jnum(p.gammas[i].real());
  std::vector<double> D0_active(p.D0.begin(), p.D0.begin() + NB);
  f << "],\n  \"D0_PHONON\": " << jarr(D0_active) << ",\n"
    << "  \"NTH\": " << jnum(o.nth) << "\n }\n}\n";
  f.close();
  if (!f) {
    std::cerr << "error: failed while writing '" << tmp_out << "'\n";
    return 1;
  }

  // Publish atomically: remove any stale target first (Windows rename fails if
  // the destination exists), then rename. A reader therefore never observes a
  // partially written document at the final path.
  std::remove(o.out.c_str());
  if (std::rename(tmp_out.c_str(), o.out.c_str()) != 0) {
    std::cerr << "error: wrote '" << tmp_out << "' but could not rename it to '"
              << o.out << "'. Is the target open in another program, or is the "
                          "directory synced/scanned by another process?\n";
    return 1;
  }

  if (!o.quiet)
    std::printf("\nTotal: %.1fs -> %s\n", total, o.out.c_str());

  // Everything below is teardown, and on Windows teardown is where this program
  // used to hang forever.
  //
  // 1) Flush explicitly. When stdout is a FILE rather than a console (the sweep
  //    script redirects the solver's output into its log), the C runtime uses
  //    full buffering, so the final line above would otherwise sit in the buffer
  //    until exit — and be lost outright if the process is killed. Every
  //    per-pump-point line already fflush()es for the same reason.
  //
  // 2) Then leave via std::_Exit, bypassing static destructors and DLL unload.
  //    With MinGW + libgomp, a process that has used an OpenMP parallel region
  //    can deadlock during shutdown: winpthreads tears the thread pool down from
  //    DllMain, which needs the loader lock that the exiting thread already
  //    holds. The observable symptom is exactly what a run showed here: all
  //    pump points computed, output file complete, then an indefinite wait at
  //    zero CPU load, ended only by an external kill. Nothing that matters is
  //    skipped: the output stream is closed and renamed above, and both stdio
  //    streams are flushed on the two lines below, so _Exit has no pending work
  //    to lose.
  std::fflush(stdout);
  std::fflush(stderr);
  std::_Exit(0);
}
