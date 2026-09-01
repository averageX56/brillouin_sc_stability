#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int MAX_MODES = 16;
constexpr int MAX_PHONONS = MAX_MODES - 1;
constexpr double PI = 3.141592653589793238462643383279502884;

#ifdef PAIRWISE_PHONONS
constexpr bool USE_PAIRWISE_PHONONS = true;
#else
constexpr bool USE_PAIRWISE_PHONONS = false;
#endif
constexpr int MAX_ACTIVE_PHONONS = USE_PAIRWISE_PHONONS ? MAX_PHONONS : 2;

__host__ __device__ constexpr int phonon_count(int n_ph) {
    return USE_PAIRWISE_PHONONS ? n_ph - 1 : 2;
}

struct Config {
    double E_min = 0.0, E_max = 1.0, dt = 1e-9;
    int nE = 50, n_ph = 3, n_steps = 10000, burn = 1000, thin = 1;
    int n_paths = 100, device = 0, g1_lags = 64, g1_origins = 256;
    int pump_chunk = 0, rk_substeps = 4;
    unsigned long long seed = 0;
    double g = 1.11e4, Gamma = 2.0 * PI * 13.1e6;
    double gamma = 2.0 * PI * 83.0e6, nth = 0.0;
    double gpu_memory_fraction = 0.35;
    bool quiet = false, verbose = false;
    std::string noise = "gauss", scheme = "splitting", out;
    std::vector<double> E_list;
};

struct PathMoments {
    double abs_a[MAX_MODES], i2_a[MAX_MODES], i4_a[MAX_MODES];
    double abs_b[MAX_ACTIVE_PHONONS], i2_b[MAX_ACTIVE_PHONONS], i4_b[MAX_ACTIVE_PHONONS];
    unsigned long long count;
    int diverged;
};

struct CorrOut {
    double re, im;
    unsigned long long count;
};

struct PumpResult {
    double E;
    std::vector<double> A_det, B_det, A_mean, B_mean, g2_a, g2_b;
    std::vector<double> fwhm_g1, r2_g1;
    int n_diverged = 0;
    bool steady = false;
};

inline void cuda_check(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
    }
}

int env_int(const char* name, int fallback) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    return std::stoi(s);
}

double env_double(const char* name, double fallback) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    return std::stod(s);
}

double parse_double(const std::string& s) { return std::stod(s); }
int parse_int(const std::string& s) { return std::stoi(s); }

std::vector<double> parse_list(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(std::stod(item));
    return out;
}

Config parse_args(int argc, char** argv) {
    Config c;
    c.device = env_int("SDE_CUDA_DEVICE", 0);
    c.g1_lags = env_int("SDE_CUDA_G1_LAGS", 64);
    c.g1_origins = env_int("SDE_CUDA_G1_ORIGINS", 256);
    c.pump_chunk = env_int("SDE_CUDA_PUMP_CHUNK", 0);
    c.rk_substeps = env_int("SDE_CUDA_RK_SUBSTEPS", 4);
    c.gpu_memory_fraction = env_double("SDE_CUDA_MEMORY_FRACTION", 0.35);
    auto value = [&](int& i) -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value after ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--E-min") c.E_min = parse_double(value(i));
        else if (a == "--E-max") c.E_max = parse_double(value(i));
        else if (a == "--nE") c.nE = parse_int(value(i));
        else if (a == "--E-list") c.E_list = parse_list(value(i));
        else if (a == "--N-photons") c.n_ph = parse_int(value(i));
        else if (a == "--dt") c.dt = parse_double(value(i));
        else if (a == "--n-steps") c.n_steps = parse_int(value(i));
        else if (a == "--burn") c.burn = parse_int(value(i));
        else if (a == "--thin") c.thin = parse_int(value(i));
        else if (a == "--n-paths") c.n_paths = parse_int(value(i));
        else if (a == "--seed") c.seed = std::stoull(value(i));
        else if (a == "--g") c.g = parse_double(value(i));
        else if (a == "--Gamma") c.Gamma = parse_double(value(i));
        else if (a == "--gamma-opt") c.gamma = parse_double(value(i));
        else if (a == "--nth") c.nth = parse_double(value(i));
        else if (a == "--scheme") c.scheme = value(i);
        else if (a == "--noise") c.noise = value(i);
        else if (a == "--out") c.out = value(i);
        else if (a == "--device") c.device = parse_int(value(i));
        else if (a == "--g1-lags") c.g1_lags = parse_int(value(i));
        else if (a == "--g1-origins") c.g1_origins = parse_int(value(i));
        else if (a == "--pump-chunk") c.pump_chunk = parse_int(value(i));
        else if (a == "--rk-substeps") c.rk_substeps = parse_int(value(i));
        else if (a == "--gpu-memory-fraction") c.gpu_memory_fraction = parse_double(value(i));
        else if (a == "--quiet") c.quiet = true;
        else if (a == "--verbose") c.verbose = true;
        else if (a == "--threads") (void)value(i);  // accepted for CPU CLI compatibility
        else throw std::invalid_argument("unknown option: " + a);
    }
    if (c.E_list.empty()) {
        if (c.nE < 2) throw std::invalid_argument("nE must be >= 2");
        c.E_list.resize(c.nE);
        for (int i = 0; i < c.nE; ++i)
            c.E_list[i] = c.E_min + (c.E_max - c.E_min) * i / (c.nE - 1.0);
    } else c.nE = static_cast<int>(c.E_list.size());
    if (c.n_ph < 2 || c.n_ph > MAX_MODES) throw std::invalid_argument("N-photons must be in [2,16]");
    if (c.dt <= 0 || c.n_steps <= c.burn || c.burn < 0 || c.thin < 1 || c.n_paths < 1)
        throw std::invalid_argument("invalid integration/sampling parameters");
    if (c.g <= 0 || c.Gamma <= 0 || c.gamma <= 0 || c.nth < 0)
        throw std::invalid_argument("g, Gamma, gamma must be positive and nth non-negative");
    if (!(c.gpu_memory_fraction > 0.0 && c.gpu_memory_fraction <= 0.8))
        throw std::invalid_argument("gpu-memory-fraction must be in (0, 0.8]");
    if (c.noise != "gauss") throw std::invalid_argument("CUDA backend supports --noise gauss only");
    if (c.out.empty()) throw std::invalid_argument("--out is required");
    c.g1_lags = std::max(2, c.g1_lags);
    c.g1_origins = std::max(1, c.g1_origins);
    c.rk_substeps = std::max(1, c.rk_substeps);
    return c;
}

__host__ __device__ inline double2 z(double x = 0.0, double y = 0.0) { return make_double2(x, y); }
__host__ __device__ inline double2 add(double2 a, double2 b) { return z(a.x + b.x, a.y + b.y); }
__host__ __device__ inline double2 mul(double2 a, double2 b) {
    return z(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__host__ __device__ inline double2 scale(double2 a, double s) { return z(a.x * s, a.y * s); }
__host__ __device__ inline double2 conjz(double2 a) { return z(a.x, -a.y); }
__host__ __device__ inline double2 iz(double2 a) { return z(-a.y, a.x); }
__host__ __device__ inline double norm2(double2 a) { return a.x * a.x + a.y * a.y; }
__host__ __device__ inline bool finite_number(double x) {
    return x == x && fabs(x) < std::numeric_limits<double>::infinity();
}

__device__ void rhs_device(const double2* a, const double2* b, double E, int n_ph,
                           double g, double Gamma, double gamma,
                           double2* da, double2* db) {
    for (int j = 0; j < n_ph; ++j) {
        double2 v = scale(a[j], -0.5 * gamma);
        const int right_b = USE_PAIRWISE_PHONONS ? j : (j & 1);
        const int left_b = USE_PAIRWISE_PHONONS ? j - 1 : ((j - 1) & 1);
        if (j + 1 < n_ph) v = add(v, scale(iz(mul(b[right_b], a[j + 1])), g));
        if (j > 0) v = add(v, scale(iz(mul(conjz(b[left_b]), a[j - 1])), g));
        if (j == 0) v = add(v, z(0.0, -E));
        da[j] = v;
    }
    const int n_b = phonon_count(n_ph);
    for (int p = 0; p < n_b; ++p) {
        double2 s = z();
        if (USE_PAIRWISE_PHONONS) {
            // Matches the photon equations: b_p is sourced by a_p a^*_{p+1}.
            s = mul(a[p], conjz(a[p + 1]));
        } else {
            for (int j = p; j + 1 < n_ph; j += 2)
                s = add(s, mul(conjz(a[j]), a[j + 1]));
        }
        db[p] = add(scale(b[p], -0.5 * Gamma), scale(iz(s), g));
    }
}

__device__ void rk4_step(double2* a, double2* b, double E, int n_ph,
                         double g, double Gamma, double gamma, double dt) {
    double2 k1a[MAX_MODES], k2a[MAX_MODES], k3a[MAX_MODES], k4a[MAX_MODES];
    double2 ta[MAX_MODES], k1b[MAX_ACTIVE_PHONONS], k2b[MAX_ACTIVE_PHONONS];
    double2 k3b[MAX_ACTIVE_PHONONS], k4b[MAX_ACTIVE_PHONONS], tb[MAX_ACTIVE_PHONONS];
    const int n_b = phonon_count(n_ph);
    rhs_device(a, b, E, n_ph, g, Gamma, gamma, k1a, k1b);
    for (int j = 0; j < n_ph; ++j) ta[j] = add(a[j], scale(k1a[j], 0.5 * dt));
    for (int p = 0; p < n_b; ++p) tb[p] = add(b[p], scale(k1b[p], 0.5 * dt));
    rhs_device(ta, tb, E, n_ph, g, Gamma, gamma, k2a, k2b);
    for (int j = 0; j < n_ph; ++j) ta[j] = add(a[j], scale(k2a[j], 0.5 * dt));
    for (int p = 0; p < n_b; ++p) tb[p] = add(b[p], scale(k2b[p], 0.5 * dt));
    rhs_device(ta, tb, E, n_ph, g, Gamma, gamma, k3a, k3b);
    for (int j = 0; j < n_ph; ++j) ta[j] = add(a[j], scale(k3a[j], dt));
    for (int p = 0; p < n_b; ++p) tb[p] = add(b[p], scale(k3b[p], dt));
    rhs_device(ta, tb, E, n_ph, g, Gamma, gamma, k4a, k4b);
    for (int j = 0; j < n_ph; ++j)
        a[j] = add(a[j], scale(add(add(k1a[j], scale(add(k2a[j], k3a[j]), 2.0)), k4a[j]), dt / 6.0));
    for (int p = 0; p < n_b; ++p)
        b[p] = add(b[p], scale(add(add(k1b[p], scale(add(k2b[p], k3b[p]), 2.0)), k4b[p]), dt / 6.0));
}

__global__ void integrate_kernel(const double* E, const double2* det_a, const double2* det_b,
                                 int n_pumps, int n_paths, int n_ph, int n_steps,
                                 int burn, int thin, int n_keep, double dt, double g,
                                 double Gamma, double gamma, double nth,
                                 int rk_substeps, int pump_offset,
                                 unsigned long long seed,
                                 double2* samples, PathMoments* moments,
                                 volatile int* progress, int progress_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_pumps * n_paths;
    if (idx >= total) return;
    int q = idx / n_paths;
    double2 a[MAX_MODES], b[MAX_ACTIVE_PHONONS];
    const int n_b = phonon_count(n_ph);
    for (int j = 0; j < n_ph; ++j) a[j] = det_a[q * n_ph + j];
    for (int p = 0; p < n_b; ++p) b[p] = det_b[q * n_b + p];

    curandStatePhilox4_32_10_t rng;
    unsigned long long global_trajectory =
        (static_cast<unsigned long long>(pump_offset + q) * n_paths + (idx % n_paths));
    curand_init(seed, global_trajectory, 0, &rng);
    double init_sigma = sqrt(0.5 * nth);
    for (int p = 0; p < n_b; ++p)
        b[p] = add(b[p], z(init_sigma * curand_normal_double(&rng),
                           init_sigma * curand_normal_double(&rng)));

    PathMoments m{};
    double sub_dt = dt / rk_substeps;
    double noise_sigma = sqrt(0.5 * Gamma * nth * sub_dt);
    int keep = 0;
    for (int step = 0; step < n_steps; ++step) {
        for (int sub = 0; sub < rk_substeps; ++sub) {
            rk4_step(a, b, E[q], n_ph, g, Gamma, gamma, sub_dt);
            for (int p = 0; p < n_b; ++p)
                b[p] = add(b[p], z(noise_sigma * curand_normal_double(&rng),
                                   noise_sigma * curand_normal_double(&rng)));
        }
        bool finite = true;
        for (int j = 0; j < n_ph; ++j)
            finite = finite && finite_number(a[j].x) && finite_number(a[j].y) && norm2(a[j]) < 1e200;
        for (int p = 0; p < n_b; ++p)
            finite = finite && finite_number(b[p].x) && finite_number(b[p].y) && norm2(b[p]) < 1e200;
        if (!finite) { m.diverged = 1; break; }
        if (step >= burn && ((step - burn) % thin == 0)) {
            for (int j = 0; j < n_ph; ++j) {
                double i2 = norm2(a[j]);
                m.abs_a[j] += sqrt(i2); m.i2_a[j] += i2; m.i4_a[j] += i2 * i2;
                size_t off = (((static_cast<size_t>(q) * n_paths + (idx % n_paths)) * n_ph + j)
                              * n_keep + keep);
                samples[off] = a[j];
            }
            for (int p = 0; p < n_b; ++p) {
                double i2 = norm2(b[p]);
                m.abs_b[p] += sqrt(i2); m.i2_b[p] += i2; m.i4_b[p] += i2 * i2;
            }
            ++keep;
        }
        if (idx == 0 && ((step + 1) % progress_stride == 0 || step + 1 == n_steps)) {
            *progress = step + 1;
            __threadfence_system();
        }
    }
    m.count = keep;
    moments[idx] = m;
}

__global__ void correlation_kernel(const double2* samples, const PathMoments* moments,
                                   const int* lags, int n_lags, int origins,
                                   int n_pumps, int n_paths, int n_ph, int n_keep,
                                   CorrOut* out) {
    int item = blockIdx.x;
    int lag_i = item % n_lags;
    int mode = (item / n_lags) % n_ph;
    int q = item / (n_lags * n_ph);
    if (q >= n_pumps) return;
    int lag = lags[lag_i], avail = n_keep - lag;
    double re = 0.0, im = 0.0;
    unsigned long long count = 0;
    int total = n_paths * origins;
    for (int k = threadIdx.x; k < total; k += blockDim.x) {
        int path = k / origins, o = k % origins;
        const PathMoments& pm = moments[q * n_paths + path];
        if (pm.diverged || pm.count <= static_cast<unsigned long long>(lag) || avail <= 0) continue;
        int t = origins == 1 ? 0 : static_cast<int>((static_cast<long long>(o) * (avail - 1)) / (origins - 1));
        size_t base = ((static_cast<size_t>(q) * n_paths + path) * n_ph + mode) * n_keep;
        double2 prod = mul(conjz(samples[base + t]), samples[base + t + lag]);
        re += prod.x; im += prod.y; ++count;
    }
    extern __shared__ unsigned char raw[];
    double* sr = reinterpret_cast<double*>(raw);
    double* si = sr + blockDim.x;
    unsigned long long* sc = reinterpret_cast<unsigned long long*>(si + blockDim.x);
    sr[threadIdx.x] = re; si[threadIdx.x] = im; sc[threadIdx.x] = count;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sr[threadIdx.x] += sr[threadIdx.x + stride];
            si[threadIdx.x] += si[threadIdx.x + stride];
            sc[threadIdx.x] += sc[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[item] = {sr[0], si[0], sc[0]};
}

using C = std::complex<double>;

void rhs_host(const std::vector<C>& a, const C* b, double E, const Config& c,
              std::vector<C>& da, C* db) {
    const C I(0.0, 1.0);
    for (int j = 0; j < c.n_ph; ++j) {
        C v = -0.5 * c.gamma * a[j];
        const int right_b = USE_PAIRWISE_PHONONS ? j : (j & 1);
        const int left_b = USE_PAIRWISE_PHONONS ? j - 1 : ((j - 1) & 1);
        if (j + 1 < c.n_ph) v += I * c.g * b[right_b] * a[j + 1];
        if (j > 0) v += I * c.g * std::conj(b[left_b]) * a[j - 1];
        if (j == 0) v -= I * E;
        da[j] = v;
    }
    for (int p = 0; p < phonon_count(c.n_ph); ++p) {
        C s = 0.0;
        if (USE_PAIRWISE_PHONONS) {
            s = a[p] * std::conj(a[p + 1]);
        } else {
            for (int j = p; j + 1 < c.n_ph; j += 2) s += std::conj(a[j]) * a[j + 1];
        }
        db[p] = I * c.g * s - 0.5 * c.Gamma * b[p];
    }
}

void rk4_host(std::vector<C>& a, C* b, double E, const Config& c, double h) {
    std::vector<C> k1(c.n_ph), k2(c.n_ph), k3(c.n_ph), k4(c.n_ph), t(c.n_ph);
    C q1[MAX_ACTIVE_PHONONS], q2[MAX_ACTIVE_PHONONS], q3[MAX_ACTIVE_PHONONS];
    C q4[MAX_ACTIVE_PHONONS], tb[MAX_ACTIVE_PHONONS];
    const int n_b = phonon_count(c.n_ph);
    rhs_host(a, b, E, c, k1, q1);
    for (int j = 0; j < c.n_ph; ++j) t[j] = a[j] + 0.5 * h * k1[j];
    for (int p = 0; p < n_b; ++p) tb[p] = b[p] + 0.5 * h * q1[p];
    rhs_host(t, tb, E, c, k2, q2);
    for (int j = 0; j < c.n_ph; ++j) t[j] = a[j] + 0.5 * h * k2[j];
    for (int p = 0; p < n_b; ++p) tb[p] = b[p] + 0.5 * h * q2[p];
    rhs_host(t, tb, E, c, k3, q3);
    for (int j = 0; j < c.n_ph; ++j) t[j] = a[j] + h * k3[j];
    for (int p = 0; p < n_b; ++p) tb[p] = b[p] + h * q3[p];
    rhs_host(t, tb, E, c, k4, q4);
    for (int j = 0; j < c.n_ph; ++j) a[j] += h * (k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j]) / 6.0;
    for (int p = 0; p < n_b; ++p) b[p] += h * (q1[p] + 2.0*q2[p] + 2.0*q3[p] + q4[p]) / 6.0;
}

bool deterministic_state(double E, const Config& c, std::vector<C>& a, C* b) {
    // Exact zero is an invariant solution even above threshold.  A vanishingly
    // small symmetry-breaking seed lets an unstable generated branch grow;
    // below threshold the same seed decays back to zero.
    double seed_amp = 1e-9 * std::max(1.0, E / c.gamma);
    for (int j = 1; j < c.n_ph; ++j)
        if (std::abs(a[j]) < seed_amp) a[j] += C(seed_amp, 0.37 * seed_amp * (j + 1));
    const int n_b = phonon_count(c.n_ph);
    for (int p = 0; p < n_b; ++p)
        if (std::abs(b[p]) < seed_amp) b[p] += C(0.23 * seed_amp * (p + 1), seed_amp);
    int steps = std::max(10000, c.burn);
    double h = c.dt / c.rk_substeps;
    for (int i = 0; i < steps; ++i)
        for (int sub = 0; sub < c.rk_substeps; ++sub) rk4_host(a, b, E, c, h);
    std::vector<C> da(c.n_ph); C db[MAX_ACTIVE_PHONONS];
    rhs_host(a, b, E, c, da, db);
    double err = 0.0;
    for (int j = 0; j < c.n_ph; ++j)
        err = std::max(err, std::abs(da[j]) / (c.gamma * (1.0 + std::abs(a[j]))));
    for (int p = 0; p < n_b; ++p)
        err = std::max(err, std::abs(db[p]) / (c.Gamma * (1.0 + std::abs(b[p]))));
    bool finite = std::isfinite(err);
    for (const C& x : a) finite = finite && std::isfinite(x.real()) && std::isfinite(x.imag());
    return finite && err < 1e-6;
}

std::vector<int> make_lags(int requested, int n_keep) {
    int max_lag = std::max(1, n_keep / 2);
    std::vector<int> lags{0};
    if (max_lag == 1) { lags.push_back(1); return lags; }
    for (int i = 0; i < requested - 1; ++i) {
        double u = requested == 2 ? 1.0 : static_cast<double>(i) / (requested - 2);
        int lag = std::max(1, static_cast<int>(std::llround(std::exp(u * std::log(max_lag)))));
        if (lag < n_keep && lag != lags.back()) lags.push_back(lag);
    }
    return lags;
}

std::pair<double,double> fit_linewidth(const std::vector<int>& lags,
                                       const std::vector<CorrOut>& corr,
                                       double intensity, double sample_dt) {
    std::vector<double> x, y;
    auto collect = [&](bool strict) {
        x.clear(); y.clear();
        for (size_t i = 1; i < lags.size(); ++i) {
            if (!corr[i].count || !(intensity > 0)) continue;
            double mag = std::hypot(corr[i].re, corr[i].im) / corr[i].count / intensity;
            bool good = strict ? (mag > 0.05 && mag < 0.95) : (mag > 0.0 && mag < 1.0);
            if (good && std::isfinite(mag)) { x.push_back(lags[i] * sample_dt); y.push_back(std::log(mag)); }
        }
    };
    collect(true); if (x.size() < 3) collect(false);
    if (x.size() < 2) return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    double sx=0, sy=0, sxx=0, sxy=0;
    for (size_t i=0;i<x.size();++i){sx+=x[i];sy+=y[i];sxx+=x[i]*x[i];sxy+=x[i]*y[i];}
    double n=x.size(), den=n*sxx-sx*sx;
    if (!(den>0)) return {NAN,NAN};
    double slope=(n*sxy-sx*sy)/den, intercept=(sy-slope*sx)/n;
    double ssr=0,sst=0, mean=sy/n;
    for(size_t i=0;i<x.size();++i){double d=y[i]-(intercept+slope*x[i]);ssr+=d*d;double q=y[i]-mean;sst+=q*q;}
    double r2=sst>0?1.0-ssr/sst:NAN;
    return {slope<0?-2.0*slope:0.0,r2};
}

void write_number(std::ostream& o, double x) {
    if (std::isfinite(x)) o << std::setprecision(17) << x; else o << "null";
}
void write_vec(std::ostream& o, const std::vector<double>& v) {
    o << '['; for (size_t i=0;i<v.size();++i){if(i)o<<',';write_number(o,v[i]);} o << ']';
}

void write_json(const Config& c, const std::vector<PumpResult>& results, int n_keep,
                const std::vector<int>& lags, const cudaDeviceProp& prop) {
    const int n_b = phonon_count(c.n_ph);
    std::ofstream o(c.out + ".partial");
    if (!o) throw std::runtime_error("cannot open output: " + c.out + ".partial");
    o << "{\n\"meta\":{\"backend\":\"cuda_cpp\",\"phonon_layout\":\""
      << (USE_PAIRWISE_PHONONS ? "pairwise" : "shared_two") << "\",\"device\":\"" << prop.name
      << "\",\"scheme\":\"cuda_rk4_additive_noise\",\"noise\":\"gauss\",\"dt\":";
    write_number(o,c.dt); o << ",\"n_steps\":"<<c.n_steps<<",\"burn\":"<<c.burn
      << ",\"thin\":"<<c.thin<<",\"rk_substeps\":"<<c.rk_substeps
      << ",\"n_paths\":"<<c.n_paths<<",\"n_keep\":"<<n_keep
      << ",\"gpu_memory_fraction\":"<<c.gpu_memory_fraction
      << ",\"g1_origins\":"<<c.g1_origins<<",\"g1_lags\":"; std::vector<double> ld(lags.begin(),lags.end()); write_vec(o,ld);
    o << "},\n\"params\":{\"ORDER\":"<<c.n_ph<<",\"N_PHON\":"<<n_b<<",\"G\":";write_number(o,c.g);
    o << ",\"GAMMA_OPT\":";write_number(o,c.gamma);o<<",\"Gamma\":";write_number(o,c.Gamma);
    o << ",\"nth\":";write_number(o,c.nth);o<<",\"D0_PHONON\":";write_number(o,0.5*c.Gamma*c.nth);o<<"},\n\"results\":[\n";
    std::vector<double> null_a(c.n_ph,NAN);
    for(size_t i=0;i<results.size();++i){const auto&r=results[i];if(i)o<<",\n";o<<'{';
      o<<"\"E\":";write_number(o,r.E);o<<",\"A_det\":";write_vec(o,r.A_det);o<<",\"B_det\":";write_vec(o,r.B_det);
      o<<",\"A_mean\":";write_vec(o,r.A_mean);o<<",\"B_mean\":";write_vec(o,r.B_mean);
      o<<",\"g2_0\":";write_vec(o,r.g2_a);o<<",\"g2_lin\":";write_vec(o,null_a);
      o<<",\"g2_0_phonon\":";write_vec(o,r.g2_b);o<<",\"fwhm_g1\":";write_vec(o,r.fwhm_g1);
      o<<",\"fwhm_msd\":";write_vec(o,null_a);o<<",\"r2_g1\":";write_vec(o,r.r2_g1);
      o<<",\"r2_msd\":";write_vec(o,null_a);o<<",\"lw_lin\":";write_vec(o,null_a);
      o<<",\"n_diverged\":"<<r.n_diverged<<",\"steady_converged\":"<<(r.steady?"true":"false")<<'}';}
    o << "\n]}\n"; o.close();
#ifdef _WIN32
    // MSVCRT rename does not replace an existing file. POSIX rename below is
    // atomic and must not be preceded by remove on cluster filesystems.
    std::remove(c.out.c_str());
#endif
    if (std::rename((c.out+".partial").c_str(),c.out.c_str())!=0) throw std::runtime_error("failed to publish output JSON");
}

int auto_chunk(const Config& c, int n_keep) {
    if (c.pump_chunk > 0) return std::min(c.pump_chunk,c.nE);
    size_t free_b=0,total_b=0; cuda_check(cudaMemGetInfo(&free_b,&total_b),"cudaMemGetInfo");
    size_t per = static_cast<size_t>(c.n_paths)*c.n_ph*n_keep*sizeof(double2)
               + static_cast<size_t>(c.n_paths)*sizeof(PathMoments)
               + static_cast<size_t>(c.n_ph)*c.g1_lags*sizeof(CorrOut)
               + static_cast<size_t>(c.n_ph + phonon_count(c.n_ph))*sizeof(double2) + sizeof(double);
    // Keep most currently-free memory untouched. This matters on shared cluster
    // nodes and leaves room for the CUDA context, cuRAND and transient allocations.
    size_t safe = static_cast<size_t>(free_b*c.gpu_memory_fraction);
    return std::max(1,std::min(c.nE,static_cast<int>(safe/std::max<size_t>(per,1))));
}

int run(const Config& c) {
    cuda_check(cudaSetDeviceFlags(cudaDeviceMapHost),"cudaSetDeviceFlags");
    cuda_check(cudaSetDevice(c.device),"cudaSetDevice");
    cudaDeviceProp prop{}; cuda_check(cudaGetDeviceProperties(&prop,c.device),"cudaGetDeviceProperties");
    int n_keep=(c.n_steps-c.burn+c.thin-1)/c.thin;
    auto lags=make_lags(c.g1_lags,n_keep);
    int chunk_size=auto_chunk(c,n_keep);
    if(!c.quiet)std::cout<<"CUDA device: "<<prop.name<<"; pump chunk="<<chunk_size<<"; n_keep="<<n_keep<<"\n"<<std::flush;

    const int n_b = phonon_count(c.n_ph);
    std::vector<PumpResult> R(c.nE);
    std::vector<C> a(c.n_ph,0.0); C b[MAX_ACTIVE_PHONONS]{};
    std::vector<double2> detA(c.nE*c.n_ph),detB(c.nE*n_b);
    for(int q=0;q<c.nE;++q){R[q].E=c.E_list[q];R[q].steady=deterministic_state(R[q].E,c,a,b);
      R[q].A_det.resize(c.n_ph);R[q].B_det.resize(n_b);
      for(int j=0;j<c.n_ph;++j){R[q].A_det[j]=std::abs(a[j]);detA[q*c.n_ph+j]=z(a[j].real(),a[j].imag());}
      for(int p=0;p<n_b;++p){R[q].B_det[p]=std::abs(b[p]);detB[q*n_b+p]=z(b[p].real(),b[p].imag());}}

    int* d_lags=nullptr;cuda_check(cudaMalloc(&d_lags,lags.size()*sizeof(int)),"malloc lags");
    cuda_check(cudaMemcpy(d_lags,lags.data(),lags.size()*sizeof(int),cudaMemcpyHostToDevice),"copy lags");
    int* h_progress_raw=nullptr;int* d_progress=nullptr;
    cuda_check(cudaHostAlloc(&h_progress_raw,sizeof(int),cudaHostAllocMapped),"cudaHostAlloc progress");
    cuda_check(cudaHostGetDevicePointer(&d_progress,h_progress_raw,0),"cudaHostGetDevicePointer progress");
    volatile int* h_progress=h_progress_raw;
    int n_chunks=(c.nE+chunk_size-1)/chunk_size,chunk_number=0;
    for(int start=0;start<c.nE;start+=chunk_size){int nq=std::min(chunk_size,c.nE-start);
      ++chunk_number;*h_progress=0;
      double *dE=nullptr;double2 *dA=nullptr,*dB=nullptr,*dS=nullptr;PathMoments*dM=nullptr;CorrOut*dC=nullptr;
      size_t ns=static_cast<size_t>(nq)*c.n_paths*c.n_ph*n_keep;
      cuda_check(cudaMalloc(&dE,nq*sizeof(double)),"malloc E");cuda_check(cudaMalloc(&dA,nq*c.n_ph*sizeof(double2)),"malloc detA");
      cuda_check(cudaMalloc(&dB,nq*n_b*sizeof(double2)),"malloc detB");cuda_check(cudaMalloc(&dS,ns*sizeof(double2)),"malloc samples");
      cuda_check(cudaMalloc(&dM,static_cast<size_t>(nq)*c.n_paths*sizeof(PathMoments)),"malloc moments");
      size_t nc=static_cast<size_t>(nq)*c.n_ph*lags.size();cuda_check(cudaMalloc(&dC,nc*sizeof(CorrOut)),"malloc corr");
      cuda_check(cudaMemcpy(dE,c.E_list.data()+start,nq*sizeof(double),cudaMemcpyHostToDevice),"copy E");
      cuda_check(cudaMemcpy(dA,detA.data()+start*c.n_ph,nq*c.n_ph*sizeof(double2),cudaMemcpyHostToDevice),"copy detA");
      cuda_check(cudaMemcpy(dB,detB.data()+start*n_b,nq*n_b*sizeof(double2),cudaMemcpyHostToDevice),"copy detB");
      int threads=128,blocks=(nq*c.n_paths+threads-1)/threads;
      int progress_stride=std::max(1,c.n_steps/1000);
      integrate_kernel<<<blocks,threads>>>(dE,dA,dB,nq,c.n_paths,c.n_ph,c.n_steps,c.burn,c.thin,n_keep,c.dt,c.g,c.Gamma,c.gamma,c.nth,c.rk_substeps,start,c.seed,dS,dM,d_progress,progress_stride);
      cuda_check(cudaGetLastError(),"integrate kernel launch");
      cudaEvent_t finished;cuda_check(cudaEventCreateWithFlags(&finished,cudaEventDisableTiming),"create progress event");
      cuda_check(cudaEventRecord(finished),"record progress event");
      int last_reported=-1;
      while(true){cudaError_t status=cudaEventQuery(finished);if(status==cudaSuccess)break;
        if(status!=cudaErrorNotReady)cuda_check(status,"query progress event");
        int local=*h_progress;if(local!=last_reported){last_reported=local;
          long long done=static_cast<long long>(chunk_number-1)*c.n_steps+local;
          long long total_steps=static_cast<long long>(n_chunks)*c.n_steps;
          std::cout<<"SDE_PROGRESS "<<done<<' '<<total_steps<<"\n"<<std::flush;}
        std::this_thread::sleep_for(std::chrono::milliseconds(200));}
      cuda_check(cudaEventDestroy(finished),"destroy progress event");
      std::cout<<"SDE_PROGRESS "<<static_cast<long long>(chunk_number)*c.n_steps<<' '
               <<static_cast<long long>(n_chunks)*c.n_steps<<"\n"<<std::flush;
      int corr_threads=256;size_t sh=corr_threads*(2*sizeof(double)+sizeof(unsigned long long));
      correlation_kernel<<<static_cast<int>(nc),corr_threads,sh>>>(dS,dM,d_lags,lags.size(),c.g1_origins,nq,c.n_paths,c.n_ph,n_keep,dC);
      cuda_check(cudaGetLastError(),"correlation kernel launch");cuda_check(cudaDeviceSynchronize(),"correlation kernel");
      std::vector<PathMoments> hm(static_cast<size_t>(nq)*c.n_paths);std::vector<CorrOut> hc(nc);
      cuda_check(cudaMemcpy(hm.data(),dM,hm.size()*sizeof(PathMoments),cudaMemcpyDeviceToHost),"copy moments");
      cuda_check(cudaMemcpy(hc.data(),dC,hc.size()*sizeof(CorrOut),cudaMemcpyDeviceToHost),"copy corr");
      for(int q=0;q<nq;++q){auto&r=R[start+q];r.A_mean.assign(c.n_ph,NAN);r.B_mean.assign(n_b,NAN);r.g2_a.assign(c.n_ph,NAN);r.g2_b.assign(n_b,NAN);r.fwhm_g1.assign(c.n_ph,NAN);r.r2_g1.assign(c.n_ph,NAN);
        unsigned long long cnt=0;std::vector<double>sa(c.n_ph),s2(c.n_ph),s4(c.n_ph),sb(n_b),b2(n_b),b4(n_b);
        for(int pth=0;pth<c.n_paths;++pth){const auto&m=hm[q*c.n_paths+pth];if(m.diverged){++r.n_diverged;continue;}cnt+=m.count;
          for(int j=0;j<c.n_ph;++j){sa[j]+=m.abs_a[j];s2[j]+=m.i2_a[j];s4[j]+=m.i4_a[j];}
          for(int p=0;p<n_b;++p){sb[p]+=m.abs_b[p];b2[p]+=m.i2_b[p];b4[p]+=m.i4_b[p];}}
        if(cnt){for(int j=0;j<c.n_ph;++j){r.A_mean[j]=sa[j]/cnt;double m2=s2[j]/cnt;r.g2_a[j]=m2>0?(s4[j]/cnt)/(m2*m2):NAN;
            std::vector<CorrOut> cc(lags.size());for(size_t l=0;l<lags.size();++l)cc[l]=hc[(static_cast<size_t>(q)*c.n_ph+j)*lags.size()+l];
            auto fit=fit_linewidth(lags,cc,m2,c.dt*c.thin);r.fwhm_g1[j]=fit.first;r.r2_g1[j]=fit.second;}
          for(int p=0;p<n_b;++p){r.B_mean[p]=sb[p]/cnt;double m2=b2[p]/cnt;r.g2_b[p]=m2>0?(b4[p]/cnt)/(m2*m2):NAN;}}
        if(!c.quiet)std::cout<<"pump "<<(start+q+1)<<"/"<<c.nE<<" E="<<r.E<<" diverged="<<r.n_diverged<<"\n"<<std::flush;}
      cudaFree(dE);cudaFree(dA);cudaFree(dB);cudaFree(dS);cudaFree(dM);cudaFree(dC);
    }
    cudaFreeHost(h_progress_raw);cudaFree(d_lags);write_json(c,R,n_keep,lags,prop);return 0;
}

}  // namespace

int main(int argc,char**argv){try{return run(parse_args(argc,argv));}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 2;}}
