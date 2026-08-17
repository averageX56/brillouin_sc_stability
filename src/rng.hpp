// rng.hpp — counter-seeded, OpenMP-friendly random number generation.
//
// Replaces std::mt19937_64 + std::normal_distribution:
//   * xoshiro256++ core (Blackman & Vigna), 4 words of state, ~2x faster than
//     mt19937_64 and with a far smaller per-thread footprint;
//   * each path gets an independent stream seeded by splitmix64(seed, path_id)
//     — no shared state, no seed_seq 32-bit truncation, safe under
//     `#pragma omp parallel for` and bit-reproducible for any thread count;
//   * the normal variate uses the Marsaglia polar method implemented here, so
//     results are identical across compilers (std::normal_distribution is
//     implementation-defined and differs between libstdc++ / libc++ / MSVC).
//
// Noise increments follow Kloeden, Platen & Schurz (1994):
//   * Gaussian dW ~ N(0, dt): the standard choice for the STRONG schemes of
//     Ch. 4 (Euler-Maruyama, order-1.5 strong Taylor);
//   * "telegraph" two-point dW = ±sqrt(dt) with probability 1/2 each,
//     eq. (5.1.5): admissible under the WEAK convergence criterion only, giving
//     the simplified Euler scheme (5.1.6) of weak order 1.0. Note p. 182:
//     a two-point variable is NOT sufficient for weak order 2.0, and it breaks
//     the strong (pathwise) convergence order of the Taylor-1.5 scheme.
#pragma once

#include <cmath>
#include <cstdint>

namespace brillouin {

enum class NoiseKind { Gauss, Telegraph };

inline std::uint64_t splitmix64(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

class Xoshiro256pp {
 public:
  // Independent stream per (seed, stream): state filled by splitmix64, the
  // recommended seeding procedure for the xoshiro family.
  Xoshiro256pp(std::uint64_t seed, std::uint64_t stream) {
    std::uint64_t sm = seed ^ (stream * 0xD2B74407B1CE6E93ULL) ^ 0xA0761D6478BD642FULL;
    for (auto& w : s_) w = splitmix64(sm);
    // A few warm-up draws in case seed/stream are both tiny integers.
    for (int i = 0; i < 8; ++i) (void)next();
  }

  std::uint64_t next() {
    const std::uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }

  // Uniform double in (0, 1): top 53 bits, offset by half an ulp so 0 is excluded
  // (log() below stays finite).
  double uniform01() {
    return (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53;
  }

  // Standard normal N(0,1), Marsaglia polar method (pairs cached).
  double normal() {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    double u, v, q;
    do {
      u = 2.0 * uniform01() - 1.0;
      v = 2.0 * uniform01() - 1.0;
      q = u * u + v * v;
    } while (q >= 1.0 || q == 0.0);
    const double f = std::sqrt(-2.0 * std::log(q) / q);
    spare_ = v * f;
    has_spare_ = true;
    return u * f;
  }

  // Two-point ("telegraph") variable: +1 or -1 with probability 1/2, so that
  // dW = telegraph()*sqrt(dt) satisfies K&P (5.1.5).
  double telegraph() { return (next() >> 63) ? 1.0 : -1.0; }

  // Unit-variance increment of the requested kind.
  double draw(NoiseKind kind) {
    return kind == NoiseKind::Gauss ? normal() : telegraph();
  }

 private:
  static std::uint64_t rotl(std::uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }
  std::uint64_t s_[4]{};
  double spare_ = 0.0;
  bool has_spare_ = false;
};

}  // namespace brillouin
