// linear_theory.hpp — linear (Lyapunov) reference: eq. (63)-(66) and (16),(20),(27),(28).
//
// The Lyapunov equation M S + S M^T = -D is solved for n=5 by forming the 25x25
// Kronecker system and running LU with partial pivoting. For this size that is both
// faster and simpler than Bartels-Stewart, and it lets us detect singularity cleanly.
#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "model.hpp"

namespace brillouin {

using Vec3 = std::array<double, 3>;
using Vec2 = std::array<double, 2>;

// Amplitude denominator, eq. (63).
inline double Q0(const Vec3& A, const Vec2& B) {
  return (B[0] * B[0] + A[1] * A[1]) * (B[1] * B[1] + A[2] * A[2]) -
         A[2] * A[2] * B[1] * B[1];
}

// The linear theory divides by the steady-state amplitudes, so it is undefined at
// the trivial zero fixed point (E = 0: no pump => A, B -> 0). There the formulas
// return values set purely by where the ODE happened to stop (1e-12 vs 1e-36),
// i.e. numerical noise amplified to ~1e27. We detect that case and return NaN
// instead of a meaningless finite number.
inline bool degenerate(const Vec3& A, const Vec2& B, double tol = 1e-8) {
  for (double v : A)
    if (!(std::abs(v) > tol)) return true;
  for (double v : B)
    if (!(std::abs(v) > tol)) return true;
  return false;
}

// Linear-theory linewidths, eq. (64)-(66).
inline Vec3 lw_linear(const Vec3& A, const Vec2& B, const std::array<double, 2>& D0) {
  const double nan_ = std::numeric_limits<double>::quiet_NaN();
  if (degenerate(A, B)) return Vec3{nan_, nan_, nan_};
  const double q = Q0(A, B);
  Vec3 lw{0.0, nan_, nan_};
  if (!(std::abs(q) > 0) || !std::isfinite(q)) return lw;
  const double q2 = q * q;

  const double t2 =
      std::pow(B[1] * B[1] * (A[2] * A[2] - A[1] * A[1]) - A[1] * A[1] * A[2] * A[2], 2) / q2;
  lw[0] = 0.0;
  lw[1] = D0[0] / (2 * B[0] * B[0]) * (1 + t2) +
          D0[1] / (2 * B[1] * B[1]) * (std::pow(A[2], 4) * std::pow(B[1], 4)) / q2;

  const double t3a = std::pow(B[1] * B[1] * (A[2] * A[2] - A[1] * A[1]) -
                                  A[2] * A[2] * (A[1] * A[1] + B[0] * B[0]),
                              2) /
                     q2;
  const double t3b =
      (std::pow(A[2], 4) * std::pow(A[1] * A[1] + B[0] * B[0] - B[1] * B[1], 2)) / q2;
  lw[2] = D0[0] / (2 * B[0] * B[0]) * (1 + t3a) + D0[1] / (2 * B[1] * B[1]) * (1 + t3b);
  return lw;
}

// Solve A_ x = b_ (n x n) by LU with partial pivoting. Returns false if singular.
inline bool lu_solve(std::vector<double>& A_, std::vector<double>& b_, int n) {
  for (int col = 0; col < n; ++col) {
    int piv = col;
    double best = std::abs(A_[col * n + col]);
    for (int r = col + 1; r < n; ++r) {
      const double v = std::abs(A_[r * n + col]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (!(best > 1e-300)) return false;
    if (piv != col) {
      for (int c = 0; c < n; ++c) std::swap(A_[col * n + c], A_[piv * n + c]);
      std::swap(b_[col], b_[piv]);
    }
    const double d = A_[col * n + col];
    for (int r = col + 1; r < n; ++r) {
      const double f = A_[r * n + col] / d;
      if (f == 0.0) continue;
      for (int c = col; c < n; ++c) A_[r * n + c] -= f * A_[col * n + c];
      b_[r] -= f * b_[col];
    }
  }
  for (int r = n - 1; r >= 0; --r) {
    double s = b_[r];
    for (int c = r + 1; c < n; ++c) s -= A_[r * n + c] * b_[c];
    b_[r] = s / A_[r * n + r];
  }
  return true;
}

// Solve M S + S M^T = -D for symmetric S (n = 5), via Kronecker + LU.
inline bool solve_lyapunov(const std::array<std::array<double, 5>, 5>& M,
                           const std::array<double, 5>& Ddiag,
                           std::array<std::array<double, 5>, 5>& S) {
  constexpr int n = 5, nn = n * n;
  std::vector<double> K(nn * nn, 0.0), rhs(nn, 0.0);
  // vec(M S) + vec(S M^T) = (I (x) M + M (x) I) vec(S), column-major-free indexing:
  // row index = i*n + j for entry S(i,j).
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const int row = i * n + j;
      for (int k = 0; k < n; ++k) {
        K[row * nn + (k * n + j)] += M[i][k];  // (M S)_{ij} = sum_k M_ik S_kj
        K[row * nn + (i * n + k)] += M[j][k];  // (S M^T)_{ij} = sum_k S_ik M_jk
      }
      rhs[row] = (i == j) ? -Ddiag[i] : 0.0;
    }
  if (!lu_solve(K, rhs, nn)) return false;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) S[i][j] = rhs[i * n + j];
  return true;
}

// Linear g2(0)-1 via the Lyapunov equation, eq. (16),(20),(27),(28).
inline Vec3 g2_linear_minus_one(const Vec3& A, const Vec2& B, const Vec3& gam,
                                const Vec2& Gam, double g1c, double g2c,
                                const std::array<double, 2>& D0) {
  const double nan_ = std::numeric_limits<double>::quiet_NaN();
  if (degenerate(A, B)) return Vec3{nan_, nan_, nan_};
  const double A1 = A[0], A2 = A[1], A3 = A[2];
  const double B1 = B[0], B2 = B[1];
  const double y1 = gam[0], y2 = gam[1], y3 = gam[2];
  const double G1 = Gam[0], G2 = Gam[1];

  const double s1 = -G1 * B1 / (2 * g1c * A1 * A2 + 1e-30);  // eq. (16)
  const double s2 = -G2 * B2 / (2 * g2c * A2 * A3 + 1e-30);
  const double m1 = g1c * s1, m2 = g2c * s2;

  const std::array<std::array<double, 5>, 5> M{{// eq. (20)
                                                {-y1 / 2, m1 * B1, 0, m1 * A2, 0},
                                                {-m1 * B1, -y2 / 2, m2 * B2, -m1 * A1, m2 * A3},
                                                {0, -m2 * B2, -y3 / 2, 0, -m2 * A2},
                                                {-m1 * A2, -m1 * A1, 0, -G1 / 2, 0},
                                                {0, -m2 * A3, -m2 * A2, 0, -G2 / 2}}};
  const std::array<double, 5> Dd{0.0, 0.0, 0.0, D0[0], D0[1]};

  std::array<std::array<double, 5>, 5> S{};
  if (!solve_lyapunov(M, Dd, S)) return Vec3{nan_, nan_, nan_};
  return Vec3{4 * S[0][0] / (A1 * A1), 4 * S[1][1] / (A2 * A2), 4 * S[2][2] / (A3 * A3)};  // eq. (27)
}

}  // namespace brillouin
