"""Exact finite-pump linear theory for the N=3 Brillouin cascade.

This module is deliberately plot-free.  It provides the closed-form linearized
reference used by the solver together with the same physical temperature and
pump-power conversions as ``sweep_temperature_otterstrom.py``.

Conventions
-----------
All dynamical rates are physical SI rates in s^-1.  The thermal occupation is

    n_th(T) = 1 / expm1(hbar*Omega_b/(k_B*T)).

The solver drive E is converted to Otterstrom's on-chip pump power via

    P_cav = 4*hbar*omega_p*v_g/(L*gamma^2) * E^2,
    P_in  = P_cav / 1.8.

For the symmetric N=3 model above the second deterministic threshold,

    E2 = gamma^(3/2) sqrt(Gamma) / (2 g),
    p  = E/E2,

and the exact finite-pump linear result is

    g2_a2(0) - 1 = 32 g^2 n_th/gamma^2 * S22(Gamma/gamma, p^2).
"""
from __future__ import annotations

import numpy as np


HBAR = 1.054571817e-34
K_B = 1.380649e-23
C0 = 299792458.0

# Otterstrom et al. device values used by the temperature-sweep script.
OMEGA_B = 2.0 * np.pi * 6.02e9
PUMP_WAVELENGTH_M = 1535e-9
PUMP_OMEGA = 2.0 * np.pi * C0 / PUMP_WAVELENGTH_M
PUMP_GROUP_VELOCITY = 7.163e7
RACETRACK_LENGTH = 4.576e-2
PUMP_POWER_ENHANCEMENT = 1.8


def nth_from_temperature(T_K, omega_b: float = OMEGA_B):
    """Bose--Einstein thermal occupation of the acoustic mode.

    Accepts a scalar or NumPy array in kelvin.  The exact T=0 limit is returned
    as zero.  Negative temperatures are rejected.
    """
    T = np.asarray(T_K, dtype=float)
    if np.any(T < 0):
        raise ValueError("temperature must be >= 0 K")
    theta = HBAR * float(omega_b) / K_B
    out = np.zeros_like(T, dtype=float)
    pos = T > 0
    x = theta / T[pos]
    # For x >> 1 the occupation is numerically zero; avoid exp overflow.
    safe = x < 700.0
    vals = np.zeros_like(x)
    vals[safe] = 1.0 / np.expm1(x[safe])
    out[pos] = vals
    return float(out) if out.ndim == 0 else out


def power_coefficients(
    gamma: float,
    *,
    pump_omega: float = PUMP_OMEGA,
    pump_group_velocity: float = PUMP_GROUP_VELOCITY,
    racetrack_length: float = RACETRACK_LENGTH,
    enhancement: float = PUMP_POWER_ENHANCEMENT,
) -> tuple[float, float]:
    """Return coefficients (C_cav, C_in) such that P = C E^2 in watts."""
    gamma = float(gamma)
    c_cav = 4.0 * HBAR * pump_omega * pump_group_velocity / (racetrack_length * gamma**2)
    return c_cav, c_cav / enhancement


def intracavity_power_from_E(E, gamma: float):
    """Otterstrom intracavity pump power in watts from solver drive E."""
    c_cav, _ = power_coefficients(gamma)
    E = np.asarray(E, dtype=float)
    out = c_cav * E**2
    return float(out) if out.ndim == 0 else out


def input_power_from_E(E, gamma: float):
    """On-chip input pump power in watts from solver drive E."""
    _, c_in = power_coefficients(gamma)
    E = np.asarray(E, dtype=float)
    out = c_in * E**2
    return float(out) if out.ndim == 0 else out


def E_from_input_power(P_input_W, gamma: float):
    """Inverse of input_power_from_E for non-negative on-chip power."""
    P = np.asarray(P_input_W, dtype=float)
    if np.any(P < 0):
        raise ValueError("pump power must be >= 0 W")
    _, c_in = power_coefficients(gamma)
    out = np.sqrt(P / c_in)
    return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# Exact N=3 linear theory at finite pump.
# ---------------------------------------------------------------------------
def S22_exact(e, u):
    """Dimensionless rational covariance factor S22(e,u)."""
    e = np.asarray(e, dtype=float)
    u = np.asarray(u, dtype=float)
    num = (
        u**3 * (-16*e**3 - 16*e**2 - 4*e)
        + u**2 * (-4*e**5 - 10*e**4 + 8*e**3 - 6*e**2 - 14*e - 4)
        + u * (2*e**5 - e**4 - 7*e**3 + 20*e**2 + 10*e - 4)
        + (-12*e**3 - 28*e**2 - 13*e + 3)
    )
    den = (
        u**3 * (32*e**3 - 24*e - 8)
        + u**2 * (16*e**5 + 28*e**4 - 52*e**3 - 22*e**2 + 2*e - 8)
        + u * (-2*e**6 - 29*e**5 - 44*e**4 + 8*e**3 - 28*e**2 - 31*e + 6)
        + (-24*e**4 - 56*e**3 - 26*e**2 + 6*e)
    )
    return num / den


def E_threshold2(g: float, gamma: float, Gamma: float) -> float:
    """Second deterministic threshold E2 of the symmetric N=3 cascade."""
    return float(gamma)**1.5 * np.sqrt(float(Gamma)) / (2.0 * float(g))


def input_power_threshold2(g: float, gamma: float, Gamma: float) -> float:
    """Second deterministic threshold expressed as on-chip pump power in watts."""
    return input_power_from_E(E_threshold2(g, gamma, Gamma), gamma)


def g2_lin_exact(E, g: float, gamma: float, Gamma: float, nth):
    """Exact finite-pump linear-theory g2_a2(0).

    Valid for p=E/E2>1 while the deterministic fixed point used for the
    linearization remains stable.
    """
    E = np.asarray(E, dtype=float)
    nth = np.asarray(nth, dtype=float)
    p = E / E_threshold2(g, gamma, Gamma)
    return 1.0 + 32.0 * g * g * nth / gamma**2 * S22_exact(Gamma / gamma, p * p)


def g2_lin_temperature(P_input_W, T_K, g: float, gamma: float, Gamma: float):
    """Same exact linear result parameterized by on-chip pump power and T."""
    E = E_from_input_power(P_input_W, gamma)
    nth = nth_from_temperature(T_K)
    return g2_lin_exact(E, g, gamma, Gamma, nth)


def boundary_exact(Gamma, gamma: float, nth, p: float):
    """Critical g for g2_a2(0)=2 at fixed normalized pump p=E/E2."""
    G = np.asarray(Gamma, dtype=float)
    nth = np.asarray(nth, dtype=float)
    s22 = S22_exact(G / gamma, np.full_like(G, float(p) ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = gamma / np.sqrt(32.0 * nth * s22)
    out = np.where(s22 > 0, out, np.nan)
    return float(out) if np.ndim(out) == 0 else out
