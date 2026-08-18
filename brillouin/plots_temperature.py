"""brillouin.plots — compact plotting helpers for the Otterstrom temperature sweep.

Only the temperature-sweep plots used by ``Brillouin_nonlinear_SDE_cpp.ipynb``
are kept here. The sweep parameter is bath temperature T; the pump axis is the
solver drive E in its original solver units.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover
    raise ImportError("brillouin.plots needs plotly: pip install plotly") from exc


COLORS = [
    "royalblue", "crimson", "forestgreen", "orange", "purple", "magenta",
    "teal", "goldenrod", "darkred", "navy", "olive", "sienna",
]
MARKERS = [
    "circle", "square", "triangle-up", "diamond", "cross", "x",
    "star", "pentagon", "hexagon", "triangle-down", "bowtie", "hourglass",
]


def _c(i: int) -> str:
    return COLORS[i % len(COLORS)]


def _m(i: int) -> str:
    return MARKERS[i % len(MARKERS)]


def _arr2(x) -> np.ndarray:
    return np.array(
        [[np.nan if v is None else float(v) for v in row] for row in x],
        dtype=float,
    )


def load_temperature_sweep(path="data/temperature_sweep.json") -> dict:
    """Load the aggregated JSON written by sweep_temperature_otterstrom.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/sweep_temperature_otterstrom.py first"
        )
    with open(path, encoding="utf-8") as f:
        S = json.load(f)

    if S.get("meta", {}).get("kind") != "temperature_sweep":
        raise ValueError(f"{path} is not a temperature_sweep JSON")
    if not S.get("entries"):
        raise ValueError(f"{path}: empty entries list")

    for e in S["entries"]:
        e["T_K"] = float(e["T_K"])
        e["nth"] = float(e["nth"])
        e["E"] = np.asarray(e["E"], dtype=float)

        for key in (
            "A_det", "A_mean", "B_det", "B_mean", "g2_0", "g2_lin",
            "g2_0_phonon", "fwhm_g1", "fwhm_msd",
        ):
            if key in e:
                e[key] = _arr2(e[key])

        e["n_diverged"] = np.asarray(e.get("n_diverged", []), dtype=int)
        e["steady_ok"] = np.asarray(e.get("steady_ok", []), dtype=bool)

    return S


def temperature_sweep_report(S: dict) -> None:
    """Print the sweep configuration and a compact numerical sanity report."""
    m = S["meta"]
    entries = S["entries"]
    T = np.array([e["T_K"] for e in entries], dtype=float)
    nth = np.array([e["nth"] for e in entries], dtype=float)
    E = np.asarray(entries[0]["E"], dtype=float)

    print("=== Otterstrom temperature sweep ===")
    print("T [K]   :", ", ".join(f"{x:g}" for x in T))
    print("n_th(T) :", ", ".join(f"{x:.6g}" for x in nth))
    print(f"pump E  : {np.nanmin(E):.6g} .. {np.nanmax(E):.6g} (solver units)")
    if "E_threshold2" in m:
        E2 = float(m["E_threshold2"])
        print(f"E1 = {E2/2:.6g}; E2 = {E2:.6g}")
    print(
        f"gamma={float(m['gamma_opt']):.6g} s^-1, "
        f"Gamma={float(m['Gamma']):.6g} s^-1, g={float(m['g']):.6g} s^-1"
    )
    if "Omega_b_reference" in m:
        print(
            f"Omega_b/2pi={float(m['Omega_b_reference'])/(2*np.pi)/1e9:.6g} GHz, "
            f"dt={float(m['dt']):.3g} s"
        )

    div = sum(int(np.sum(e["n_diverged"])) for e in entries if e["n_diverged"].size)
    bad_steady = sum(int(np.sum(~e["steady_ok"])) for e in entries if e["steady_ok"].size)
    print(f"diverged trajectories: {div}; non-converged deterministic points: {bad_steady}")


def masked_g2(entry: dict, mode: int, amp_floor: float = 1e-3) -> np.ndarray:
    """Mask g2 where the corresponding photon amplitude is numerical residue."""
    g2 = np.asarray(entry["g2_0"][mode], dtype=float).copy()
    if amp_floor <= 0:
        return g2
    A = np.asarray(entry["A_mean"][mode], dtype=float)
    if not np.isfinite(A).any():
        return g2
    amax = float(np.nanmax(A))
    if amax > 0:
        g2[~(A > amp_floor * amax)] = np.nan
    return g2


def threshold_estimate(E, amplitude, frac: float = 0.1) -> float:
    """First solver-drive E where amplitude exceeds ``frac`` of its sweep maximum."""
    E = np.asarray(E, dtype=float)
    A = np.asarray(amplitude, dtype=float)
    good = np.isfinite(E) & np.isfinite(A)
    if not good.any():
        return np.nan
    amax = float(np.nanmax(A[good]))
    if not (amax > 0):
        return np.nan
    idx = np.nonzero(good & (A > frac * amax))[0]
    return float(E[idx[0]]) if idx.size else np.nan


def _hover_custom(entry: dict) -> np.ndarray:
    """Columns: temperature T_K and internal Bose occupation n_th."""
    n = len(entry["E"])
    return np.column_stack([
        np.full(n, entry["T_K"], dtype=float),
        np.full(n, entry["nth"], dtype=float),
    ])


def plot_temperature_generation(S: dict, mode: int = 1) -> "go.Figure":
    """Generation curve of one photon mode vs solver pump E at every temperature."""
    fig = go.Figure()
    for i, e in enumerate(S["entries"]):
        fig.add_trace(go.Scatter(
            x=e["E"],
            y=e["A_mean"][mode],
            mode="lines+markers",
            name=f"T = {e['T_K']:g} K",
            customdata=_hover_custom(e),
            hovertemplate=(
                "E=%{x:.6g}<br><|a|>=%{y:.6g}<br>"
                "T=%{customdata[0]:.6g} K<br>"
                "n_th=%{customdata[1]:.6g}<extra></extra>"
            ),
            line=dict(color=_c(i), width=2),
            marker=dict(symbol=_m(i), size=6),
        ))

    e0 = S["entries"][0]
    fig.add_trace(go.Scatter(
        x=e0["E"],
        y=e0["A_det"][mode],
        mode="lines",
        name="deterministic",
        line=dict(color="black", width=1.5, dash="dash"),
    ))

    fig.update_xaxes(title_text="pump E (solver units)")
    fig.update_yaxes(title_text=rf"<|a_{mode + 1}|>")
    fig.update_layout(
        title=f"Generation of photon mode a_{mode + 1} vs pump E and temperature",
        template="plotly_white",
        height=540,
        width=980,
        hovermode="x unified",
    )
    return fig


def plot_temperature_g2(
    S: dict,
    mode: int = 1,
    amp_floor: float = 1e-3,
) -> "go.Figure":
    """Stationary photon g2(0) vs solver pump E, one curve per temperature."""
    fig = go.Figure()
    for i, e in enumerate(S["entries"]):
        fig.add_trace(go.Scatter(
            x=e["E"],
            y=masked_g2(e, mode, amp_floor),
            mode="lines+markers",
            name=f"T = {e['T_K']:g} K",
            customdata=_hover_custom(e),
            hovertemplate=(
                "E=%{x:.6g}<br>g2=%{y:.6g}<br>"
                "T=%{customdata[0]:.6g} K<br>"
                "n_th=%{customdata[1]:.6g}<extra></extra>"
            ),
            line=dict(color=_c(i), width=2),
            marker=dict(symbol=_m(i), size=6),
        ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="black", opacity=0.5)
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", opacity=0.5)
    fig.update_xaxes(title_text="pump E (solver units)")
    fig.update_yaxes(title_text=rf"g^(2)_{{a_{mode + 1}}}(0)")
    fig.update_layout(
        title=f"Stationary g2(0) of photon mode a_{mode + 1}",
        template="plotly_white",
        height=540,
        width=980,
        hovermode="x unified",
    )
    return fig


def plot_temperature_g2_map(
    S: dict,
    mode: int = 1,
    amp_floor: float = 1e-3,
) -> "go.Figure":
    """Heat map of photon g2(0) over the (E, T) sweep plane."""
    entries = S["entries"]
    E = np.asarray(entries[0]["E"], dtype=float)
    T = np.array([e["T_K"] for e in entries], dtype=float)
    Z = np.array([masked_g2(e, mode, amp_floor) for e in entries], dtype=float)

    fig = go.Figure(go.Heatmap(
        x=E,
        y=T,
        z=Z,
        colorscale="RdYlGn_r",
        zmin=1.0,
        zmax=3.0,
        colorbar=dict(title=rf"g2 a_{mode + 1}"),
        hovertemplate="E=%{x:.6g}<br>T=%{y:.6g} K<br>g2=%{z:.6g}<extra></extra>",
    ))
    fig.update_xaxes(title_text="pump E (solver units)")
    fig.update_yaxes(title_text="bath temperature T (K)")
    fig.update_layout(
        title=f"g2(0) of photon mode a_{mode + 1} over (E, T)",
        template="plotly_white",
        height=540,
        width=900,
    )
    return fig


def plot_temperature_thresholds(S: dict, frac: float = 0.1) -> "go.Figure":
    """Apparent generation thresholds in solver pump E vs temperature."""
    entries = S["entries"]
    T = np.array([e["T_K"] for e in entries], dtype=float)
    n_ph = int(S["meta"]["N_photons"])

    fig = go.Figure()
    for j in range(1, n_ph):
        th = [threshold_estimate(e["E"], e["A_mean"][j], frac) for e in entries]
        fig.add_trace(go.Scatter(
            x=T,
            y=np.asarray(th, dtype=float),
            mode="lines+markers",
            name=f"a_{j + 1}",
            line=dict(color=_c(j), width=2),
            marker=dict(symbol=_m(j), size=8),
        ))

    if "E_threshold2" in S["meta"]:
        E2 = float(S["meta"]["E_threshold2"])
        fig.add_hline(y=E2, line_dash="dash", line_color="grey", annotation_text="E2")
        fig.add_hline(y=E2 / 2.0, line_dash="dot", line_color="grey", annotation_text="E1")

    fig.update_xaxes(title_text="bath temperature T (K)")
    fig.update_yaxes(title_text=f"apparent threshold E, A > {frac:g} max(A)")
    fig.update_layout(
        title="Temperature drift of apparent generation thresholds",
        template="plotly_white",
        height=520,
        width=900,
        hovermode="x unified",
    )
    return fig


def plot_temperature_phonon_g2(S: dict) -> "go.Figure":
    """Phonon g2(0) vs solver pump E; thermal/noise sanity check."""
    fig = go.Figure()
    for i, e in enumerate(S["entries"]):
        if e["T_K"] == 0.0 or "g2_0_phonon" not in e:
            continue
        g2p = e["g2_0_phonon"]
        for k in range(g2p.shape[0]):
            fig.add_trace(go.Scatter(
                x=e["E"],
                y=g2p[k],
                mode="lines+markers",
                name=f"b_{k + 1}, T={e['T_K']:g} K",
                line=dict(color=_c(i), width=1.8, dash="solid" if k == 0 else "dot"),
                marker=dict(symbol=_m(k), size=5),
            ))
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="thermal g2 = 2")
    fig.update_xaxes(title_text="pump E (solver units)")
    fig.update_yaxes(title_text="phonon g2(0)")
    fig.update_layout(
        title="Phonon thermal-statistics check across temperature sweep",
        template="plotly_white",
        height=520,
        width=980,
        hovermode="x unified",
    )
    return fig
