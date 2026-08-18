"""brillouin.plots — loading + plotting for the Brillouin cascade SDE solver.

Everything the old notebook did inline now lives here, so the notebook is just a
thin driver. Two families of functions:

Single pump sweep (one JSON from build/sde_solver):
    load_sweep(path)                -> dict D
    quality_report(D)               -> printed diagnostics
    plot_steady_amps(D)             -> amplitudes vs pump
    plot_g2(D)                      -> photon g2(0) vs pump, SDE vs Lyapunov
    plot_g2_phonon(D)               -> phonon g2(0) vs pump
    plot_linewidths(D)              -> FWHM vs pump (|g1| and phase-MSD)
    plot_spectrum(D, E_index)       -> Lorentzian Stokes spectrum at one pump
    plot_spectra_waterfall(D)       -> all spectra overlaid
    check_phonon_g2(D)              -> validation: decoupled phonon must give 2

n_th sweep (JSON from scripts/nth_sweep.py):
    load_nth_sweep(path)            -> dict S
    plot_temperature_generation(S)  -> one all-mode generation figure per T
    plot_temperature_linewidth_g1(S)-> one all-mode g1-linewidth figure per T
    plot_temperature_g2(S)          -> one all-mode g2 figure per T
    plot_nth_generation(S)          -> generation curves A_j(E) per n_th
    plot_nth_g2_photon2(S)          -> g2_a2(E) per n_th (+ map)
    masked_g2(entry, mode, floor)   -> g2 with sub-threshold points masked
    plot_nth_g2_phonon(S)           -> phonon g2 vs n_th (should sit at 2)
    plot_nth_thresholds(S)          -> threshold drift vs n_th

Conventions worth remembering
-----------------------------
* Mode a_1 is phase-locked to the pump, so its phase does not diffuse: a low
  R^2 for the mode-1 linewidth fit is expected, not a bug.
* Linewidth from |g1| decay is the primary estimator. FWHM read off the
  Lorentzian spectrum is unreliable here because neighbouring Stokes lines
  overlap and carry very different powers.
* --nth sets D0 = Gamma*nth/2, i.e. <|b_k|^2> = nth exactly. Results predating
  that fix used D0 = 2*Gamma*nth, i.e. nth_old = nth_new/4.
* The Lyapunov linear reference exists only for N = 3; otherwise g2_lin/lw_lin
  are NaN.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover
    raise ImportError("brillouin.plots needs plotly: pip install plotly") from exc

COLORS = ['royalblue', 'crimson', 'forestgreen', 'orange', 'purple',
          'magenta', 'teal', 'gold', 'darkred', 'navy', 'olive', 'sienna']
MARKERS = ['circle', 'square', 'triangle-up', 'diamond', 'cross', 'x',
           'star', 'pentagon', 'hexagon', 'triangle-down', 'bowtie', 'hourglass']


def _c(i):
    return COLORS[i % len(COLORS)]


def _m(i):
    return MARKERS[i % len(MARKERS)]


# ---------------------------------------------------------------------------
# Single pump sweep
# ---------------------------------------------------------------------------
def load_sweep(path="data/sde_sweep.json") -> dict:
    """Read one solver JSON and assemble arrays for plotting.

    JSON null means NaN (JSON has no NaN/Inf) and becomes np.nan here.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run the solver first")
    with open(path) as f:
        J = json.load(f)
    for key in ("meta", "params", "results"):
        if key not in J:
            raise ValueError(f'{path}: no "{key}" section — corrupt or foreign file')
    R = J["results"]
    if not R:
        raise ValueError(f"{path}: empty results list")

    def col(key):
        return np.array([[np.nan if v is None else float(v) for v in r[key]] for r in R])

    def col_opt(key, width):
        if not all(key in r for r in R):
            return np.full((len(R), width), np.nan)
        return col(key)

    def spectra():
        if not all("spectrum_w" in r and "spectrum_S" in r for r in R):
            return None
        out = []
        for r in R:
            w = np.array([np.nan if v is None else float(v) for v in r["spectrum_w"]])
            S = np.array([np.nan if v is None else float(v) for v in r["spectrum_S"]])
            out.append((w, S))
        return out

    P = J["params"]
    order = int(P["ORDER"])
    n_phon = int(P["N_PHON"])

    return dict(
        meta=J["meta"],
        params=P,
        order=order,
        n_phon=n_phon,
        E=np.array([float(r["E"]) for r in R]),
        A_det=col("A_det"),
        B_det=col("B_det"),
        A_mean=col("A_mean"),
        B_mean=col("B_mean"),
        fwhm=col("fwhm_msd"),
        fwhm_g1_=col("fwhm_g1"),
        r2_msd=col("r2_msd"),
        r2_g1=col("r2_g1"),
        g2_nl=col("g2_0"),
        g2_li=col("g2_lin"),
        g2_phon=col_opt("g2_0_phonon", n_phon),
        lw_li=col("lw_lin"),
        n_diverged=np.array([int(r["n_diverged"]) for r in R]),
        steady_ok=np.array([bool(r["steady_converged"]) for r in R]),
        spectra=spectra(),
    )


def quality_report(D: dict) -> None:
    """Print a fit-quality summary plus a list of suspicious points."""
    Es = D["E"]
    order = D["order"]
    problems = []

    n_div = D["n_diverged"]
    if n_div.any():
        for i in np.nonzero(n_div)[0]:
            problems.append(f"E={Es[i]:.3f}: {n_div[i]} trajectories diverged -> reduce dt")
    if not D["steady_ok"].all():
        for i in np.nonzero(~D["steady_ok"])[0]:
            problems.append(f"E={Es[i]:.3f}: deterministic steady state did not converge")

    print("=== run configuration ===")
    m = D["meta"]
    print(f"  scheme = {m.get('scheme')}, noise = {m.get('noise')}, dt = {m.get('dt')}, "
          f"n_paths = {m.get('n_paths')}, burn = {m.get('burn')}, thin = {m.get('thin')}")
    G = float(D["params"]["GAMMAS_RE"][order])  # first phonon decay rate
    dt, n_steps, burn = float(m["dt"]), int(m["n_steps"]), int(m["burn"])
    t_burn, t_rec = burn * dt, (n_steps - burn) * dt
    print(f"  Gamma = {G:g} (1/Gamma = {1.0 / G:g}); burn = {t_burn * G:.2f}/Gamma, "
          f"record = {t_rec * G:.1f}/Gamma")
    n_eff = m.get("n_paths", 1) * t_rec * G
    print(f"  N_eff ~ n_paths*Gamma*T = {n_eff:.0f}  ->  g2 scatter ~ "
          f"{100.0 / max(np.sqrt(n_eff), 1e-9):.1f}%")
    if t_rec * G < 20:
        problems.append(f"recorded window is only {t_rec * G:.1f}/Gamma — g2 will be noisy")
    if t_burn * G < 1:
        problems.append(f"burn is only {t_burn * G:.2f}/Gamma — photon transient may survive")

    print("\n=== linewidth fit quality (R^2) ===")
    print(f"{'E':>6} | {'R2 |g1|':>{9 * order}} | {'R2 MSD':>{9 * order}} | div")
    for i, E in enumerate(Es):
        rg = ' '.join(f'{v:8.4f}' for v in D["r2_g1"][i])
        rm = ' '.join(f'{v:8.4f}' for v in D["r2_msd"][i])
        print(f"{E:6.2f} | {rg} | {rm} | {n_div[i]:3d}")

    # Mode 1 is phase-locked to the pump: its phase does not diffuse, so a poor
    # fit there is expected and is not flagged.
    bad = np.isfinite(D["r2_g1"][:, 1:]) & (D["r2_g1"][:, 1:] < 0.9)
    if bad.any():
        for i, j in zip(*np.nonzero(bad)):
            problems.append(f"E={Es[i]:.3f}, mode {j + 2}: R2(|g1|)={D['r2_g1'][i, j + 1]:.3f} < 0.9")

    print("\n=== agreement of the two linewidth estimators (|g1| / MSD) ===")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = D["fwhm_g1_"] / D["fwhm"]
    for i, E in enumerate(Es):
        print(f"  E={E:5.2f}: " + ' '.join(f'{v:7.3f}' for v in ratio[i]))

    print()
    if problems:
        print("!!! PROBLEMS:")
        for s in problems:
            print("  -", s)
    else:
        print("no problems detected")


def check_phonon_g2(D: dict, tol: float = 0.05) -> None:
    """Validation for N = 2: the second phonon is exactly decoupled there.

    At order 2 the b2 source sum is empty (it runs over even 1-based modes j with
    a j+1 neighbour, and there is none) and b2 also drops out of the photon
    equations. So b2 is a free complex OU process: stationary it is complex
    Gaussian with zero mean, |b2|^2 is exponential, and g2(0) = 2 EXACTLY.
    Any deviation is sampling error or a leftover transient, never physics.
    """
    if D["order"] != 2:
        print(f"check_phonon_g2: order = {D['order']}, not 2 — b2 is coupled here, "
              f"g2 < 2 is legitimate. Nothing to check.")
        return
    g2b2 = D["g2_phon"][:, 1]
    fin = g2b2[np.isfinite(g2b2)]
    if fin.size == 0:
        print("check_phonon_g2: no finite g2 for b2")
        return
    m = float(np.mean(fin))
    worst = float(np.max(np.abs(fin - 2.0)))
    print(f"decoupled phonon b2 at N=2:  mean g2 = {m:.4f}  (expected exactly 2)")
    print(f"  max |g2 - 2| over the pump grid = {worst:.4f}")
    n_eff = D["meta"].get("n_paths", 1)
    print(f"  note: expected scatter is ~2/sqrt(N_eff); with n_paths = {n_eff} and the "
          f"recorded window above, see quality_report for the estimate.")
    print("  VERDICT:", "OK" if abs(m - 2.0) < tol else "OFF — check burn / record length")


def plot_steady_amps(D: dict, show_sde=True, title_suffix="") -> "go.Figure":
    """Amplitudes vs pump: deterministic fixed point and, optionally, SDE means."""
    Es, order, n_phon = D["E"], D["order"], D["n_phon"]
    fig = go.Figure()
    for k in range(order):
        fig.add_trace(go.Scatter(
            x=Es, y=D["A_det"][:, k], mode="lines+markers", name=f"A{k + 1}",
            line=dict(width=2, color=_c(k)), marker=dict(size=6, symbol=_m(k))))
    for k in range(n_phon):
        fig.add_trace(go.Scatter(
            x=Es, y=D["B_det"][:, k], mode="lines+markers", name=f"rho{k + 1}",
            line=dict(width=1.5, dash="dot", color=_c(order + k)),
            marker=dict(size=4, symbol=_m(order + k))))
    if show_sde:
        for k in range(order):
            fig.add_trace(go.Scatter(
                x=Es, y=D["A_mean"][:, k], mode="markers", name=f"<|a{k + 1}|> SDE",
                marker=dict(size=9, symbol="circle-open", color=_c(k),
                            line=dict(width=2))))
        for k in range(n_phon):
            fig.add_trace(go.Scatter(
                x=Es, y=D["B_mean"][:, k], mode="markers", name=f"<|b{k + 1}|> SDE",
                marker=dict(size=8, symbol="square-open", color=_c(order + k),
                            line=dict(width=2))))
    fig.update_xaxes(title_text="pump E")
    fig.update_yaxes(title_text="steady amplitude")
    fig.update_layout(title="Generation curves: deterministic fixed point vs SDE means"
                            + title_suffix,
                      template="plotly_white", height=520, width=950,
                      hovermode="x unified")
    return fig


def plot_g2(D: dict) -> "go.Figure":
    """Photon g2(0) vs pump: nonlinear SDE against the Lyapunov linear theory."""
    Es, order = D["E"], D["order"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("g2(0)", "|g2(0) - 1| (log scale)"))
    for j in range(order):
        fig.add_trace(go.Scatter(
            x=Es, y=D["g2_nl"][:, j], mode="lines+markers",
            name=f"g2 mode {j + 1} (SDE)",
            line=dict(color=_c(j), width=2), marker=dict(symbol=_m(j), size=8)),
            row=1, col=1)
        fig.add_trace(go.Scatter(
            x=Es, y=D["g2_li"][:, j], mode="lines",
            name=f"g2 mode {j + 1} (linear)",
            line=dict(color=_c(j), width=2, dash="dash")), row=1, col=1)
    for j in range(order):
        fig.add_trace(go.Scatter(
            x=Es, y=np.abs(D["g2_nl"][:, j] - 1.0), mode="lines+markers",
            line=dict(color=_c(j), width=2), marker=dict(symbol=_m(j), size=8),
            showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=Es, y=np.abs(D["g2_li"][:, j] - 1.0), mode="lines",
            line=dict(color=_c(j), width=2, dash="dash"), showlegend=False),
            row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dot", line_color="black", opacity=0.5, row=1, col=1)
    fig.add_hline(y=2.0, line_dash="dot", line_color="red", opacity=0.5, row=1, col=1)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_xaxes(title_text="pump E", row=2, col=1)
    fig.update_layout(title="Photon g2(0): nonlinear SDE vs linear theory",
                      template="plotly_white", height=760, width=950,
                      hovermode="x unified")
    return fig


def plot_g2_phonon(D: dict) -> "go.Figure | None":
    """Phonon g2(0) vs pump. For a thermalised, decoupled phonon this must be 2."""
    g2_phon = D["g2_phon"]
    if g2_phon is None or not np.isfinite(g2_phon).any():
        print("g2_0_phonon missing or all-NaN (old file, or --linear-only) — nothing to plot.")
        return None
    Es, order = D["E"], D["order"]
    fig = go.Figure()
    for k in range(g2_phon.shape[1]):
        fig.add_trace(go.Scatter(
            x=Es, y=g2_phon[:, k], mode="lines+markers", name=f"g2 phonon b{k + 1}",
            line=dict(color=_c(order + k), width=2, dash="dot"),
            marker=dict(symbol=_m(order + k), size=8)))
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", opacity=0.7,
                  annotation_text="thermal g2 = 2")
    fig.add_hline(y=1.0, line_dash="dot", line_color="black", opacity=0.5)
    fig.update_xaxes(title_text="pump E")
    fig.update_yaxes(title_text="phonon g2(0)")
    fig.update_layout(title="Phonon g2(0) vs pump (nonlinear SDE)",
                      template="plotly_white", height=500, width=900,
                      hovermode="x unified")
    return fig


def plot_linewidths(D: dict, ref_slope=True) -> "go.Figure":
    """Linewidths vs pump (log-log). |g1| decay is the primary estimator."""
    Es, order = D["E"], D["order"]
    fig = go.Figure()
    for j in range(order):
        fig.add_trace(go.Scatter(
            x=Es, y=D["fwhm_g1_"][:, j], mode="lines+markers",
            name=f"FWHM mode {j + 1} (SDE, |g1|)",
            line=dict(color=_c(j), width=2), marker=dict(symbol=_m(j), size=8)))
        fig.add_trace(go.Scatter(
            x=Es, y=D["fwhm"][:, j], mode="markers",
            name=f"FWHM mode {j + 1} (SDE, phase MSD)",
            marker=dict(symbol="circle-open", size=10, color=_c(j),
                        line=dict(width=2))))
        fig.add_trace(go.Scatter(
            x=Es, y=D["lw_li"][:, j], mode="lines",
            name=f"FWHM mode {j + 1} (linear)",
            line=dict(color=_c(j), width=2, dash="dash")))
    if ref_slope:
        m = Es > 0
        y = D["fwhm_g1_"][m, 1] if order > 1 else D["fwhm_g1_"][m, 0]
        good = np.isfinite(y)
        if good.any():
            E0, y0 = Es[m][good][0], y[good][0]
            fig.add_trace(go.Scatter(
                x=Es[m], y=y0 * (Es[m] / E0) ** -2.0, mode="lines",
                name="~ E^-2 (Schawlow-Townes)",
                line=dict(color="black", width=1.5, dash="dot")))
    fig.update_xaxes(title_text="pump E", type="log")
    fig.update_yaxes(title_text="FWHM = D_eff", type="log")
    fig.update_layout(title="Linewidth narrowing (solid = SDE via |g1|, "
                            "open = phase MSD, dashed = linear theory)",
                      template="plotly_white", height=560, width=980,
                      hovermode="x unified")
    return fig


def plot_spectrum(D: dict, E_index=None, log_y=True, normalize=True) -> "go.Figure | None":
    """Stokes spectrum S(w) at one pump value. Odd modes red, even modes blue."""
    spectra = D["spectra"]
    if spectra is None:
        print("No spectra in this JSON. Rerun the solver with --spectrum.")
        return None
    Es, order = D["E"], D["order"]
    OMEGA = float(D["params"]["OMEGA"])
    if E_index is None:
        E_index = int(np.argmax(Es))
    w, S = spectra[E_index]
    S = np.asarray(S, dtype=float)
    if normalize and np.nanmax(S) > 0:
        S = S / np.nanmax(S)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=w, y=S, mode="lines", line=dict(color="royalblue", width=1.5),
        name=f"S(w), E={Es[E_index]:.3g}", fill="tozeroy",
        fillcolor="rgba(65,105,225,0.15)"))
    for j in range(order):
        fig.add_vline(x=-j * OMEGA, line_width=1, line_dash="dot",
                      line_color=("crimson" if j % 2 == 0 else "royalblue"),
                      opacity=0.5)
    fig.update_xaxes(title_text="detuning from the pump line w_1")
    fig.update_yaxes(title_text="S(w), normalised", type="log" if log_y else "linear")
    fig.update_layout(title=f"Stokes spectrum at E = {Es[E_index]:.3g}",
                      template="plotly_white", height=520, width=980)
    return fig


def plot_spectra_waterfall(D: dict, log_y=True) -> "go.Figure | None":
    spectra = D["spectra"]
    if spectra is None:
        print("No spectra in this JSON (--spectrum).")
        return None
    Es = D["E"]
    fig = go.Figure()
    for i, (w, S) in enumerate(spectra):
        S = np.asarray(S, dtype=float)
        mx = np.nanmax(S)
        if not (mx > 0):
            continue
        fig.add_trace(go.Scatter(x=w, y=S / mx, mode="lines", name=f"E={Es[i]:.2g}",
                                 line=dict(width=1.2, color=_c(i)), opacity=0.8))
    fig.update_xaxes(title_text="detuning from w_1")
    fig.update_yaxes(title_text="S(w), normalised", type="log" if log_y else "linear")
    fig.update_layout(title="Stokes spectra at several pump values",
                      template="plotly_white", height=550, width=1000,
                      hovermode="x unified")
    return fig


# ---------------------------------------------------------------------------
# n_th sweep (sweep_nth.py output)
# ---------------------------------------------------------------------------
def load_nth_sweep(path="data/nth_sweep.json") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/nth_sweep.py first")
    with open(path) as f:
        S = json.load(f)
    if S.get("meta", {}).get("kind") != "nth_sweep":
        raise ValueError(f"{path} is not an n_th sweep file")

    def arr(x):
        return np.array([[np.nan if v is None else float(v) for v in row] for row in x])

    for e in S["entries"]:
        e["E"] = np.array(e["E"], dtype=float)
        for key in ("A_det", "A_mean", "B_det", "B_mean", "g2_0", "g2_lin",
                    "g2_0_phonon", "fwhm_g1", "fwhm_msd"):
            e[key] = arr(e[key])
    return S


def plot_temperature_generation(S: dict) -> list["go.Figure"]:
    """Return one generation figure per temperature, with every photon mode.

    Each solid curve is the stochastic mean amplitude ``<|a_j|>`` from
    ``A_mean``.  Temperature and its Bose--Einstein occupation are kept in the
    title so each returned figure is self-contained.
    """
    figures = []
    n_ph = int(S["meta"]["N_photons"])
    for entry in S["entries"]:
        fig = go.Figure()
        for mode in range(n_ph):
            fig.add_trace(go.Scatter(
                x=entry["E"], y=entry["A_mean"][mode], mode="lines+markers",
                name=f"a_{mode + 1}",
                line=dict(color=_c(mode), width=2),
                marker=dict(symbol=_m(mode), size=6)))
        fig.update_xaxes(title_text="pump E")
        fig.update_yaxes(title_text="<|a_j|>")
        fig.update_layout(
            title=(f"Generation of all photon modes: T = {entry['T_K']:g} K, "
                   f"n_th = {entry['nth']:.8g}"),
            template="plotly_white", height=560, width=980,
            hovermode="x unified")
        figures.append(fig)
    return figures


def plot_temperature_linewidth_g1(S: dict) -> list["go.Figure"]:
    """Return one g1-linewidth figure per temperature, with every photon mode."""
    figures = []
    n_ph = int(S["meta"]["N_photons"])
    for entry in S["entries"]:
        fig = go.Figure()
        for mode in range(n_ph):
            fig.add_trace(go.Scatter(
                x=entry["E"], y=entry["fwhm_g1"][mode], mode="lines+markers",
                name=f"a_{mode + 1}",
                line=dict(color=_c(mode), width=2),
                marker=dict(symbol=_m(mode), size=6)))
        fig.update_xaxes(title_text="pump E")
        fig.update_yaxes(title_text="FWHM from |g1| decay, s^-1")
        fig.update_layout(
            title=(f"g1 linewidths of all photon modes: T = {entry['T_K']:g} K, "
                   f"n_th = {entry['nth']:.8g}"),
            template="plotly_white", height=560, width=980,
            hovermode="x unified")
        figures.append(fig)
    return figures


def plot_temperature_g2(S: dict, amp_floor: float = 0.0) -> list["go.Figure"]:
    """Return one stationary-photon-g2 figure per temperature, all modes.

    ``amp_floor=0`` keeps every solver point.  A positive value masks points
    where ``<|a_j|>`` is below that fraction of the mode maximum; this is useful
    only when sub-threshold ratios are dominated by numerical residue.
    """
    figures = []
    n_ph = int(S["meta"]["N_photons"])
    for entry in S["entries"]:
        fig = go.Figure()
        for mode in range(n_ph):
            fig.add_trace(go.Scatter(
                x=entry["E"], y=masked_g2(entry, mode, amp_floor),
                mode="lines+markers", name=f"a_{mode + 1}",
                line=dict(color=_c(mode), width=2),
                marker=dict(symbol=_m(mode), size=6)))
        fig.add_hline(y=1.0, line_dash="dot", line_color="black",
                      annotation_text="coherent g2 = 1")
        fig.add_hline(y=2.0, line_dash="dash", line_color="red",
                      annotation_text="thermal g2 = 2")
        fig.update_xaxes(title_text="pump E")
        fig.update_yaxes(title_text="g_j^(2)(0)")
        fig.update_layout(
            title=(f"Stationary g2 of all photon modes: T = {entry['T_K']:g} K, "
                   f"n_th = {entry['nth']:.8g}"),
            template="plotly_white", height=560, width=980,
            hovermode="x unified")
        figures.append(fig)
    return figures


def plot_nth_generation(S: dict, mode=1) -> "go.Figure":
    """Generation curve of one photon mode at every n_th (0-based `mode`).

    Solid = SDE mean <|a|>, dashed = deterministic fixed point. The n_th = 0 run
    has no noise at all, so the two coincide there — that is the noiseless limit
    the finite-temperature curves should approach.
    """
    fig = go.Figure()
    for i, e in enumerate(S["entries"]):
        nth = e["nth"]
        fig.add_trace(go.Scatter(
            x=e["E"], y=e["A_mean"][mode], mode="lines+markers",
            name=f"n_th = {nth:g} (SDE)",
            line=dict(color=_c(i), width=2), marker=dict(symbol=_m(i), size=6)))
        if i == 0:
            fig.add_trace(go.Scatter(
                x=e["E"], y=e["A_det"][mode], mode="lines",
                name="deterministic A_det",
                line=dict(color="black", width=1.5, dash="dash")))
    E2 = S["meta"]["E_threshold2"]
    fig.add_vline(x=E2 / 2.0, line_dash="dot", line_color="grey",
                  annotation_text="E1")
    fig.add_vline(x=E2, line_dash="dot", line_color="grey", annotation_text="E2")
    fig.update_xaxes(title_text="pump E")
    fig.update_yaxes(title_text=f"<|a_{mode + 1}|>")
    fig.update_layout(
        title=f"Generation curve of mode a_{mode + 1} vs pump, by thermal occupancy "
              f"(<|b|^2> = n_th)",
        template="plotly_white", height=560, width=980, hovermode="x unified")
    return fig


def plot_nth_generation_all(S: dict) -> "go.Figure":
    """All photon modes, one subplot per n_th — how the thresholds smear out."""
    ents = S["entries"]
    n = len(ents)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, shared_xaxes=True,
                        subplot_titles=[f"n_th = {e['nth']:g}" for e in ents])
    n_ph = S["meta"]["N_photons"]
    for i, e in enumerate(ents):
        r, c = i // ncol + 1, i % ncol + 1
        for j in range(n_ph):
            fig.add_trace(go.Scatter(
                x=e["E"], y=e["A_mean"][j], mode="lines",
                name=f"A{j + 1}", legendgroup=f"A{j + 1}", showlegend=(i == 0),
                line=dict(color=_c(j), width=2)), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=e["E"], y=e["A_det"][j], mode="lines",
                name=f"A{j + 1} det", legendgroup=f"A{j + 1}d", showlegend=(i == 0),
                line=dict(color=_c(j), width=1, dash="dash")), row=r, col=c)
    fig.update_layout(title="Generation curves per thermal occupancy",
                      template="plotly_white", height=300 * nrow, width=1050)
    return fig


def masked_g2(e: dict, mode: int, amp_floor: float = 0.0) -> np.ndarray:
    """g2 of one photon mode with meaningless sub-threshold points removed.

    The model has no direct photon noise — the photon modes are driven only
    through their coupling to the phonons, starting from the deterministic fixed
    point. Below the first threshold that fixed point is a_j = 0, so whatever
    amplitude the SDE reports there is set by where the steady-state ODE happened
    to stop (a residue of order 1e-12), not by physics, and g2 of such a field is
    numerically arbitrary — typically a large number at E = 0. Passing
    amp_floor = 1e-3 masks every point whose <|a|> is below that fraction of the
    maximum along the sweep. Default 0 keeps everything, so nothing is hidden
    unless you ask.
    """
    g2 = np.array(e["g2_0"][mode], dtype=float).copy()
    if amp_floor <= 0:
        return g2
    A = np.array(e["A_mean"][mode], dtype=float)
    mx = np.nanmax(A) if np.isfinite(A).any() else np.nan
    if np.isfinite(mx) and mx > 0:
        g2[~(A > amp_floor * mx)] = np.nan
    return g2


def plot_nth_g2_photon2(S: dict, mode=1, amp_floor=0.0) -> "go.Figure":
    """g2(0) of photon mode a_2 at EVERY pump point, one curve per n_th.

    amp_floor > 0 drops sub-threshold points where the photon amplitude — and
    hence g2 — is numerical residue rather than physics (see masked_g2).
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"g2 of a_{mode + 1} vs pump", f"g2 of a_{mode + 1} vs n_th (per pump point)"))
    for i, e in enumerate(S["entries"]):
        fig.add_trace(go.Scatter(
            x=e["E"], y=masked_g2(e, mode, amp_floor), mode="lines+markers",
            name=f"n_th = {e['nth']:g}", legendgroup=f"{i}",
            line=dict(color=_c(i), width=2), marker=dict(symbol=_m(i), size=6)),
            row=1, col=1)
    # transpose: g2 vs n_th at fixed E
    nths = np.array([e["nth"] for e in S["entries"]], dtype=float)
    Egrid = S["entries"][0]["E"]
    Z = np.array([masked_g2(e, mode, amp_floor) for e in S["entries"]])  # (n_nth, nE)
    step = max(1, len(Egrid) // 6)
    for k in range(0, len(Egrid), step):
        fig.add_trace(go.Scatter(
            x=nths, y=Z[:, k], mode="lines+markers", name=f"E = {Egrid[k]:.3g}",
            line=dict(width=2), marker=dict(size=6)), row=1, col=2)
    fig.add_hline(y=1.0, line_dash="dot", line_color="black", opacity=0.5)
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", opacity=0.5)
    fig.update_xaxes(title_text="pump E", row=1, col=1)
    fig.update_xaxes(title_text="n_th = <|b|^2>", row=1, col=2)
    fig.update_yaxes(title_text=f"g2(0) of a_{mode + 1}")
    fig.update_layout(title=f"Stationary g2(0) of photon mode a_{mode + 1}",
                      template="plotly_white", height=520, width=1100)
    return fig


def plot_nth_g2_map(S: dict, mode=1, amp_floor=0.0) -> "go.Figure":
    """Heat map g2(0) of one photon mode over the (E, n_th) grid."""
    nths = [e["nth"] for e in S["entries"]]
    Egrid = S["entries"][0]["E"]
    Z = np.array([masked_g2(e, mode, amp_floor) for e in S["entries"]])
    fig = go.Figure(go.Heatmap(
        x=Egrid, y=nths, z=Z, colorscale="RdYlGn_r", zmin=1.0, zmax=3.0,
        colorbar=dict(title=f"g2 a_{mode + 1}")))
    fig.update_xaxes(title_text="pump E")
    fig.update_yaxes(title_text="n_th")
    fig.update_layout(title=f"g2(0) of a_{mode + 1} over (E, n_th)",
                      template="plotly_white", height=520, width=900)
    return fig


def plot_nth_g2_phonon(S: dict) -> "go.Figure":
    """Phonon g2(0) vs pump for every n_th, with the thermal value 2 marked.

    For N = 2 the second phonon is decoupled and this must sit at 2 for all E and
    all n_th; deviations measure sampling error, not physics.
    """
    fig = go.Figure()
    for i, e in enumerate(S["entries"]):
        if e["nth"] == 0:
            continue  # no noise -> phonon g2 undefined/1
        for k in range(e["g2_0_phonon"].shape[0]):
            fig.add_trace(go.Scatter(
                x=e["E"], y=e["g2_0_phonon"][k], mode="lines+markers",
                name=f"b{k + 1}, n_th = {e['nth']:g}",
                line=dict(color=_c(i), width=2, dash=("solid" if k == 0 else "dot")),
                marker=dict(symbol=_m(k), size=5)))
    fig.add_hline(y=2.0, line_dash="dash", line_color="red",
                  annotation_text="thermal g2 = 2")
    fig.update_xaxes(title_text="pump E")
    fig.update_yaxes(title_text="phonon g2(0)")
    fig.update_layout(title="Phonon g2(0): thermal benchmark across the n_th sweep",
                      template="plotly_white", height=520, width=980,
                      hovermode="x unified")
    return fig


def threshold_estimate(E, A, frac=0.1):
    """Crude threshold: the pump where |a| first exceeds frac of its max."""
    A = np.asarray(A, dtype=float)
    good = np.isfinite(A)
    if not good.any():
        return np.nan
    mx = np.nanmax(A)
    if not (mx > 0):
        return np.nan
    idx = np.nonzero(good & (A > frac * mx))[0]
    return float(E[idx[0]]) if idx.size else np.nan


def plot_nth_thresholds(S: dict, frac=0.1) -> "go.Figure":
    """How the apparent generation thresholds drift with thermal occupancy."""
    nths = [e["nth"] for e in S["entries"]]
    n_ph = S["meta"]["N_photons"]
    fig = go.Figure()
    for j in range(1, n_ph):  # mode 1 is always pumped, no threshold
        th = [threshold_estimate(e["E"], e["A_mean"][j], frac) for e in S["entries"]]
        fig.add_trace(go.Scatter(
            x=nths, y=th, mode="lines+markers", name=f"threshold of a_{j + 1}",
            line=dict(color=_c(j), width=2), marker=dict(symbol=_m(j), size=8)))
    E2 = S["meta"]["E_threshold2"]
    fig.add_hline(y=E2, line_dash="dash", line_color="grey",
                  annotation_text="E2 (deterministic)")
    fig.add_hline(y=E2 / 2.0, line_dash="dot", line_color="grey",
                  annotation_text="E1 (deterministic)")
    fig.update_xaxes(title_text="n_th = <|b|^2>")
    fig.update_yaxes(title_text=f"apparent threshold (|a| > {frac:g} of max)")
    fig.update_layout(title="Threshold drift with thermal occupancy",
                      template="plotly_white", height=520, width=900,
                      hovermode="x unified")
    return fig
