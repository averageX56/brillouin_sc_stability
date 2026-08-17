"""brillouin — analysis helpers for the Brillouin cascade SDE solver.

Submodules
----------
plots          plotly loading + plotting for solver JSONs and n_th sweeps
linear_theory  exact finite-pump linear (Lyapunov) theory, closed form

Typical use
-----------
    from brillouin import plots
    D = plots.load_sweep("data/sde_sweep.json")
    plots.quality_report(D)
    plots.plot_g2(D).show()
"""

__all__ = ["plots", "linear_theory"]
