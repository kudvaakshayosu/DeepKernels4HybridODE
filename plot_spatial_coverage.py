# -*- coding: utf-8 -*-
"""
plot_spatial_coverage.py — Fig. 4 of the manuscript.

Spatial coverage of the withheld generating kinetics by the +/- k*sigma
function-level uncertainty bands.

    Top    : one-dimensional diagonal slice through the state-space grid,
             showing each method's predictive mean and band against the
             withheld kinetics.
    Bottom : contour of the signed deviation g(.) - mu(.) at grid points where
             the withheld kinetics fall OUTSIDE the +/- k*sigma envelope;
             subplot titles report the percentage of grid points covered, and
             grey points mark the states visited by the training trajectories.

Models are reloaded from the ablation checkpoints written by run_ablation.py,
and the grid is the FIXED box in config.GRID_BOUNDS (Eq. 14) — the same one the
ablation metrics use, so this figure and Fig. 3 are consistent by construction.

Run from an IDE (no arguments) or:  python plot_spatial_coverage.py
"""

# ════════════════════════════════════════════════════════════════════════
# CONFIG — edit these, then press Run.
# ════════════════════════════════════════════════════════════════════════
PLOT_CONFIG = {
    "case":     "cstr",        # "cstr" or "bioreactor"
    "obs":      "full",
    "n_train":  5,             # which N_TRAIN checkpoint to plot
    "rep":      2,             # which repeat (None -> average over all repeats)
    "methods":  ["bnn", "vgp", "dkl"],
    "n_mc":     1000,          # MC weight samples for the BNN predictive
    "grid_n":   None,          # None -> config.GRID_N
    "k_sigma":  None,          # None -> config.K_SIGMA
    "out_name": None,          # None -> spatial_coverage_<case>_N<n>.png
}
# ════════════════════════════════════════════════════════════════════════

import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import config
import figure_utils as fu


def _diagonal_slice(gx, gy, n_pts=200):
    """Query points along the grid diagonal, and the arclength parameter."""
    tx = np.linspace(gx[0], gx[-1], n_pts)
    ty = np.linspace(gy[0], gy[-1], n_pts)
    pts = torch.tensor(np.stack([tx, ty], 1), dtype=torch.float32)
    return pts, tx, ty


def main(cfg=None):
    cfg = dict(PLOT_CONFIG if cfg is None else cfg)
    case, obs = cfg["case"], cfg["obs"]
    n_train = int(cfg["n_train"])
    k_sigma = float(cfg["k_sigma"] if cfg["k_sigma"] is not None
                    else config.K_SIGMA)
    methods = list(cfg["methods"])

    sys_mod = fu.import_system(case)
    grid, g_true, gx, gy, box = fu.build_grid(case, sys_mod, cfg["grid_n"])
    shape = g_true.shape

    reps = ([int(cfg["rep"])] if cfg["rep"] is not None
            else fu.available_repeats(case, n_train, obs))
    if not reps:
        raise FileNotFoundError(
            f"No checkpoints for case='{case}', N_TRAIN={n_train}. "
            f"Run run_ablation.py first.")
    print(f"  Spatial coverage | {fu.system_label(case)} | N={n_train} | "
          f"repeats={reps} | k_sigma={k_sigma} | N_MC={cfg['n_mc']}")

    # ── accumulate per-repeat predictions on grid and diagonal ──────────────
    slice_pts, tx, ty = _diagonal_slice(gx, gy)
    g_slice = sys_mod.true_residual(tx, ty)

    acc = {m: {"mu": [], "sig": [], "smu": [], "ssig": []} for m in methods}
    meta_last = None
    for rep in reps:
        predictors, meta = fu.load_predictors(
            case, n_train, rep, methods, obs=obs, n_mc=cfg["n_mc"])
        meta_last = meta
        for m in methods:
            if m not in predictors:
                continue
            mu, sig = fu.predict_on_grid(predictors[m], grid, shape)
            acc[m]["mu"].append(mu)
            acc[m]["sig"].append(sig)
            smu, ssig = fu.predict_on_grid(predictors[m], slice_pts,
                                           (len(tx),))
            acc[m]["smu"].append(smu)
            acc[m]["ssig"].append(ssig)

    methods = [m for m in methods if acc[m]["mu"]]
    mu_g = {m: np.mean(acc[m]["mu"], 0) for m in methods}
    sig_g = {m: np.mean(acc[m]["sig"], 0) for m in methods}
    mu_s = {m: np.mean(acc[m]["smu"], 0) for m in methods}
    sig_s = {m: np.mean(acc[m]["ssig"], 0) for m in methods}

    # ── coverage and signed exceedance fields ───────────────────────────────
    covers, fields = {}, {}
    for m in methods:
        inside = np.abs(g_true - mu_g[m]) <= k_sigma * sig_g[m]
        covers[m] = 100.0 * inside.mean()
        dev = g_true - mu_g[m]
        fields[m] = np.where(inside, np.nan, dev)   # only uncovered points
        print(f"     {fu.METHOD_LABEL[m]:<4} coverage = {covers[m]:6.2f} %")

    # ── figure ──────────────────────────────────────────────────────────────
    ncol = len(methods)
    fig = plt.figure(figsize=(4.1 * ncol, 7.4))
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.0, 1.15],
                          hspace=0.30, wspace=0.22)

    xlab, ylab = fu.case_axis_labels(case)
    clab = fu.case_closure_label(case)
    colors = {"bnn": "#d62728", "vgp": "#2ca02c", "dkl": "#1f77b4"}

    # top: diagonal slice
    ax = fig.add_subplot(gs[0, :])
    ax.plot(tx, g_slice, "k-", lw=2.2, label=clab, zorder=5)
    for m in methods:
        c = colors.get(m, None)
        ax.plot(tx, mu_s[m], "-", color=c, lw=1.6,
                label=fu.METHOD_LABEL[m])
        ax.fill_between(tx, mu_s[m] - k_sigma * sig_s[m],
                        mu_s[m] + k_sigma * sig_s[m],
                        color=c, alpha=0.18, lw=0)
    ax.set_xlabel(xlab, fontsize = 11)
    ax.set_ylabel(clab, fontsize = 11)
    ax.grid(alpha=0.25)
    ax.legend(ncol=len(methods) + 1, fontsize=11, loc="upper left",
              framealpha=0.9)
    # ax.set_title(f"Diagonal slice through the evaluation grid "
    #              f"($\\pm{k_sigma:g}\\sigma$ bands)", fontsize=10)

    # bottom: signed exceedance contours
    finite = [np.abs(f[np.isfinite(f)]) for f in fields.values()]
    finite = [a for a in finite if a.size]
    vmax = max(1e-6, max(a.max() for a in finite)) if finite else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    dev_levels = np.linspace(-vmax, vmax, 21)

    ox = oy = None
    if meta_last is not None:
        ox, oy = fu.training_footprint(meta_last, sys_mod)

    im = None
    for j, m in enumerate(methods):
        axc = fig.add_subplot(gs[1, j])
        axc.set_facecolor("#f2f2f2")
        im = axc.contourf(gx, gy, fields[m], levels=dev_levels,
                          cmap="RdBu_r", norm=norm, extend="both")
        axc.contour(gx, gy, np.isfinite(fields[m]).astype(float),
                    levels=[0.5], colors="0.45", linewidths=0.7)
        if ox is not None:
            axc.scatter(ox, oy, s=9, c="0.25", alpha=0.85, zorder=4,
                        linewidths=0)
        axc.plot([gx[0], gx[-1]], [gy[0], gy[-1]], "k--", lw=0.8, alpha=0.6)
        axc.set_xlim(gx[0], gx[-1])
        axc.set_ylim(gy[0], gy[-1])
        axc.set_xlabel(xlab, fontsize = 11)
        if j == 0:
            axc.set_ylabel(ylab, fontsize = 11)
        axc.set_title(f"{fu.METHOD_LABEL[m]} : {covers[m]:.0f}%", fontsize=11)

    if im is not None:
        cbar = fig.colorbar(im, ax=fig.axes[1:], fraction=0.025, pad=0.02)
        # cbar.set_label(r"$g(\cdot) - \mu(\cdot)$ where outside "
        #                f"$\\pm{k_sigma:g}\\sigma$", fontsize=9)

    out_dir = fu.ensure_outdir(case, obs)
    name = cfg["out_name"] or f"spatial_coverage_{case}_N{n_train}.png"
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return covers


if __name__ == "__main__":
    main()
