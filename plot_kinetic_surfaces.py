# -*- coding: utf-8 -*-
"""
plot_kinetic_surfaces.py — Fig. 5 of the manuscript.

Recovered representations of the withheld generating kinetics.

    Top row    : the true kinetics, followed by each method's predictive MEAN
                 over the evaluation grid.
    Bottom row : each probabilistic method's predictive STANDARD DEVIATION
                 (epistemic uncertainty). The deterministic DNN has none and is
                 left blank.

Maroon points mark the state-space locations visited by the training
trajectories and are identical across models.

Models are reloaded from the ablation checkpoints written by run_ablation.py,
on the FIXED grid in config.GRID_BOUNDS (Eq. 14).

Run from an IDE (no arguments) or:  python plot_kinetic_surfaces.py
"""

# ════════════════════════════════════════════════════════════════════════
# CONFIG — edit these, then press Run.
# ════════════════════════════════════════════════════════════════════════
PLOT_CONFIG = {
    "case":     "cstr",        # "cstr" or "bioreactor"
    "obs":      "full",
    "n_train":  5,             # which N_TRAIN checkpoint to plot
    "rep":      2,             # which repeat (None -> average over all repeats)
    "methods":  ["mlp", "bnn", "vgp", "dkl"],
    "n_mc":     1000,          # MC weight samples for the BNN predictive
    "grid_n":   None,          # None -> config.GRID_N
    "out_name": None,          # None -> kinetic_surfaces_<case>_N<n>.png
}
# ════════════════════════════════════════════════════════════════════════

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import figure_utils as fu


def main(cfg=None):
    cfg = dict(PLOT_CONFIG if cfg is None else cfg)
    case, obs = cfg["case"], cfg["obs"]
    n_train = int(cfg["n_train"])
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
    print(f"  Kinetic surfaces | {fu.system_label(case)} | N={n_train} | "
          f"repeats={reps}")

    acc = {m: {"mu": [], "sig": []} for m in methods}
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

    methods = [m for m in methods if acc[m]["mu"]]
    mu_g = {m: np.mean(acc[m]["mu"], 0) for m in methods}
    sig_g = {m: np.mean(acc[m]["sig"], 0) for m in methods}

    ox, oy = fu.training_footprint(meta_last, sys_mod)
    xlab, ylab = fu.case_axis_labels(case)
    clab = fu.case_closure_label(case)

    # shared colour scales
    all_means = [g_true] + [mu_g[m] for m in methods]
    mmax = max(np.abs(a).max() for a in all_means)
    mlev = np.linspace(-mmax, mmax, 21)
    prob = [m for m in methods if m != "mlp"]
    smax = max([sig_g[m].max() for m in prob], default=1.0)
    slev = np.linspace(0.0, max(smax, 1e-9), 21)

    ncol = 1 + len(methods)
    fig, axes = plt.subplots(2, ncol, figsize=(3.05 * ncol, 6.2),
                             sharex=True, sharey=True)
    if ncol == 1:
        axes = axes.reshape(2, 1)

    # ── top row: true kinetics + predictive means ───────────────────────────
    im_mean = axes[0, 0].contourf(gx, gy, g_true, levels=mlev, cmap="RdBu_r")
    axes[0, 0].scatter(ox, oy, s=8, c="maroon", zorder=4, linewidths=0,
                       label="Training points")
    axes[0, 0].set_title(f"True kinetics: {clab}", fontsize=10)
    axes[0, 0].set_ylabel(ylab)
    axes[0, 0].legend(fontsize=7, loc="upper right", framealpha=0.9)

    for j, m in enumerate(methods):
        ax = axes[0, j + 1]
        ax.contourf(gx, gy, mu_g[m], levels=mlev, cmap="RdBu_r")
        ax.scatter(ox, oy, s=8, c="maroon", zorder=4, linewidths=0)
        ax.set_title(fu.METHOD_LABEL[m], fontsize=11)

    cb1 = fig.colorbar(im_mean, ax=axes[0, :].tolist(), fraction=0.020,
                       pad=0.012)
    cb1.set_label("Predictive mean (closure units)", fontsize=8)

    # ── bottom row: predictive standard deviations ──────────────────────────
    axes[1, 0].axis("off")
    im_sig = None
    first_sig_ax = None
    for j, m in enumerate(methods):
        ax = axes[1, j + 1]
        if m == "mlp":
            ax.axis("off")
            ax.text(0.5, 0.5, "DNN\n(no epistemic\nuncertainty)",
                    ha="center", va="center", fontsize=10, color="0.35",
                    transform=ax.transAxes)
            continue
        im_sig = ax.contourf(gx, gy, sig_g[m], levels=slev, cmap="viridis")
        ax.scatter(ox, oy, s=8, c="maroon", zorder=4, linewidths=0)
        ax.set_xlabel(xlab)
        if first_sig_ax is None:
            first_sig_ax = ax
            ax.set_ylabel(ylab)

    if im_sig is not None:
        cb2 = fig.colorbar(im_sig, ax=axes[1, :].tolist(), fraction=0.020,
                           pad=0.012)
        cb2.set_label("Predictive std. dev. (epistemic)", fontsize=8)

    for ax in axes.ravel():
        if ax.has_data():
            ax.set_xlim(gx[0], gx[-1])
            ax.set_ylim(gy[0], gy[-1])

    out_dir = fu.ensure_outdir(case, obs)
    name = cfg["out_name"] or f"kinetic_surfaces_{case}_N{n_train}.png"
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
