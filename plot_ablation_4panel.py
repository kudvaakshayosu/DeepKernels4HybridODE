import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- Style (journal quality) -----------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 15,
    "axes.labelsize": 23,
    "legend.fontsize": 15,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
})

# ----------------------- GLOBAL CONFIG -----------------------
USE_LOG_NLL = True      # toggle log scale for NLL
USE_LOG_MAE = True     # toggle log scale for MAE
CLIP_FOR_VIS = True     # clip negatives ONLY for visualization
CLIP_MIN = 1e-4         # floor for log stability
SHOW_LEGEND = True      # legend only in first plot
SAVE = True             # write the figure to disk

# Axis labels adapt to the case recorded in the pickle.
STATE_LABEL = {
    "cstr":       r"($C_A$, $C_B$)",
    "bioreactor": r"($X$, $S$)",
    "lv":         r"($x_1$, $x_2$)",
}

# ----------------------- COLORS -----------------------
COLORS = {
    "bnn": "#D55E00",
    "vgp": "#009E73",
    "dkl": "#0072B2",
}

LABELS = {
    "bnn": "BNN",
    "vgp": "VGP",
    "dkl": "DKL",
}

METHODS = ["bnn", "vgp", "dkl"]


# ----------------------- Helpers -----------------------
def _mean_sd(values):
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, 0.0
    return arr.mean(), arr.std()


def _series(d, key, n_values):
    """
    Pull a per-N (mean, sd) series for `key` from whichever UQ store
    the pickle has: prefer all_uq_rep, else all_uq.
    """
    rep = d.get("all_uq_rep")
    flat = d.get("all_uq")

    out = {}

    for m in METHODS:
        means, sds = [], []

        for N in n_values:

            if rep and N in rep and m in rep[N] and key in rep[N][m]:
                mu, sd = _mean_sd(rep[N][m][key])

            elif flat and N in flat and m in flat[N] and key in flat[N][m]:
                mu, sd = _mean_sd(flat[N][m][key])

            else:
                mu, sd = np.nan, 0.0

            means.append(mu)
            sds.append(sd)

        out[m] = (np.array(means), np.array(sds))

    return out


def _series_traj(d, n_values):
    """
    Per-N (mean, sd) of trajectory coverage from
    all_results[N]['coverage'].
    """
    ar = d.get("all_results", {})
    out = {}

    for m in METHODS:
        means, sds = [], []

        for N in n_values:
            block = ar.get(N, {}).get("coverage", {})

            if m in block:
                mu, sd = _mean_sd(block[m])
            else:
                mu, sd = np.nan, 0.0

            means.append(mu)
            sds.append(sd)

        out[m] = (np.array(means), np.array(sds))

    return out


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.18, linewidth=0.7)


def _plot_panel(ax, n_values, series, ylabel, title,
                percent=False, logy=False,
                show_legend=False, bounded=False):

    for m in METHODS:
        mean, sd = series[m]
        if np.all(np.isnan(mean)):
            continue

        # ---------------- SAFE CLIP FOR LOG VISUALIZATION ----------------
        if logy and CLIP_FOR_VIS:
            mean = np.clip(mean, CLIP_MIN, None)
            sd = np.clip(sd, 0, None)

        c = COLORS[m]

        ax.plot(
            n_values,
            mean,
            "-o",
            color=c,
            lw=2.2,
            ms=5.5,
            label=LABELS[m]
        )

        # ---------------- UNCERTAINTY BAND ----------------
        lower = mean - 1.96*sd/np.sqrt(5)
        upper = mean + 1.96*sd/np.sqrt(5)

        # Keep trajectory coverage clouds within [0,1]
        if bounded:
            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)

        # MAE is non-negative; do not let the band cross zero.
        if logy and CLIP_FOR_VIS:
            lower = np.clip(lower, CLIP_MIN, None)

        ax.fill_between(
            n_values,
            lower,
            upper,
            color=c,
            alpha=0.15
        )

    ax.set_xlabel(r"Training trajectories, $N$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if percent:
        ax.set_ylim(-2, 102)
        ax.axhline(100, ls=":", lw=1.0, color="gray")

    if logy:
        ax.set_yscale("log")
        ax.set_ylim(bottom=CLIP_MIN)

        ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs="auto"))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())

    ax.set_xticks(n_values)
    _style_axes(ax)

    if show_legend:
        ax.legend(frameon=False)


# ----------------------- MAIN -----------------------
def main(pkl_path, out_path=None):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    n_values = sorted(d.get("n_values") or d.get("all_results", {}).keys())
    case = d.get("case", "bioreactor")
    grid = STATE_LABEL.get(case, r"($x_1$, $x_2$)")

    nll = _series(d, "nll", n_values)
    mae = _series(d, "mae", n_values)
    area = _series(d, "coverage_area", n_values)
    traj = _series_traj(d, n_values)

    if all(np.all(np.isnan(mae[m][0])) for m in METHODS):
        raise KeyError(
            "No 'mae' entries in this pickle. It predates the grid-MAE metric; "
            "re-run run_ablation.py to record it.")

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.4))

    # ---------------- PANEL 1 (LEGEND + LOG SCALE) ----------------
    _plot_panel(
        axes[0], n_values, nll,
        ylabel=f"Average NLL over {grid} grid",
        title="",
        logy=USE_LOG_NLL,
        show_legend=False
    )

    # ---------------- PANEL 2 (MEAN ABSOLUTE ERROR) ----------------
    _plot_panel(
        axes[1],
        n_values,
        mae,
        ylabel=f"MAE over {grid} grid",
        title="",
        logy=USE_LOG_MAE,
    )

    # ---------------- PANEL 3 ----------------
    area_percent = {
        m: (100.0 * area[m][0], 100.0 * area[m][1])
        for m in METHODS
    }

    _plot_panel(
        axes[2],
        n_values,
        area_percent,
        ylabel=f"% Coverage Area over {grid} grid",
        title="",
        percent=True,
        show_legend=False
    )

    # ---------------- PANEL 4 ----------------
    _plot_panel(
        axes[3],
        n_values,
        traj,
        ylabel="Fraction Coverage over Test",
        title="",
        show_legend=SHOW_LEGEND,
        bounded=True
    )

    fig.tight_layout()

    if SAVE:
        if out_path is None:
            out_path = os.path.splitext(pkl_path)[0] + "_4panel.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        fig.savefig(os.path.splitext(out_path)[0] + ".png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"Saved publication-quality figure -> {out_path}")

    # ---------------- CONSOLE TABLE ----------------
    print(f"\n{'N':<6}{'method':<8}{'NLL':>12}{'MAE':>12}"
          f"{'%cov area':>12}{'traj cov':>12}")
    for i, N in enumerate(n_values):
        for m in METHODS:
            print(f"{N:<6}{LABELS[m]:<8}"
                  f"{nll[m][0][i]:>12.4f}{mae[m][0][i]:>12.4f}"
                  f"{100*area[m][0][i]:>12.1f}{traj[m][0][i]:>12.3f}")


if __name__ == "__main__":  
    CASE = "CSTR"          # or "bioreactor"
    default = f"results/ablation_{CASE}_full/checkpoint/ablation_results.pkl"
    pkl = sys.argv[1] if len(sys.argv) > 1 else default
    main(pkl)
