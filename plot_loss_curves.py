# -*- coding: utf-8 -*-
"""
plot_loss_curves.py — training loss vs epoch for the ablation sweep.

Draws the optimization history recorded by `run_ablation.py`: for each
N_TRAIN, the training objective against epoch for every model, with the spread
across repeats shown as a band. The diagnostic of interest is whether the loss
has flattened by the end of the training budget — i.e. whether 1000 epochs is
enough — which is what a reviewer asking about convergence wants to see.

Reads `results/ablation_<case>_<obs>/checkpoint/ablation_results.pkl`, so no
retraining is needed. The objective differs between models (the deterministic
DNN minimizes the measurement NLL, the probabilistic closures a negative ELBO),
so curves are comparable across N and repeats for a GIVEN model, not across
models; each model therefore gets its own panel row by default.

Run from an IDE (no arguments) or:  python plot_loss_curves.py
"""

# ════════════════════════════════════════════════════════════════════════
# CONFIG — edit these, then press Run.
# ════════════════════════════════════════════════════════════════════════
PLOT_CONFIG = {
    "case":      "cstr",        # "cstr" or "bioreactor"
    "obs":       "full",
    "models":    None,          # None -> every model found in the results
    "n_values":  None,          # None -> every N found in the results
    "layout":    "by_model",    # "by_model": one panel per model (N overlaid)
                                # "by_n":     one panel per N (models overlaid)
    "logy":      True,          # log scale on the loss axis
    "smooth":    1,             # moving-average window in epochs (1 = none)
    "show_band": True,          # min-max band across repeats
    "out_name":  None,          # None -> loss_curves_<case>.png
}
# ════════════════════════════════════════════════════════════════════════

import os
import pickle

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_LABEL = {"mlp": "DNN", "bnn": "BNN", "vgp": "VGP", "dkl": "DKL"}
MODEL_COLOR = {"mlp": "#7f7f7f", "bnn": "#d62728",
               "vgp": "#2ca02c", "dkl": "#1f77b4"}
SYS_DIR = {"cstr": "cstr", "bioreactor": "bioreactor",
           "lv": "lotka_volterra"}
SYS_LABEL = {"cstr": "CSTR", "bioreactor": "Bioreactor",
             "lv": "Lotka-Volterra"}


def _results_path(case, obs):
    return os.path.join("results", f"ablation_{SYS_DIR[case]}_{obs}",
                        "checkpoint", "ablation_results.pkl")


def _load(case, obs):
    path = _results_path(case, obs)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No ablation results at {path}.\n"
            f"Run run_ablation.py for case='{case}' first.")
    with open(path, "rb") as f:
        saved = pickle.load(f)
    curves = saved.get("loss_curves")
    if not curves:
        raise KeyError(
            f"{path} contains no 'loss_curves'. It was probably written by an "
            f"older run; re-run run_ablation.py to record them.")
    return {int(k): v for k, v in curves.items()}, saved


def _smooth(y, w):
    if w is None or w <= 1 or len(y) < w:
        return np.asarray(y, dtype=float)
    k = np.ones(int(w)) / float(w)
    return np.convolve(np.asarray(y, dtype=float), k, mode="valid")


def _stack(curves, w):
    """Align repeat curves to a common length and smooth. -> [n_rep, T]"""
    cs = [_smooth(c, w) for c in curves if c is not None and len(c)]
    if not cs:
        return None
    T = min(len(c) for c in cs)
    return np.stack([c[:T] for c in cs], 0)


def main(cfg=None):
    cfg = dict(PLOT_CONFIG if cfg is None else cfg)
    case, obs = cfg["case"], cfg["obs"]
    loss_curves, saved = _load(case, obs)

    n_values = cfg["n_values"] or sorted(loss_curves.keys())
    found = []
    for N in n_values:
        for nm in loss_curves.get(N, {}):
            if nm not in found and any(loss_curves[N][nm]):
                found.append(nm)
    order = [m for m in ["mlp", "bnn", "vgp", "dkl"] if m in found]
    models = cfg["models"] or order
    models = [m for m in models if m in found]
    if not models:
        raise ValueError("No loss curves found for the requested models.")

    n_rep = saved.get("n_repeats", "?")
    print(f"  Loss curves | {SYS_LABEL[case]} | N={n_values} | "
          f"models={[MODEL_LABEL[m] for m in models]} | repeats={n_rep}")

    w = int(cfg.get("smooth", 1) or 1)
    band = bool(cfg.get("show_band", True))

    if cfg["layout"] == "by_n":
        panels, panel_label = n_values, lambda N: f"$N = {N}$"
        series = lambda N: [(nm, loss_curves.get(N, {}).get(nm, []),
                             MODEL_LABEL[nm], MODEL_COLOR[nm])
                            for nm in models]
    else:
        panels, panel_label = models, lambda m: MODEL_LABEL[m]
        cmap = plt.get_cmap("viridis")
        shades = {N: cmap(i / max(len(n_values) - 1, 1))
                  for i, N in enumerate(n_values)}
        series = lambda m: [(N, loss_curves.get(N, {}).get(m, []),
                             f"$N={N}$", shades[N]) for N in n_values]

    ncol = len(panels)
    fig, axes = plt.subplots(1, ncol, figsize=(3.7 * ncol, 3.7),
                             squeeze=False, sharex=True)
    axes = axes[0]

    for ax, key in zip(axes, panels):
        for _tag, curves, label, color in series(key):
            arr = _stack(curves, w)
            if arr is None:
                continue
            ep = np.arange(arr.shape[1])
            mu = arr.mean(0)
            ax.plot(ep, mu, "-", color=color, lw=1.5, label=label)
            if band and arr.shape[0] > 1:
                ax.fill_between(ep, arr.min(0), arr.max(0), color=color,
                                alpha=0.15, lw=0)
        ax.set_title(panel_label(key), fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        if cfg.get("logy", True):
            ax.set_yscale("symlog", linthresh=1e-2)
        ax.legend(fontsize=7, framealpha=0.9)

    axes[0].set_ylabel("Training objective")
    sm = f", {w}-epoch moving average" if w > 1 else ""
    fig.suptitle(f"{SYS_LABEL[case]}: training loss vs epoch "
                 f"(band = min-max over repeats{sm})", fontsize=12)
    fig.tight_layout()

    out_dir = os.path.join("results", f"ablation_{SYS_DIR[case]}_{obs}",
                           "figures")
    os.makedirs(out_dir, exist_ok=True)
    name = cfg["out_name"] or f"loss_curves_{case}.png"
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    # convergence summary: how much the loss still moves over the last 10%
    print(f"\n  {'model':<8}{'N':<6}{'final loss':>14}{'drift over last 10%':>22}")
    for m in models:
        for N in n_values:
            arr = _stack(loss_curves.get(N, {}).get(m, []), 1)
            if arr is None:
                continue
            mu = arr.mean(0)
            tail = mu[int(0.9 * len(mu)):]
            drift = float(tail[0] - tail[-1]) if len(tail) > 1 else 0.0
            rel = abs(drift) / max(abs(mu[-1]), 1e-9)
            print(f"  {MODEL_LABEL[m]:<8}{N:<6}{mu[-1]:>14.4f}"
                  f"{drift:>14.4f} ({100*rel:.1f}%)")
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
