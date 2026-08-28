# -*- coding: utf-8 -*-
"""
sensitivity_samples_per_traj.py — Sensitivity study 2 (DKL only).

Holds the number of training trajectories N FIXED and varies the number of
measurement times per trajectory, K. This isolates measurement sparsity from
state-space coverage: varying N changes how many initial conditions are
explored and therefore which regions of the state space the trajectories visit,
whereas varying K changes only how densely each trajectory is sampled in time.

Reports the average grid NLL (Eq. 15) and the mean absolute error of the
predictive mean against the withheld generating kinetics, both on the FIXED
evaluation grid of config.GRID_BOUNDS (Eq. 14).

NOTE ON K: the measurement indices are drawn once per K from the integration
grid, excluding t = 0, using the same helper the systems use at import
(config.get_obs_indices). At small K the particular draw matters, so the study
re-draws them per repeat by default (`vary_obs_seed`).

Run from an IDE (no arguments) or:  python sensitivity_samples_per_traj.py
"""

# ════════════════════════════════════════════════════════════════════════
# CONFIG — edit these, then press Run.
# ════════════════════════════════════════════════════════════════════════
STUDY_CONFIG = {
    "case":       "bioreactor",            # "cstr" or "bioreactor"
    "n_train":    5,                 # N held FIXED
    "epochs":     1000,
    "n_repeats":  5,
    "k_values":   [2, 3, 5, 8, 10, 15],   # measurements per trajectory, K
    "vary_obs_seed": True,           # re-draw measurement times each repeat
    "save_models": True,             # checkpoint every fit
    "grid_n":     None,
    "out_prefix": None,
}
# ════════════════════════════════════════════════════════════════════════

import os
import csv
import pickle

import numpy as np

import matplotlib
import matplotlib as mpl
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import sensitivity_utils as su


def _outdir(case):
    d = os.path.join("results", f"sensitivity_{case}")
    os.makedirs(d, exist_ok=True)
    return d


def main(cfg=None):
    cfg = dict(STUDY_CONFIG if cfg is None else cfg)
    case = cfg["case"]
    ks = list(cfg["k_values"])
    n_rep = int(cfg["n_repeats"])
    n_train = int(cfg["n_train"])

    n_int = {"cstr": config.CSTR_N_INTEGRATION,
             "bioreactor": config.BIOREACTOR_N_INTEGRATION,
             "lv": config.LV_N_INTEGRATION}[case]
    ks = [k for k in ks if k < n_int]

    print(f"\n{'='*70}\nSENSITIVITY 2: measurements per trajectory (DKL only)"
          f"\n  {su.system_label(case)} | N fixed at {n_train} | "
          f"K in {ks} | {n_rep} repeats\n{'='*70}")

    sys_mod = su.import_system(case)
    rows = []
    for k in ks:
        print(f"\n  K = {k}")
        for rep in range(n_rep):
            # Re-draw the measurement times per repeat so the result is not an
            # artefact of one particular set of sampling instants.
            obs_seed = (config.SEED + 7919 * rep) if cfg["vary_obs_seed"] \
                else config.SEED
            prev = su.set_obs_per_traj(sys_mod, case, k, seed=obs_seed)
            try:
                print(f"    [repeat {rep+1}/{n_rep}]", end=" ", flush=True)
                sd = (os.path.join(_outdir(case), "checkpoint",
                                   f"K_{case}_N{n_train}")
                      if cfg.get("save_models", True) else None)
                m, _ = su.fit_dkl_once(case, n_train, cfg["epochs"],
                                       seed_offset=rep, grid_n=cfg["grid_n"],
                                       save_dir=sd, tag=f"K{k}_rep{rep}")
                m["k_obs"] = int(k)
                m["repeat"] = rep
                m["obs_seed"] = int(obs_seed)
                rows.append(m)
                print(f"NLL={m['nll']:+.4f}  MAE={m['mae']:.4f}  "
                      f"({m['train_s']:.1f}s)")
            finally:
                sys_mod.OBS_INDICES = prev

    out = _outdir(case)
    prefix = cfg["out_prefix"] or f"sensitivity_K_{case}_N{n_train}"

    csv_path = os.path.join(out, f"{prefix}.csv")
    with open(csv_path, "w", newline="") as f:
        # loss_curve is a full per-epoch history; keep it in the .pkl only
        fields = [k for k in rows[0].keys() if k != "loss_curve"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, f"{prefix}.pkl"), "wb") as f:
        pickle.dump({"rows": rows, "config": cfg}, f)

    nll = su.summarize(rows, ["k_obs"], "nll")
    mae = su.summarize(rows, ["k_obs"], "mae")
    sig = su.summarize(rows, ["k_obs"], "mean_sigma")

    m_nll = np.array([nll[(k,)][0] for k in ks])
    s_nll = np.array([nll[(k,)][1] for k in ks])
    m_mae = np.array([mae[(k,)][0] for k in ks])
    s_mae = np.array([mae[(k,)][1] for k in ks])
    m_sig = np.array([sig[(k,)][0] for k in ks])
    s_sig = np.array([sig[(k,)][1] for k in ks])

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1))
    for ax, mu, sd, lab in zip(
            axes, [m_nll, m_mae, m_sig], [s_nll, s_mae, s_sig],
            ["Average grid NLL",
             "MAE of predictive mean",
             r"Mean predictive $\sigma$ over grid"]):
        ax.plot(ks, mu, "o-", color="#1f77b4", lw=1.8, ms=5)
        ax.fill_between(ks, mu - sd, mu + sd, color="#1f77b4", alpha=0.18,
                        lw=0)
        ax.set_xlabel("Measurements per trajectory, $K$")
        ax.set_ylabel(lab)
        ax.set_xticks(ks)
        ax.grid(alpha=0.25)
    axes[0].set_yscale("symlog", linthresh=1e-2)
    # fig.suptitle(f"{su.system_label(case)}: DKL sensitivity to measurement "
    #              f"sparsity at fixed $N={n_train}$ "
    #              f"(mean $\\pm$ 1 s.d. over {n_rep} repeats)", fontsize=12)
    fig.tight_layout()
    png = os.path.join(out, f"{prefix}.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"\n  {'K':<6}{'NLL':>20}{'MAE':>20}{'mean sigma':>18}")
    for k in ks:
        print(f"  {k:<6}{nll[(k,)][0]:>12.4f} ± {nll[(k,)][1]:<5.3f}"
              f"{mae[(k,)][0]:>12.4f} ± {mae[(k,)][1]:<5.3f}"
              f"{sig[(k,)][0]:>11.4f} ± {sig[(k,)][1]:<5.3f}")
    print(f"\n  Saved: {png}\n         {csv_path}")




# ══════════════════════════════════════════════════════════════════════
# PLOT-ONLY ENTRY POINT (no training)
# ══════════════════════════════════════════════════════════════════════

K_PLOT_OPTS = {
    "source":    "pkl",          # "pkl" (fast) or "checkpoints" (re-evaluate)
    "metrics":   ["nll", "mae"], # nll, mae, rmse, mean_sigma
    "metric_label": {
        "nll":        "Average grid NLL",
        "mae":        "MAE of predictive mean",
        "rmse":       "RMSE of predictive mean",
        "mean_sigma": r"Mean predictive $\sigma$ over grid",
    },
    "cases":     None,           # None -> cfg["case"]; or ["cstr","bioreactor"]
    "symlog":    ["nll"],        # metrics drawn on a symmetric-log axis
    "symlog_lin": 1.0,           # linear region half-width for symlog
    "logx":      False,          # log-spaced K axis
    "show_points": True,         # overlay the individual repeats
    "errorbar":  "sd",           # "sd" | "sem" | "minmax"
    "figsize_per_panel": (5.3, 4.2),
    "legend_first_only": True,
    "out_suffix": "_pub",
    "dpi":       400,
}

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
MARKERS = ["o", "s", "^", "D"]
LINESTYLES = ["-", "--", "-.", ":"]

PUB_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "lines.linewidth": 1.8,
    "legend.frameon": False,
    "savefig.bbox": "tight",
}


def _rows_from_pkl_K(case, prefix):
    path = os.path.join(_outdir(case), f"{prefix}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved results at {path}. Either run main() once, or set "
            f"K_PLOT_OPTS['source'] = 'checkpoints'.")
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["rows"], d.get("config", {})


def _rows_from_checkpoints_K(case, cfg):
    """Reload every saved DKL fit for this sweep and re-evaluate it.

    Useful when the .pkl was overwritten by a later partial run, or when a
    metric is wanted that the original run did not record.
    """
    import torch
    from surrogates.dkl import DKLFeat, FeatureGP, DKLWrapper
    from surrogates.traj_train import compute_standardizer

    n_train = int(cfg.get("n_train", 5))
    root = os.path.join(_outdir(case), "checkpoint", f"K_{case}_N{n_train}")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"No checkpoints at {root}.")

    sys_mod = su.import_system(case)
    grid, g_true, _, _ = su.build_grid(case, sys_mod, cfg.get("grid_n"))

    rows = []
    for tag in sorted(os.listdir(root)):
        d = os.path.join(root, tag)
        if not os.path.isdir(d) or not tag.startswith("K"):
            continue
        try:                                    # tag: K{k}_rep{r}
            parts = tag.split("_")
            k = int(parts[0][1:])
            rep = int(parts[1][3:])
        except (IndexError, ValueError):
            print(f"    skipping unrecognized tag '{tag}'")
            continue

        with open(os.path.join(d, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        std = compute_standardizer(meta["full_train"], sys_mod)
        feat = DKLFeat(meta["input_dim"], meta["hidden_dim"],
                       meta["dkl_feature_dim"], standardizer=std)
        svgp = FeatureGP(
            torch.zeros(meta["n_inducing"], meta["dkl_feature_dim"]),
            meta["dkl_feature_dim"])
        svgp.load_state_dict(torch.load(os.path.join(d, "dkl_part0.pt"),
                                        map_location="cpu"))
        feat.load_state_dict(torch.load(os.path.join(d, "dkl_part1.pt"),
                                        map_location="cpu"))
        feat.eval()
        svgp.eval()
        wrapper = DKLWrapper(feat, svgp, standardizer=std)

        m = su.grid_metrics((svgp, feat, wrapper), grid, g_true)
        m.update({"k_obs": k, "repeat": rep, "n_train": meta.get("n_train"),
                  "train_s": np.nan})
        rows.append(m)
        print(f"    {tag}: NLL={m['nll']:+.4f}  MAE={m['mae']:.4f}")

    if not rows:
        raise RuntimeError(f"No usable checkpoints under {root}.")
    return rows, cfg


def _agg_K(vals, mode):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, 0.0, 0.0
    m = v.mean()
    if mode == "minmax":
        return m, m - v.min(), v.max() - m
    e = v.std(ddof=1) / np.sqrt(v.size) if (mode == "sem" and v.size > 1) \
        else v.std()
    return m, e, e


def _cells_K(rows, metric):
    """{K: [per-repeat values]}"""
    from collections import defaultdict
    d = defaultdict(list)
    for r in rows:
        if metric in r and np.isfinite(r.get(metric, np.nan)):
            d[r["k_obs"]].append(r[metric])
    return d


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.18, linewidth=0.7)


def _K_panel(ax, rows_by_case, metric, opts, show_legend=True):
    """`metric` against K, one line per case study."""
    for i, (case, rows) in enumerate(rows_by_case.items()):
        cells = _cells_K(rows, metric)
        if not cells:
            continue
        ks = sorted(cells)
        col = PALETTE[i % len(PALETTE)]
        mk = MARKERS[i % len(MARKERS)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        mu, lo, hi = [], [], []
        for k in ks:
            m, l, h = _agg_K(cells[k], opts["errorbar"])
            mu.append(m); lo.append(l); hi.append(h)
            if opts["show_points"]:
                span = (max(ks) - min(ks)) or 1.0
                jit = 0.012 * (i - (len(rows_by_case) - 1) / 2) * span
                ax.scatter([k + jit] * len(cells[k]), cells[k], s=13,
                           color=col, alpha=0.28, zorder=1, linewidths=0)
        ax.errorbar(ks, mu, yerr=[lo, hi], fmt=mk, linestyle=ls, color=col,
                    ms=6, capsize=3.5, capthick=1.0, elinewidth=1.2,
                    zorder=3, label=su.system_label(case))

    all_ks = sorted({k for rows in rows_by_case.values()
                     for k in _cells_K(rows, metric)})
    ax.set_xlabel("Measurements per trajectory, $K$")
    ax.set_ylabel(opts["metric_label"].get(metric, metric))
    ax.set_xticks(all_ks)
    if opts["logx"]:
        ax.set_xscale("log")
        ax.set_xticklabels([str(k) for k in all_ks])
    if metric in opts.get("symlog", []):
        ax.set_yscale("symlog", linthresh=opts.get("symlog_lin", 1.0))
        ax.axhline(0.0, color="0.6", lw=0.8, ls=":", zorder=0)
    _style_axes(ax)
    if show_legend and len(rows_by_case) > 1:
        ax.legend(loc="best", handlelength=2.2, borderaxespad=0.4)


def main_plot(cfg=None, opts=None):
    """Redraw the K sweep from saved results. Trains nothing.

    With one case this gives a single line per panel; with
    `opts={"cases": ["cstr", "bioreactor"]}` both benchmarks are overlaid on
    the same axes with a shared legend, which is usually the more compact
    presentation for a paper.
    """
    cfg = dict(STUDY_CONFIG if cfg is None else cfg)
    opts = dict(K_PLOT_OPTS, **(opts or {}))
    cases = list(opts["cases"] or [cfg["case"]])
    n_train = int(cfg["n_train"])

    rows_by_case = {}
    for case in cases:
        prefix = cfg["out_prefix"] or f"sensitivity_K_{case}_N{n_train}"
        try:
            if opts["source"] == "checkpoints":
                rows, _ = _rows_from_checkpoints_K(case, {**cfg,
                                                          "case": case})
            else:
                rows, _ = _rows_from_pkl_K(case, prefix)
            rows_by_case[case] = rows
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  [{case}] {e}")
    if not rows_by_case:
        raise FileNotFoundError("No saved results found for any case.")

    n_rep = max(len({r.get("repeat", 0) for r in rows})
                for rows in rows_by_case.values())
    for case, rows in rows_by_case.items():
        ks = sorted(_cells_K(rows, "nll"))
        print(f"  {su.system_label(case)}: {len(rows)} fits | K={ks} | "
              f"{n_rep} repeats")

    mets = [m for m in opts["metrics"]
            if any(np.isfinite(r.get(m, np.nan))
                   for rows in rows_by_case.values() for r in rows)]

    w, h = opts["figsize_per_panel"]
    with mpl.rc_context(PUB_STYLE):
        fig, axes = plt.subplots(1, len(mets), figsize=(w * len(mets), h),
                                 squeeze=False)
        axes = axes[0]
        for i, met in enumerate(mets):
            _K_panel(axes[i], rows_by_case, met, opts,
                     show_legend=(i == 0 or
                                  not opts["legend_first_only"]))
        # fig.suptitle(f"DKL sensitivity to measurement sparsity at fixed "
        #              f"$N = {n_train}$ ({n_rep} repeats, "
        #              f"bars = {opts['errorbar']})", fontsize=12)
        fig.tight_layout()

        out = _outdir(cases[0])
        stem = (f"sensitivity_K_N{n_train}" if len(cases) > 1
                else f"sensitivity_K_{cases[0]}_N{n_train}")
        png = os.path.join(out, f"{stem}{opts['out_suffix']}.png")
        fig.savefig(png, dpi=opts["dpi"])
        fig.savefig(png.replace(".png", ".pdf"))
        plt.close(fig)

    for case, rows in rows_by_case.items():
        print(f"\n  {su.system_label(case)}")
        hdr = "".join(f"{m:>20}" for m in mets)
        print(f"  {'K':<6}{hdr}")
        cells = {m: _cells_K(rows, m) for m in mets}
        for k in sorted(cells[mets[0]]):
            line = ""
            for m in mets:
                mu, lo, _ = _agg_K(cells[m].get(k, []), opts["errorbar"])
                line += f"{mu:>12.4f} ± {lo:<5.3f}"
            print(f"  {k:<6}{line}")
    print(f"\n  Saved: {png}")
    return rows_by_case


if __name__ == "__main__":
    main_plot()