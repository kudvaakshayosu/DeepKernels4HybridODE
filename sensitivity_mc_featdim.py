# -*- coding: utf-8 -*-
"""
sensitivity_mc_featdim.py — Sensitivity study 1.

Sweeps the number of Monte Carlo rollouts per gradient step, M_S, against the
DKL latent feature dimension, and reports for each combination:

    * average grid NLL of the withheld generating kinetics (Eq. 15)
    * mean absolute error of the predictive MEAN against those kinetics

Both are computed on the FIXED evaluation grid of config.GRID_BOUNDS (Eq. 14),
so the numbers are directly comparable to the main ablation.

Outputs a heat map for each metric plus a CSV of the raw per-repeat values.

Run from an IDE (no arguments) or:  python sensitivity_mc_featdim.py
"""

# ════════════════════════════════════════════════════════════════════════
# CONFIG — edit these, then press Run.
# ════════════════════════════════════════════════════════════════════════
STUDY_CONFIG = {
    "case":       "bioreactor",          # "cstr" or "bioreactor"
    "n_train":    5,               # N held fixed for this study
    "epochs":     1000,
    "n_repeats":  3,               # repeats per (M_S, feat_dim) cell
    "mc_values":  [3, 6, 10, 20],  # Monte Carlo rollouts per gradient step
    "feat_dims":  [3, 4, 5, 6],    # DKL latent feature dimension
    "save_models": True,           # checkpoint every fit
    "grid_n":     None,            # None -> config.GRID_N
    "out_prefix": None,            # None -> sensitivity_mc_featdim_<case>
}
# ════════════════════════════════════════════════════════════════════════

import os
import csv
from collections import defaultdict
import itertools
import pickle

import numpy as np

import torch

import matplotlib
import matplotlib as mpl
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sensitivity_utils as su


def _outdir(case):
    d = os.path.join("results", f"sensitivity_{case}")
    os.makedirs(d, exist_ok=True)
    return d


def _heatmap(ax, M, xt, yt, title, cbar_label, lower_is_better=True):
    cmap = "viridis_r" if lower_is_better else "viridis"
    im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(xt)))
    ax.set_xticklabels(xt)
    ax.set_yticks(range(len(yt)))
    ax.set_yticklabels(yt)
    ax.set_xlabel("DKL feature dimension, $n_z$")
    ax.set_ylabel("MC rollouts per gradient step, $M_S$")
    ax.set_title(title, fontsize=11)
    finite = M[np.isfinite(M)]
    mid = (finite.max() + finite.min()) / 2 if finite.size else 0.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if M[i, j] < mid else "black")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=9)


def main(cfg=None):
    cfg = dict(STUDY_CONFIG if cfg is None else cfg)
    case = cfg["case"]
    mcs = list(cfg["mc_values"])
    fds = list(cfg["feat_dims"])
    n_rep = int(cfg["n_repeats"])

    print(f"\n{'='*70}\nSENSITIVITY 1: M_S x feature dim | {su.system_label(case)} | "
          f"N={cfg['n_train']} | {len(mcs)}x{len(fds)} cells x {n_rep} repeats"
          f"\n{'='*70}")

    rows = []
    for mc, fd in itertools.product(mcs, fds):
        print(f"\n  M_S={mc}, n_z={fd}")
        for rep in range(n_rep):
            print(f"    [repeat {rep+1}/{n_rep}]", end=" ", flush=True)
            sd = (os.path.join(_outdir(case), "checkpoint",
                               f"mc_featdim_{case}")
                  if cfg.get("save_models", True) else None)
            m, _ = su.fit_dkl_once(case, cfg["n_train"], cfg["epochs"],
                                   seed_offset=rep, feat_dim=fd, n_mc=mc,
                                   grid_n=cfg["grid_n"], save_dir=sd,
                                   tag=f"mc{mc}_nz{fd}_rep{rep}")
            m["repeat"] = rep
            rows.append(m)
            print(f"NLL={m['nll']:+.4f}  MAE={m['mae']:.4f}  "
                  f"({m['train_s']:.1f}s)")

    out = _outdir(case)
    prefix = cfg["out_prefix"] or f"sensitivity_mc_featdim_{case}"

    # raw CSV
    csv_path = os.path.join(out, f"{prefix}.csv")
    with open(csv_path, "w", newline="") as f:
        # loss_curve is a full per-epoch history; keep it in the .pkl only
        fields = [k for k in rows[0].keys() if k != "loss_curve"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, f"{prefix}.pkl"), "wb") as f:
        pickle.dump({"rows": rows, "config": cfg}, f)

    # aggregate -> matrices [len(mcs), len(fds)]
    nll = su.summarize(rows, ["n_mc", "feat_dim"], "nll")
    mae = su.summarize(rows, ["n_mc", "feat_dim"], "mae")
    sec = su.summarize(rows, ["n_mc", "feat_dim"], "train_s")
    M_nll = np.array([[nll[(mc, fd)][0] for fd in fds] for mc in mcs])
    M_mae = np.array([[mae[(mc, fd)][0] for fd in fds] for mc in mcs])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    _heatmap(axes[0], M_nll, fds, mcs,
             "Average grid NLL", "NLL (lower is better)")
    _heatmap(axes[1], M_mae, fds, mcs,
             "Mean absolute error of predictive mean",
             "MAE (lower is better)")
    fig.suptitle(f"{su.system_label(case)}: sensitivity to $M_S$ and $n_z$ "
                 f"($N={cfg['n_train']}$, {n_rep} repeats)", fontsize=12)
    fig.tight_layout()
    png = os.path.join(out, f"{prefix}.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    # console summary table
    print(f"\n  {'M_S':<6}{'n_z':<6}{'NLL':>18}{'MAE':>18}{'train s':>12}")
    for mc in mcs:
        for fd in fds:
            mn, sn = nll[(mc, fd)]
            ma, sa = mae[(mc, fd)]
            ts, _ = sec[(mc, fd)]
            print(f"  {mc:<6}{fd:<6}{mn:>10.4f} ± {sn:<5.3f}"
                  f"{ma:>10.4f} ± {sa:<5.3f}{ts:>12.1f}")
    print(f"\n  Saved: {png}\n         {csv_path}")




# ══════════════════════════════════════════════════════════════════════
# PLOT-ONLY ENTRY POINT (no training)
# ══════════════════════════════════════════════════════════════════════

PLOT_OPTS = {
    "source":    "checkpoints",      # "pkl" or "checkpoints"
    "layout":    "lines",            # "lines" | "paired" | "cost"
    "metrics":   ["nll", "mae"],
    "metric_label": {
        "nll":        "Average grid NLL",
        "mae":        "MAE over Grid",
        "rmse":       "RMSE of predictive mean",
        "mean_sigma": r"Mean predictive $\sigma$",
        "train_s":    "Training time (s)",
    },
    "normalize":   "none",     # "none" | "minmax" | "z"
    "symlog":      ["nll"],    # metrics drawn on a symlog axis when un-scaled
    "symlog_lin":  1.0,        # linear region half-width for symlog
    "legend_title": {"mc": "$n_z$", "fd": "$M_S$"},
    "legend_loc":  "best",
    "legend_ncol": 1,
    "legend_first_only": True,  # legend on the first panel only
    "show_range":  True,       # print the absolute range in the axis label
    "baseline":    None,       # (M_S, n_z) for "paired"; None -> first cell
    "logy":        [],
    "show_points": True,
    "errorbar":    "sd",       # "sd" | "sem" | "minmax"
    "figsize_per_panel": (5.3, 4.2),
    "out_suffix":  "_pub",
    "dpi":         400,
}

# Colour-blind-safe palette (Okabe-Ito), consistent with the ablation figures.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--"]

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


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.18, linewidth=0.7)


def _scale(cells, mode):
    """Return (transform, inverse_range_text) for the requested normalization.

    `transform` maps a (mean, lo, hi) triple to the rescaled triple. Error bars
    are scaled by the same linear factor, so their widths stay comparable.
    """
    vals = np.concatenate([np.asarray(v, float) for v in cells.values()]) \
        if cells else np.array([np.nan])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or mode == "none":
        return (lambda m, l, h: (m, l, h)), ""

    if mode == "minmax":
        lo_v, hi_v = vals.min(), vals.max()
        rng = hi_v - lo_v
        if rng <= 0:
            return (lambda m, l, h: (0.5, 0.0, 0.0)), " (constant)"
        return ((lambda m, l, h: ((m - lo_v) / rng, l / rng, h / rng)),
                f"\n[{lo_v:.4g}, {hi_v:.4g}]")

    if mode == "z":
        mu, sd = vals.mean(), vals.std()
        if sd <= 0:
            return (lambda m, l, h: (0.0, 0.0, 0.0)), " (constant)"
        return ((lambda m, l, h: ((m - mu) / sd, l / sd, h / sd)),
                f"\n$\\mu$={mu:.4g}, $\\sigma$={sd:.4g}")

    raise ValueError(f"unknown normalize '{mode}'")


def _ylabel(metric, opts, rng_txt):
    base = opts["metric_label"].get(metric, metric)
    if opts["normalize"] == "minmax":
        base = f"{base}\n(min-max scaled)"
    elif opts["normalize"] == "z":
        base = f"{base} (standardized)"
    if opts["show_range"] and rng_txt:
        base += rng_txt
    return base


def _lines_panel(ax, rows, metric, opts, show_legend=True):
    """`metric` against the DKL feature dimension, one line per M_S."""
    cells = _cells(rows, metric)
    tf, rng_txt = _scale(cells, opts["normalize"])
    xs = sorted({k[1] for k in cells})          # feature dimensions
    ss = sorted({k[0] for k in cells})          # M_S values
    span = (max(xs) - min(xs)) if len(xs) > 1 else 1.0

    for i, mc in enumerate(ss):
        col = PALETTE[i % len(PALETTE)]
        mk = MARKERS[i % len(MARKERS)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        mu, lo, hi, xv = [], [], [], []
        for x in xs:
            key = (mc, x)
            if key not in cells:
                continue
            m, l, h = tf(*_agg(cells[key], opts["errorbar"]))
            mu.append(m); lo.append(l); hi.append(h); xv.append(x)
            if opts["show_points"]:
                pts = [tf(p, 0, 0)[0] for p in cells[key]]
                jit = 0.035 * (i - (len(ss) - 1) / 2) * span
                ax.scatter([x + jit] * len(pts), pts, s=13, color=col,
                           alpha=0.28, zorder=1, linewidths=0)
        if not xv:
            continue
        ax.errorbar(xv, mu, yerr=[lo, hi], fmt=mk, linestyle=ls, color=col,
                    ms=6, capsize=3.5, capthick=1.0, elinewidth=1.2,
                    zorder=3, label=f"$M_S = {mc}$")

    ax.set_xlabel("DKL feature dimension, $n_z$", fontsize = 15)
    ax.set_ylabel(_ylabel(metric, opts, rng_txt), fontsize = 15)
    ax.set_xticks(xs)
    ax.tick_params(axis="y", labelsize=15)
    ax.tick_params(axis="x", labelsize=15)
    ax.set_xlim(min(xs) - 0.35 * span / max(len(xs) - 1, 1),
                max(xs) + 0.35 * span / max(len(xs) - 1, 1))
    if opts["normalize"] == "minmax":
        ax.set_ylim(-0.08, 1.08)
    elif metric in opts.get("symlog", []):
        ax.set_yscale("symlog", linthresh=opts.get("symlog_lin", 1.0))
        ax.axhline(0.0, color="0.6", lw=0.8, ls=":", zorder=0)
    _style_axes(ax)
    if not show_legend:
        return
    leg = ax.legend(loc=opts.get("legend_loc", "best"),
                    ncol=opts.get("legend_ncol", 1),
                    #title=opts.get("legend_title", {}).get("fd", "$M_S$"),
                    handlelength=2.2, borderaxespad=0.4, fontsize = 12)
    leg.get_title().set_fontsize(10)


def _paired_panel(ax, rows, metric, opts):
    seeded = _by_seed(rows, metric)
    keys = sorted(seeded)
    base = tuple(opts["baseline"]) if opts["baseline"] else keys[0]
    if base not in seeded:
        raise ValueError(f"baseline {base} not among cells {keys}")

    labs, mu, lo, hi = [], [], [], []
    raw = {}
    for k in keys:
        if k == base:
            continue
        shared = sorted(set(seeded[k]) & set(seeded[base]))
        if not shared:
            continue
        d = [seeded[k][r] - seeded[base][r] for r in shared]
        raw[k] = d
        m, l, h = _agg(d, opts["errorbar"])
        labs.append(f"{k[0]}, {k[1]}")
        mu.append(m); lo.append(l); hi.append(h)

    # Differences are already relative; min-max scaling would obscure the
    # meaningful zero, so it is deliberately not applied in this layout.
    x = np.arange(len(labs))
    resolved = [(m - l > 0) or (m + h < 0) for m, l, h in zip(mu, lo, hi)]

    for i, k in enumerate([k for k in keys if k != base and k in raw]):
        if opts["show_points"]:
            ax.scatter([i] * len(raw[k]), raw[k], s=14, color="0.55",
                       alpha=0.55, zorder=1, linewidths=0)

    ax.errorbar(x, mu, yerr=[lo, hi], fmt="none", ecolor="0.30",
                capsize=3.5, capthick=1.0, elinewidth=1.3, zorder=3)
    ax.scatter(x, mu, s=52, zorder=4, edgecolor="white", linewidth=0.8,
               c=[PALETTE[0] if r else "0.62" for r in resolved])
    ax.axhline(0.0, color="k", lw=1.0, ls="--", alpha=0.75, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=9.5)
    ax.set_xlabel("$M_S$, $n_z$")
    ax.set_ylabel(f"$\\Delta$ {opts['metric_label'].get(metric, metric)}\n"
                  f"vs $M_S$={base[0]}, $n_z$={base[1]}")
    ax.set_title("filled = interval excludes zero", fontsize=9.5)
    _style_axes(ax)


def _cost_panel(ax, rows, metric, opts):
    cells_m = _cells(rows, metric)
    cells_t = _cells(rows, "train_s")
    if not cells_t:
        ax.text(0.5, 0.5, "no train\\_s recorded\n(use source='pkl')",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="0.4")
        ax.set_axis_off()
        return
    tf, rng_txt = _scale(cells_m, opts["normalize"])
    mcs = sorted({k[0] for k in cells_m})
    for k in sorted(cells_m):
        if k not in cells_t:
            continue
        my, ey, _ = tf(*_agg(cells_m[k], opts["errorbar"]))
        mx, ex, _ = _agg(cells_t[k], opts["errorbar"])
        col = PALETTE[mcs.index(k[0]) % len(PALETTE)]
        ax.errorbar(mx, my, yerr=ey, xerr=ex, fmt="o", ms=6, capsize=3,
                    color=col, elinewidth=1.1, zorder=3)
        ax.annotate(f"({k[0]}, {k[1]})", (mx, my), fontsize=8.5,
                    xytext=(5, 4), textcoords="offset points", color="0.35")
    ax.set_xlabel("Training wall-clock per fit (s)")
    ax.set_ylabel(_ylabel(metric, opts, rng_txt))
    ax.set_title(r"labels: ($M_S$, $n_z$)", fontsize=9.5)
    if opts["normalize"] == "minmax":
        ax.set_ylim(-0.08, 1.08)
    _style_axes(ax)


def main_plot(cfg=None, opts=None):
    cfg = dict(STUDY_CONFIG if cfg is None else cfg)
    opts = dict(PLOT_OPTS, **(opts or {}))
    case = cfg["case"]
    prefix = cfg["out_prefix"] or f"sensitivity_mc_featdim_{case}"

    if opts["source"] == "checkpoints":
        rows, _ = _rows_from_checkpoints(case, {**cfg, **opts})
    else:
        rows, _ = _rows_from_pkl(case, prefix)

    mcs = sorted({r["n_mc"] for r in rows})
    fds = sorted({r["feat_dim"] for r in rows})
    n_rep = len({r.get("repeat", 0) for r in rows})
    print(f"  {len(rows)} fits | M_S={mcs} | n_z={fds} | {n_rep} repeats "
          f"| normalize={opts['normalize']}")

    mets = [m for m in opts["metrics"]
            if any(np.isfinite(r.get(m, np.nan)) for r in rows)]
    layout = opts["layout"]
    ncol = len(mets)
    w, h = opts["figsize_per_panel"]

    with mpl.rc_context(PUB_STYLE):
        fig, axes = plt.subplots(1, ncol, figsize=(w * ncol, h),
                                 squeeze=False)
        axes = axes[0]
        for i, met in enumerate(mets):
            if layout == "lines":
                _lines_panel(axes[i], rows, met, opts,
                             show_legend=(i == 0 or not
                                          opts.get("legend_first_only", True)))
            elif layout == "paired":
                _paired_panel(axes[i], rows, met, opts)
            elif layout == "cost":
                _cost_panel(axes[i], rows, met, opts)
            else:
                raise ValueError(f"unknown layout '{layout}'")

        #fig.suptitle(f"{su.system_label(case)}: DKL sensitivity to $M_S$ and "
                     # f"$n_z$  ($N = {cfg['n_train']}$, {n_rep} repeats, "
                     # f"bars = {opts['errorbar']})", fontsize=12)
        fig.tight_layout()

        out = _outdir(case)
        png = os.path.join(out, f"{prefix}{opts['out_suffix']}.png")
        fig.savefig(png, dpi=opts["dpi"])
        fig.savefig(png.replace(".png", ".pdf"))
        plt.close(fig)

    print(f"  Saved: {png}")
    return rows


def _rows_from_pkl(case, prefix):
    path = os.path.join(_outdir(case), f"{prefix}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved results at {path}. Either run main() once, or set "
            f"PLOT_OPTS['source'] = 'checkpoints'.")
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["rows"], d.get("config", {})


def _rows_from_checkpoints(case, cfg):
    """Reload every saved DKL fit and re-evaluate it on the grid."""
    from surrogates.dkl import DKLFeat, FeatureGP, DKLWrapper
    from surrogates.traj_train import compute_standardizer

    root = os.path.join(_outdir(case), "checkpoint", f"mc_featdim_{case}")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"No checkpoints at {root}.")

    sys_mod = su.import_system(case)
    grid, g_true, _, _ = su.build_grid(case, sys_mod, cfg.get("grid_n"))

    rows = []
    for tag in sorted(os.listdir(root)):
        d = os.path.join(root, tag)
        if not os.path.isdir(d) or not tag.startswith("mc"):
            continue
        try:                                   # tag: mc{mc}_nz{fd}_rep{r}
            parts = tag.split("_")
            mc = int(parts[0][2:])
            fd = int(parts[1][2:])
            rep = int(parts[2][3:])
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
        m.update({"n_mc": mc, "feat_dim": fd, "repeat": rep,
                  "n_train": meta.get("n_train"), "train_s": np.nan})
        rows.append(m)
        print(f"    {tag}: NLL={m['nll']:+.4f}  MAE={m['mae']:.4f}")

    if not rows:
        raise RuntimeError(f"No usable checkpoints under {root}.")
    return rows, cfg


def _agg(vals, mode):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, 0.0, 0.0
    m = v.mean()
    if mode == "minmax":
        return m, m - v.min(), v.max() - m
    e = v.std(ddof=1) / np.sqrt(v.size) if (mode == "sem" and v.size > 1) \
        else v.std()
    return m, e, e


def _cells(rows, metric):
    """{(M_S, n_z): [per-repeat values]}"""
    d = defaultdict(list)
    for r in rows:
        if metric in r and np.isfinite(r.get(metric, np.nan)):
            d[(r["n_mc"], r["feat_dim"])].append(r[metric])
    return d


def _by_seed(rows, metric):
    """{(M_S, n_z): {repeat: value}} — needed for the paired layout."""
    d = defaultdict(dict)
    for r in rows:
        if metric in r and np.isfinite(r.get(metric, np.nan)):
            d[(r["n_mc"], r["feat_dim"])][r.get("repeat", 0)] = r[metric]
    return d



if __name__ == "__main__":
    main_plot()