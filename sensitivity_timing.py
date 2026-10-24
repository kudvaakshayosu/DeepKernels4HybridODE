# -*- coding: utf-8 -*-
"""
sensitivity_timing.py — Sensitivity study 3 (DKL only): cost breakdown.

Profiles where training time is actually spent for the DKL closure, and how the
breakdown scales with the settings varied in the other two studies.

The training step is instrumented into four components:

    GP posterior   forward pass through the sparse variational GP, including
                   the Cholesky factorization of K_ZZ and the posterior
                   moments in Eqs. (6)-(7)
    feature map    forward pass through the neural feature extractor h_w
    integrator     everything else inside the rollout, i.e. the ODE solver
                   stepping the mechanistic equations forward
    backward       gradient computation through all of the above, plus the
                   optimizer update

Components are measured by wrapping the relevant modules with timing hooks, so
the split reflects the actual call structure rather than an estimate. The
GP-posterior and feature-map times are contained WITHIN the rollout, so the
integrator figure reported is the rollout time net of those two.

Outputs a stacked-bar breakdown plus a CSV.

Run from an IDE (no arguments) or:  python sensitivity_timing.py
"""

# ════════════════════════════════════════════════════════════════════════
# CONFIG — edit these, then press Run.
# ════════════════════════════════════════════════════════════════════════
STUDY_CONFIG = {
    "case":      "cstr",           # "cstr" or "bioreactor"
    "n_train":   5,
    "epochs":    100,              # profiling needs far fewer epochs
    "n_repeats": 3,
    # Configurations to profile. Each entry overrides the defaults.
    "configs": [
        {"label": "$M_S$=3, $n_z$=6",   "n_mc": 3,  "feat_dim": 6},
        {"label": "$M_S$=6, $n_z$=6",   "n_mc": 6,  "feat_dim": 6},
        {"label": "$M_S$=10, $n_z$=6",  "n_mc": 10, "feat_dim": 6},
        {"label": "$M_S$=20, $n_z$=6",  "n_mc": 20, "feat_dim": 6},
        {"label": "$M_S$=3, $n_z$=3",   "n_mc": 3,  "feat_dim": 3},
        {"label": "$M_S$=3, $n_z$=4",   "n_mc": 3,  "feat_dim": 4},
        {"label": "$M_S$=3, $n_z$=5",   "n_mc": 3,  "feat_dim": 5},
    ],
    "out_prefix": None,
}
# ════════════════════════════════════════════════════════════════════════

import os
import csv
import time
import pickle
from collections import defaultdict

import numpy as np
import torch

import matplotlib
import matplotlib as mpl
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import sensitivity_utils as su

COMPONENTS = ["gp_posterior", "feature_map", "integrator", "backward"]
COMP_LABEL = {
    "gp_posterior": "(i) GP posterior",
    "feature_map":  "(ii) Feature extractor $h_w$",
    "integrator":   "(iii) ODE integrator",
    "backward":     "(iv) Backward pass + Optimizer step",
}
COMP_COLOR = {
    "gp_posterior": "#1f77b4",
    "feature_map":  "#2ca02c",
    "integrator":   "#ff7f0e",
    "backward":     "#d62728",
}


class _TimedWrapper(torch.nn.Module):
    """Drop-in replacement for DKLWrapper that times its two internal stages.

    GPyTorch overrides ``Module.__call__``, so PyTorch forward hooks never fire
    on the SVGP. The wrapper's forward is therefore instrumented directly: the
    feature map and the GP posterior (including the Cholesky factorization of
    K_ZZ, the posterior moments and the reparameterized sample) are timed as
    separate stages of the same call.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.feat = inner.feat
        self.svgp = inner.svgp
        self.std = inner.std
        self.acc = defaultdict(float)
        self.calls = defaultdict(int)

    def forward(self, x):
        t0 = time.perf_counter()
        z = self.feat(x)
        t1 = time.perf_counter()
        dist = self.svgp(z)
        out = dist.rsample() if self.training else dist.mean
        out = out.unsqueeze(-1) if out.dim() == 1 else out
        if self.std is not None:
            out = self.std.decode_g(out)
        t2 = time.perf_counter()
        self.acc["feature_map"] += t1 - t0
        self.acc["gp_posterior"] += t2 - t1
        self.calls["feature_map"] += 1
        self.calls["gp_posterior"] += 1
        return out


def _profile_one(case, n_train, epochs, seed_offset, n_mc, feat_dim):
    """Train DKL for `epochs` with timing hooks; return per-component seconds."""
    import random
    import gpytorch
    from surrogates.dkl import DKLFeat, FeatureGP, DKLWrapper, _init_inducing_dkl
    from surrogates.traj_train import (rollout, gaussian_obs_nll,
                                       compute_standardizer)

    sys_mod = su.import_system(case)
    seed = config.SEED + 1000 * seed_offset
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    (x0_batch, y_obs, t_grid, obs_indices, noise_var_diag, full_train) = \
        su._make_training_data(sys_mod, case, n_train, seed_offset)

    std = compute_standardizer(full_train, sys_mod)
    feat = DKLFeat(sys_mod.INPUT_DIM, config.HIDDEN_DIM, feat_dim,
                   standardizer=std)
    Z = _init_inducing_dkl(x0_batch, sys_mod, feat, config.N_INDUCING)
    svgp = FeatureGP(Z, feat_dim)
    wrapper = _TimedWrapper(DKLWrapper(feat, svgp, standardizer=std))
    factory = sys_mod.make_hybrid_factory()
    hybrid = factory(wrapper)
    C_obs = sys_mod.get_C()
    n_data = int(x0_batch.shape[0] * t_grid.shape[0] * 10)

    inducing_params = [svgp.variational_strategy.inducing_points]
    gp_params = [p for n, p in svgp.named_parameters()
                 if "inducing_points" not in n]
    opt = torch.optim.Adam([
        {"params": feat.parameters(), "lr": config.LR_GP * 0.1},
        {"params": inducing_params, "lr": config.LR_GP * 0.01},
        {"params": gp_params, "lr": config.LR_GP},
    ])
    warmup = max(int(epochs * 0.3), 1)
    feat.train(); svgp.train(); wrapper.train()

    rollout_s = 0.0
    backward_s = 0.0
    t_total = time.perf_counter()
    with gpytorch.settings.cholesky_jitter(float_value=1e-4, double_value=1e-6):
        for ep in range(epochs):
            opt.zero_grad()
            beta = min(1.0, ep / warmup)
            total_nll = 0.0
            t_roll = time.perf_counter()
            for _ in range(n_mc):
                traj = rollout(hybrid, x0_batch, t_grid)
                total_nll = total_nll + gaussian_obs_nll(
                    traj, y_obs, obs_indices, C_obs, noise_var_diag)
            rollout_s += time.perf_counter() - t_roll
            total_nll = total_nll / n_mc
            kl = svgp.variational_strategy.kl_divergence().sum() / n_data
            loss = total_nll + beta * kl
            if torch.isnan(loss) or torch.isinf(loss):
                opt.zero_grad()
                continue
            t_bwd = time.perf_counter()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(svgp.parameters(), 5.0)
            opt.step()
            backward_s += time.perf_counter() - t_bwd
    total_s = time.perf_counter() - t_total

    gp_s = wrapper.acc["gp_posterior"]
    ft_s = wrapper.acc["feature_map"]
    # The closure calls happen inside the rollout; the integrator's own cost is
    # the rollout time net of them.
    integ_s = max(rollout_s - gp_s - ft_s, 0.0)

    return {"gp_posterior": gp_s, "feature_map": ft_s,
            "integrator": integ_s, "backward": backward_s,
            "total_s": total_s, "rollout_s": rollout_s,
            "gp_calls": wrapper.calls["gp_posterior"],
            "feat_calls": wrapper.calls["feature_map"],
            "n_mc": n_mc, "feat_dim": feat_dim, "epochs": epochs}


def main(cfg=None):
    cfg = dict(STUDY_CONFIG if cfg is None else cfg)
    case = cfg["case"]
    n_rep = int(cfg["n_repeats"])

    print(f"\n{'='*70}\nSENSITIVITY 3: DKL cost breakdown | "
          f"{su.system_label(case)} | N={cfg['n_train']} | "
          f"{cfg['epochs']} epochs | {n_rep} repeats\n{'='*70}")

    rows = []
    for c in cfg["configs"]:
        print(f"\n  {c['label']}")
        for rep in range(n_rep):
            print(f"    [repeat {rep+1}/{n_rep}]", end=" ", flush=True)
            r = _profile_one(case, cfg["n_train"], cfg["epochs"], rep,
                             c["n_mc"], c["feat_dim"])
            r["label"] = c["label"]
            r["repeat"] = rep
            rows.append(r)
            frac = {k: 100 * r[k] / max(r["total_s"], 1e-9)
                    for k in COMPONENTS}
            print(f"total={r['total_s']:.1f}s | "
                  + " ".join(f"{k.split('_')[0]}={frac[k]:.0f}%"
                             for k in COMPONENTS))

    out = os.path.join("results", f"sensitivity_{case}")
    os.makedirs(out, exist_ok=True)
    prefix = cfg["out_prefix"] or f"sensitivity_timing_{case}"

    csv_path = os.path.join(out, f"{prefix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, f"{prefix}.pkl"), "wb") as f:
        pickle.dump({"rows": rows, "config": cfg}, f)

    labels = [c["label"] for c in cfg["configs"]]
    means = {k: [] for k in COMPONENTS}
    totals = []
    for lab in labels:
        sel = [r for r in rows if r["label"] == lab]
        for k in COMPONENTS:
            means[k].append(np.mean([r[k] for r in sel]))
        totals.append(np.mean([r["total_s"] for r in sel]))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6))

    # absolute stacked seconds
    bottom = np.zeros(len(labels))
    for k in COMPONENTS:
        axes[0].bar(x, means[k], bottom=bottom, label=COMP_LABEL[k],
                    color=COMP_COLOR[k], width=0.62)
        bottom += np.array(means[k])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel(f"Wall-clock time for {cfg['epochs']} epochs (s)")
    axes[0].set_title("Absolute cost", fontsize=11)
    axes[0].grid(alpha=0.25, axis="y")

    # share of total
    bottom = np.zeros(len(labels))
    for k in COMPONENTS:
        share = 100 * np.array(means[k]) / np.maximum(np.array(totals), 1e-9)
        axes[1].bar(x, share, bottom=bottom, color=COMP_COLOR[k], width=0.62)
        bottom += share
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Share of training time (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Relative cost", fontsize=11)
    axes[1].grid(alpha=0.25, axis="y")

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle(f"{su.system_label(case)}: where DKL training time is spent "
                 f"(mean over {n_rep} repeats)", fontsize=12)
    fig.tight_layout()
    png = os.path.join(out, f"{prefix}.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"\n  {'configuration':<20}{'total s':>10}"
          + "".join(f"{k.split('_')[0]:>13}" for k in COMPONENTS))
    for i, lab in enumerate(labels):
        plain = lab.replace("$", "").replace("\\", "")
        print(f"  {plain:<20}{totals[i]:>10.1f}"
              + "".join(f"{100*means[k][i]/max(totals[i],1e-9):>12.1f}%"
                        for k in COMPONENTS))
    print(f"\n  Saved: {png}\n         {csv_path}")




# ══════════════════════════════════════════════════════════════════════
# PLOT-ONLY ENTRY POINT (no profiling)
# ══════════════════════════════════════════════════════════════════════

TIMING_PLOT_OPTS = {
    "normalize":   "max",      # "none" | "max" | "minmax"
    "panels":      ["absolute", "relative"],   # either or both
    "sort_by":     None,       # None (config order) | "total" | "label"
    "annotate_total": True,    # print the total above each bar
    "errorbar":    "sd",       # error bar on the total: "sd" | "sem" | "none"
    "figsize_per_panel": (6.6, 4.6),
    "bar_width":   0.64,
    "out_suffix":  "_pub",
    "dpi":         400,
}

PUB_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 13,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "legend.frameon": False,
    "savefig.bbox": "tight",
}

# Okabe-Ito, consistent with the other sensitivity figures.
COMP_COLOR_PUB = {
    "gp_posterior": "#0072B2",
    "feature_map":  "#009E73",
    "integrator":   "#E69F00",
    "backward":     "#D55E00",
}


def _rows_from_pkl_timing(case, prefix):
    path = os.path.join("results", f"sensitivity_{case}", f"{prefix}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved timings at {path}. Run main() once first — the timing "
            f"study saves no model checkpoints, so there is nothing else to "
            f"reload from.")
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["rows"], d.get("config", {})


def _spread(vals, mode):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0 or mode == "none":
        return 0.0
    return v.std(ddof=1) / np.sqrt(v.size) if (mode == "sem" and v.size > 1) \
        else v.std()


def main_plot(cfg=None, opts=None):
    """Redraw the DKL cost breakdown from saved timings. Profiles nothing."""
    cfg = dict(STUDY_CONFIG if cfg is None else cfg)
    opts = dict(TIMING_PLOT_OPTS, **(opts or {}))
    case = cfg["case"]
    prefix = cfg["out_prefix"] or f"sensitivity_timing_{case}"

    rows, saved_cfg = _rows_from_pkl_timing(case, prefix)
    epochs = saved_cfg.get("epochs", cfg.get("epochs"))
    n_rep = len({r.get("repeat", 0) for r in rows})

    labels = []
    for r in rows:                       # preserve first-seen order
        if r["label"] not in labels:
            labels.append(r["label"])

    means = {k: [] for k in COMPONENTS}
    totals, tot_err = [], []
    for lab in labels:
        sel = [r for r in rows if r["label"] == lab]
        for k in COMPONENTS:
            means[k].append(float(np.mean([r[k] for r in sel])))
        totals.append(float(np.mean([r["total_s"] for r in sel])))
        tot_err.append(_spread([r["total_s"] for r in sel], opts["errorbar"]))

    if opts["sort_by"] == "total":
        order = np.argsort(totals)
    elif opts["sort_by"] == "label":
        order = np.argsort(labels)
    else:
        order = np.arange(len(labels))
    labels = [labels[i] for i in order]
    totals = [totals[i] for i in order]
    tot_err = [tot_err[i] for i in order]
    means = {k: [means[k][i] for i in order] for k in COMPONENTS}

    print(f"  {len(rows)} profiles | {len(labels)} configurations | "
          f"{n_rep} repeats | normalize={opts['normalize']}")

    # ── normalization of the absolute panel ────────────────────────────────
    t = np.asarray(totals, float)
    if opts["normalize"] == "max":
        denom = t.max() if t.max() > 0 else 1.0
        scale = np.full_like(t, 1.0 / denom)
        ylab_abs = "Training cost, relative to slowest setting"
        note = f"slowest = {t.max():.1f} s for {epochs} epochs"
    elif opts["normalize"] == "minmax":
        rng = t.max() - t.min()
        scale = (t - t.min()) / (rng if rng > 0 else 1.0) / \
            np.maximum(t, 1e-12)
        ylab_abs = "Training cost (min-max scaled)"
        note = f"[{t.min():.1f}, {t.max():.1f}] s for {epochs} epochs"
    else:
        scale = np.ones_like(t)
        ylab_abs = f"Wall-clock time for {epochs} epochs (s)"
        note = ""

    x = np.arange(len(labels))
    npanel = len(opts["panels"])
    w, h = opts["figsize_per_panel"]

    with mpl.rc_context(PUB_STYLE):
        fig, axes = plt.subplots(1, npanel, figsize=(w * npanel, h),
                                 squeeze=False)
        axes = axes[0]
        ai = 0

        if "absolute" in opts["panels"]:
            ax = axes[ai]; ai += 1
            bottom = np.zeros(len(labels))
            for k in COMPONENTS:
                vals = np.asarray(means[k]) * scale
                ax.bar(x, vals, bottom=bottom, label=COMP_LABEL[k],
                       color=COMP_COLOR_PUB[k], width=opts["bar_width"],
                       edgecolor="white", linewidth=0.6)
                bottom += vals
            if opts["annotate_total"]:
                for i in range(len(labels)):
                    txt = (f"{totals[i]:.0f} s" if opts["normalize"] == "none"
                           else f"{bottom[i]:.2f}")
                    ax.text(x[i], bottom[i] * 1.02, txt, ha="center",
                            va="bottom", fontsize=8.5, color="0.25")
            if opts["errorbar"] != "none" and opts["normalize"] == "none":
                ax.errorbar(x, totals, yerr=tot_err, fmt="none",
                            ecolor="0.3", capsize=3, elinewidth=1.0)
            ax.set_ylabel(ylab_abs)
            ax.set_title("Cost breakdown" + (f"\n({note})" if note else ""),
                         fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=28, ha="right")
            ax.set_ylim(0, bottom.max() * 1.14)
            ax.grid(alpha=0.18, axis="y", linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if "relative" in opts["panels"]:
            ax = axes[ai]; ai += 1
            bottom = np.zeros(len(labels))
            for k in COMPONENTS:
                share = 100 * np.asarray(means[k]) / np.maximum(t, 1e-9)
                ax.bar(x, share, bottom=bottom, color=COMP_COLOR_PUB[k],
                       width=opts["bar_width"], edgecolor="white",
                       linewidth=0.6, label=COMP_LABEL[k])
                bottom += share
            ax.set_ylabel("Share of training time (\\%)")
            ax.set_title("Relative cost", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=28, ha="right")
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.18, axis="y", linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        handles, lbls = axes[0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.16))
        fig.suptitle(f"{su.system_label(case)}: where DKL training time is "
                     f"spent (mean over {n_rep} repeats)", fontsize=12)
        fig.tight_layout()

        out = os.path.join("results", f"sensitivity_{case}")
        png = os.path.join(out, f"{prefix}{opts['out_suffix']}.png")
        fig.savefig(png, dpi=opts["dpi"])
        fig.savefig(png.replace(".png", ".pdf"))
        plt.close(fig)

    hdr = "".join(f"{k.split('_')[0]:>13}" for k in COMPONENTS)
    print(f"\n  {'configuration':<20}{'total s':>10}{'rel.':>8}{hdr}")
    for i, lab in enumerate(labels):
        plain = lab.replace("$", "").replace("\\", "")
        print(f"  {plain:<20}{totals[i]:>10.1f}{totals[i]/t.max():>8.2f}"
              + "".join(f"{100*means[k][i]/max(totals[i],1e-9):>12.1f}%"
                        for k in COMPONENTS))
    print(f"\n  Saved: {png}")
    return rows




# ══════════════════════════════════════════════════════════════════════
# main_plot_v2 — LEFT PANEL ONLY (cost breakdown), tuned for a single axis
# ══════════════════════════════════════════════════════════════════════

TIMING_PLOT_OPTS_V2 = dict(
    TIMING_PLOT_OPTS,
    panels=["absolute"],          # cost breakdown only
    figsize_per_panel=(7.2, 5.0), # squarer: no second panel to balance
    legend_ncol=2,                # legend under a single axis
    legend_below=-0.24,
    out_suffix="_cost",
)


def main_plot_v2(cfg=None, opts=None):
    """Cost-breakdown panel only. Same data path as main_plot; the relative
    share panel is dropped, since those numbers are reported in the text."""
    cfg = dict(STUDY_CONFIG if cfg is None else cfg)
    opts = dict(TIMING_PLOT_OPTS_V2, **(opts or {}))
    case = cfg["case"]
    prefix = cfg["out_prefix"] or f"sensitivity_timing_{case}"

    rows, saved_cfg = _rows_from_pkl_timing(case, prefix)
    epochs = saved_cfg.get("epochs", cfg.get("epochs"))
    n_rep = len({r.get("repeat", 0) for r in rows})

    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])

    means = {k: [] for k in COMPONENTS}
    totals = []
    for lab in labels:
        sel = [r for r in rows if r["label"] == lab]
        for k in COMPONENTS:
            means[k].append(float(np.mean([r[k] for r in sel])))
        totals.append(float(np.mean([r["total_s"] for r in sel])))

    if opts["sort_by"] == "total":
        order = np.argsort(totals)
    elif opts["sort_by"] == "label":
        order = np.argsort(labels)
    else:
        order = np.arange(len(labels))
    labels = [labels[i] for i in order]
    totals = [totals[i] for i in order]
    means = {k: [means[k][i] for i in order] for k in COMPONENTS}

    t = np.asarray(totals, float)
    if opts["normalize"] == "max":
        scale = np.full_like(t, 1.0 / (t.max() if t.max() > 0 else 1.0))
        ylab = "Training cost (relative to slowest)"
        note = f"slowest = {t.max():.1f} s for {epochs} epochs"
    elif opts["normalize"] == "minmax":
        rng = t.max() - t.min()
        scale = (t - t.min()) / (rng if rng > 0 else 1.0) / np.maximum(t, 1e-12)
        ylab = "Training cost (min-max scaled)"
        note = f"[{t.min():.1f}, {t.max():.1f}] s for {epochs} epochs"
    else:
        scale = np.ones_like(t)
        ylab = f"Wall-clock time for {epochs} epochs (s)"
        note = ""

    print(f"  {len(rows)} profiles | {len(labels)} configurations | "
          f"{n_rep} repeats | normalize={opts['normalize']}")

    x = np.arange(len(labels))
    with mpl.rc_context(PUB_STYLE):
        fig, ax = plt.subplots(figsize=opts["figsize_per_panel"])
        bottom = np.zeros(len(labels))
        for k in COMPONENTS:
            vals = np.asarray(means[k]) * scale
            ax.bar(x, vals, bottom=bottom, label=COMP_LABEL[k],
                   color=COMP_COLOR_PUB[k], width=opts["bar_width"],
                   edgecolor="white", linewidth=0.6)
            bottom += vals
        if opts["annotate_total"]:
            for i in range(len(labels)):
                txt = (f"{totals[i]:.0f} s" if opts["normalize"] == "none"
                       else f"{bottom[i]:.2f}")
                ax.text(x[i], bottom[i] * 1.02, txt, ha="center", va="bottom",
                        fontsize=9, color="0.25")
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, bottom.max() * 1.15)
        ax.grid(alpha=0.18, axis="y", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"{su.system_label(case)}: DKL training cost breakdown"
                     + (f"\n({note}, mean over {n_rep} repeats)" if note
                        else f"\n(mean over {n_rep} repeats)"),
                     fontsize=11)
        ax.legend(loc="upper center", ncol=opts["legend_ncol"],
                  bbox_to_anchor=(0.5, opts["legend_below"]),
                  borderaxespad=0.0)
        fig.tight_layout()

        out = os.path.join("results", f"sensitivity_{case}")
        os.makedirs(out, exist_ok=True)
        png = os.path.join(out, f"{prefix}{opts['out_suffix']}.png")
        fig.savefig(png, dpi=opts["dpi"])
        fig.savefig(png.replace(".png", ".pdf"))
        plt.close(fig)

    print(f"\n  {'configuration':<20}{'total s':>10}{'relative':>10}")
    for i, lab in enumerate(labels):
        plain = lab.replace("$", "").replace("\\", "")
        print(f"  {plain:<20}{totals[i]:>10.1f}{totals[i]/t.max():>10.2f}")
    print(f"\n  Saved: {png}")
    return rows




# ══════════════════════════════════════════════════════════════════════
# main_plot_v3 — BOTH CASE STUDIES SIDE BY SIDE, one shared legend
# ══════════════════════════════════════════════════════════════════════

TIMING_PLOT_OPTS_V3 = dict(
    TIMING_PLOT_OPTS,
    cases=["cstr", "bioreactor"],   # left panel, right panel
    normalize_per_case=True,        # scale each panel by its own slowest run
    figsize_per_panel=(6.4, 4.8),
    legend_ncol=4,
    legend_below=-0.06,             # in figure coords, below both axes
    share_y=True,                   # common y-axis when normalized
    out_suffix="_both",
)


def _timing_summary(case, cfg, opts):
    """Load one case and return (labels, per-component means, totals, epochs)."""
    prefix = cfg["out_prefix"] or f"sensitivity_timing_{case}"
    rows, saved_cfg = _rows_from_pkl_timing(case, prefix)
    epochs = saved_cfg.get("epochs", cfg.get("epochs"))
    n_rep = len({r.get("repeat", 0) for r in rows})

    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])

    means = {k: [] for k in COMPONENTS}
    totals = []
    for lab in labels:
        sel = [r for r in rows if r["label"] == lab]
        for k in COMPONENTS:
            means[k].append(float(np.mean([r[k] for r in sel])))
        totals.append(float(np.mean([r["total_s"] for r in sel])))

    if opts["sort_by"] == "total":
        order = np.argsort(totals)
    elif opts["sort_by"] == "label":
        order = np.argsort(labels)
    else:
        order = np.arange(len(labels))
    labels = [labels[i] for i in order]
    totals = [totals[i] for i in order]
    means = {k: [means[k][i] for i in order] for k in COMPONENTS}
    return labels, means, np.asarray(totals, float), epochs, n_rep


def main_plot_v3(cfg=None, opts=None):
    """Cost breakdown for both benchmarks side by side with a shared legend.

    Each panel is normalized by its own slowest configuration when
    `normalize_per_case` is True, so the two benchmarks are directly comparable
    in shape even though their absolute wall-clock times differ. The absolute
    anchor for each is printed in its subplot title.
    """
    cfg = dict(STUDY_CONFIG if cfg is None else cfg)
    opts = dict(TIMING_PLOT_OPTS_V3, **(opts or {}))
    cases = list(opts["cases"])

    data = {}
    for case in cases:
        try:
            data[case] = _timing_summary(case, cfg, opts)
        except FileNotFoundError as e:
            print(f"  [{case}] {e}")
    cases = [c for c in cases if c in data]
    if not cases:
        raise FileNotFoundError("No timing results found for any case.")

    # global scale when not normalizing per case
    if not opts["normalize_per_case"] and opts["normalize"] == "max":
        gmax = max(data[c][2].max() for c in cases)

    with mpl.rc_context(PUB_STYLE):
        w, h = opts["figsize_per_panel"]
        fig, axes = plt.subplots(1, len(cases), figsize=(w * len(cases), h),
                                 squeeze=False,
                                 sharey=bool(opts["share_y"]
                                             and opts["normalize"] != "none"))
        axes = axes[0]

        for ax, case in zip(axes, cases):
            labels, means, t, epochs, n_rep = data[case]

            if opts["normalize"] == "max":
                denom = (t.max() if opts["normalize_per_case"] else gmax)
                denom = denom if denom > 0 else 1.0
                scale = np.full_like(t, 1.0 / denom)
                ylab = "Training cost (relative to slowest)"
                note = "" #f"slowest = {t.max():.1f} s / {epochs} epochs"
            elif opts["normalize"] == "minmax":
                rng = t.max() - t.min()
                scale = (t - t.min()) / (rng if rng > 0 else 1.0) \
                    / np.maximum(t, 1e-12)
                ylab = "Training cost (min-max scaled)"
                note = f"[{t.min():.1f}, {t.max():.1f}] s / {epochs} epochs"
            else:
                scale = np.ones_like(t)
                ylab = f"Wall-clock time for {epochs} epochs (s)"
                note = ""

            x = np.arange(len(labels))
            bottom = np.zeros(len(labels))
            for k in COMPONENTS:
                vals = np.asarray(means[k]) * scale
                ax.bar(x, vals, bottom=bottom, label=COMP_LABEL[k],
                       color=COMP_COLOR_PUB[k], width=opts["bar_width"],
                       edgecolor="white", linewidth=0.6)
                bottom += vals
            if opts["annotate_total"]:
                for i in range(len(labels)):
                    txt = (f"{t[i]:.0f} s" if opts["normalize"] == "none"
                           else f"{bottom[i]:.2f}")
                    ax.text(x[i], bottom[i] * 1.02, txt, ha="center",
                            va="bottom", fontsize=8.5, color="0.25")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_ylim(0, bottom.max() * 1.16)
            ax.grid(alpha=0.18, axis="y", linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(f"{su.system_label(case)}"
                         + (f"\n({note})" if note else ""), fontsize=10.5)
            if ax is axes[0]:
                ax.set_ylabel(ylab)

        handles, lbls = axes[0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="upper center", ncol=opts["legend_ncol"],
                   bbox_to_anchor=(0.5, opts["legend_below"]), fontsize = 12,
                   borderaxespad=0.0)
        # fig.suptitle("Where DKL training time is spent "
        #              f"(mean over {data[cases[0]][4]} repeats)", fontsize=12)
        fig.tight_layout()

        out = os.path.join("results", f"sensitivity_{cases[0]}")
        os.makedirs(out, exist_ok=True)
        png = os.path.join(out, f"sensitivity_timing{opts['out_suffix']}.png")
        fig.savefig(png, dpi=opts["dpi"])
        fig.savefig(png.replace(".png", ".pdf"))
        plt.close(fig)

    for case in cases:
        labels, means, t, epochs, _ = data[case]
        print(f"\n  {su.system_label(case)}")
        print(f"  {'configuration':<20}{'total s':>10}{'relative':>10}"
              + "".join(f"{k.split('_')[0]:>13}" for k in COMPONENTS))
        for i, lab in enumerate(labels):
            plain = lab.replace("$", "").replace("\\", "")
            print(f"  {plain:<20}{t[i]:>10.1f}{t[i]/t.max():>10.2f}"
                  + "".join(f"{100*means[k][i]/max(t[i],1e-9):>12.1f}%"
                            for k in COMPONENTS))
    print(f"\n  Saved: {png}")
    return data


if __name__ == "__main__":
    main_plot_v3()