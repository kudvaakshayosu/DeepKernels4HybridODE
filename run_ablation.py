# -*- coding: utf-8 -*-
"""
run_ablation.py — Data-scarcity ablation study with extended UQ metrics.

Sweeps the number of training trajectories N_TRAIN across several values,
re-trains all four methods at each N, and reports a rich set of metrics that
go well beyond RMSE/coverage/width:

  RMSE                trajectory accuracy (mean fit only)
  coverage            empirical 95% interval coverage
  width               mean 95% interval width
  --- added: computed on the true function g* over a REGULAR 2-D GRID
      spanning the physically reachable input box (NOT at trajectory time
      points), so they measure GLOBAL function approximation quality ---
  NLL                 Gaussian negative log-likelihood over the grid (jointly
                      scores mean AND variance); lower = better
  coverage_area       fraction of the 2-D input box where g* falls inside the
                      +/- k_sigma band (the area analogue of coverage); the
                      global-approximation quantity DKL is expected to win
  ECE                 expected calibration error across all quantile levels
  CRPS                continuous ranked probability score (proper scoring rule)
  err_sigma_corr      Spearman corr between |error| and sigma over the grid
  interval_score      width + miss penalty at 95% (calibrated sharpness)
  cond_width          mean sigma among grid points the band actually COVERS

Run from your IDE (no args) or:  python run_ablation.py
Set N_TEST in config.py yourself to control how much test data is used.

Outputs to results/ablation_<case>_<obs>/:
    --- the two MAIN figures ---
    ablation_avg_nll_vs_N_<system>.png          average NLL of g* over the 2-D
                                                grid vs N (one line per method)
    ablation_constraint_satisfaction_vs_N_<system>.png  % of the 2-D grid where
                                                g* lies inside the +/- k_sigma
                                                band, vs N (one line per method)
    --- supporting figures ---
    ablation_lines_<system>.png       RMSE/coverage/width vs N (built-in)
    ablation_heatmap_<system>.png     method x N grid (built-in)
    ablation_uq_metrics_<system>.png  the extended UQ metrics vs N
    checkpoint/ablation_results.pkl   raw per-N metrics for replotting
    checkpoint/models/N<N>_rep<r>/    saved model weights for every fit, plus
                                      meta.pkl + sweep_meta.pkl, so the whole
                                      sweep can be reloaded and re-plotted from
                                      the checkpoints (see rerun_ablation_plots.py)
"""

# ════════════════════════════════════════════════════════════════════════
# ABLATION CONFIG — edit these, then just press Run.
# ════════════════════════════════════════════════════════════════════════
ABLATION_CONFIG = {
    # Case study to sweep: "cstr" or "bioreactor" (the two paper benchmarks).
    "case":     "cstr",
    # Observation mode: full-state observations, as in the paper.
    "obs":      "full",
    # N_TRAIN sweep reported in Fig. 3.
    "n_values": [1, 3, 5, 10, 20],
    # Training epochs per fit (paper: 1000).
    "epochs":   1000,
    # Independent repeats per N, different seeds.
    "n_repeats": 10,
    # Band half-width in sigmas for the coverage-area metric (paper: 3).
    "k_sigma":  3.0,
    # ── Evaluation grid ─────────────────────────────────────────────────────
    # The grid is FIXED in config.GRID_BOUNDS (Eq. 14) and is no longer
    # inferred from the trajectories. No padding is applied. Only the
    # resolution is set here.
    "grid_n":   60,
    # Persist every trained model (per N, per repeat) for later inspection.
    "save_models": True,
    # Reuse a previously saved ablation_results.pkl and only redraw the plots.
    "plot_only": False,
}
# ════════════════════════════════════════════════════════════════════════

import os
import pickle

import numpy as np
import torch

import config
from surrogates import mlp, bnn, vgp, dkl
from evaluate import evaluate_all
import plotting

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASE_INFO = {
    "lv":         ("lotka_volterra", "Lotka-Volterra"),
    "cstr":       ("cstr", "CSTR"),
    "bioreactor": ("bioreactor", "Bioreactor"),
}

ALL_MODELS = {
    "mlp": (mlp.train_trajectory, mlp.predict_trajectory, mlp.predict),
    "bnn": (bnn.train_trajectory, bnn.predict_trajectory, bnn.predict),
    "vgp": (vgp.train_trajectory, vgp.predict_trajectory, vgp.predict),
    "dkl": (dkl.train_trajectory, dkl.predict_trajectory, dkl.predict),
}
MODEL_ORDER = ["bnn", "vgp", "dkl"]      # probabilistic closures scored in Fig. 3
# The deterministic DNN is trained and checkpointed as well (it appears as a
# reference panel in the recovered-surface figure) but is excluded from the
# UQ metrics, which require a predictive variance.
TRAIN_ORDER = ["mlp"] + MODEL_ORDER

# Which of the extended metrics to plot, with "lower is better" flags.
UQ_METRICS = [
    ("nll",            "NLL",            True),
    ("crps",           "CRPS",           True),
    ("ece",            "ECE",            True),
    ("interval_score", "Interval score", True),
    ("err_sigma_corr", "Error–σ corr (Spearman)", False),
    ("cond_width",     "Coverage-cond. σ", True),
]


def _import_system(case: str):
    if case == "lv":
        from systems import lotka_volterra as sys_mod
    elif case == "cstr":
        from systems import cstr as sys_mod
    else:
        from systems import bioreactor as sys_mod
    return sys_mod


def _train_ranges(case: str):
    if case == "lv":
        return config.LV_X0_TRAIN_RANGE, config.LV_X0_TEST_RANGE
    if case == "cstr":
        return config.CSTR_X0_TRAIN, config.CSTR_X0_TEST
    return config.BIOREACTOR_X0_TRAIN, config.BIOREACTOR_X0_TEST


# Per-case physical lower bounds for the two surrogate inputs. All three
# systems have non-negative states (LV prey/predator densities, CSTR
# concentrations, bioreactor biomass/substrate), so the floor is 0 on both
# axes — padding below zero would invent a physically meaningless region.
# Kept per-case so a future system with a different physical floor is explicit.
CASE_FLOOR = {
    "lv":         (0.0, 0.0),
    "cstr":       (0.0, 0.0),
    "bioreactor": (0.0, 0.0),
}


def _build_global_grid(sys_mod, case, full_states_test, grid_n=None, pad=None):
    """Build the FIXED evaluation grid for `case` and the true function g* on it.

    The box is taken verbatim from config.GRID_BOUNDS (Eq. 14 of the
    manuscript) — it is NOT inferred from the trajectories and NO padding is
    applied. `full_states_test` and `pad` are accepted only for signature
    compatibility and are ignored.

    Returns:
        grid    : [grid_n**2, 2] float32 tensor of query points
        target  : [grid_n**2]    float32 tensor of g*(x, y)
        meta    : dict recording the box, resolution and source
    """
    (x0, x1), (y0, y1) = config.get_grid_bounds(case)
    grid_n = int(grid_n if grid_n is not None else config.GRID_N)

    gx = np.linspace(x0, x1, grid_n)
    gy = np.linspace(y0, y1, grid_n)
    XX, YY = np.meshgrid(gx, gy)
    grid = torch.tensor(np.stack([XX.ravel(), YY.ravel()], 1), dtype=torch.float32)
    g_true = sys_mod.true_residual(XX.ravel(), YY.ravel()).astype(np.float32)
    target = torch.tensor(g_true, dtype=torch.float32)
    meta = {"box": (float(x0), float(x1), float(y0), float(y1)),
            "grid_n": int(grid_n), "pad": 0.0,
            "source": "config.GRID_BOUNDS (fixed, Eq. 14)", "case": case}
    return grid, target, meta


# ════════════════════════════════════════════════════════════════════════
# Extended UQ metrics, computed on the residual g* over the test grids.
# Each model's predict_fn(model, grid) returns (mean, epistemic_var, _).
# ════════════════════════════════════════════════════════════════════════
def _gaussian_nll(err, sigma):
    """Mean Gaussian negative log-likelihood of the truth under N(mu, sigma^2)."""
    sigma = np.maximum(sigma, 1e-6)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2)
                         + 0.5 * (err / sigma) ** 2))


def _crps_gaussian(err, sigma):
    """Closed-form CRPS for a Gaussian predictive distribution
    (Gneiting & Raftery, 2007). Lower is better."""
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-6)
    z = err / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1)
                    + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def _ece(err, sigma, n_levels=10):
    """Expected calibration error from the reliability curve: for a set of
    nominal central-interval levels p, compare nominal p to the empirical
    fraction of truths inside the corresponding +/- z*sigma interval."""
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-6)
    ps = np.linspace(0.1, 0.9, n_levels)
    gaps = []
    for p in ps:
        z = norm.ppf(0.5 + p / 2.0)          # half-width multiplier for level p
        emp = np.mean(np.abs(err) <= z * sigma)
        gaps.append(abs(emp - p))
    return float(np.mean(gaps))


def _interval_score(err, sigma, alpha=0.05):
    """Mean 95% interval score: width + (2/alpha) * miss distance. Rewards
    sharp intervals but penalizes truths that fall outside. Lower is better."""
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-6)
    z = norm.ppf(1 - alpha / 2.0)            # ~1.96
    half = z * sigma
    width = 2 * half
    below = np.maximum(0.0, -err - half)     # truth below lower bound
    above = np.maximum(0.0,  err - half)     # truth above upper bound
    score = width + (2.0 / alpha) * (below + above)
    return float(np.mean(score))


def _uq_metrics_on_surface(predict_fn, model, grids, targets, k_sigma=2.0):
    """Aggregate the extended UQ metrics across all test-grid points for one
    model. Returns a dict of scalars.

    `k_sigma` is the band half-width (in standard deviations) used for the
    coverage-area and coverage-conditioned-sharpness metrics: 2 ~ 95%,
    3 ~ 99.7%. NLL/CRPS/ECE/interval-score are independent of this choice."""
    from scipy.stats import spearmanr
    errs, sigs = [], []
    with torch.no_grad():
        for g, t in zip(grids, targets):
            mu, var, _ = predict_fn(model, g)
            mu = mu.cpu().numpy().ravel()
            var = np.maximum(var.cpu().numpy().ravel(), 0.0)
            tt = t.cpu().numpy().ravel() if hasattr(t, "cpu") else np.asarray(t).ravel()
            errs.append(mu - tt)
            sigs.append(np.sqrt(var))
    err = np.concatenate(errs)
    sigma = np.concatenate(sigs)

    # Deterministic models (sigma ~ 0 everywhere): UQ metrics are undefined /
    # degenerate. Report NaN so they don't pollute the comparison.
    if np.allclose(sigma, 0.0):
        return {"nll": np.nan, "crps": np.nan, "ece": np.nan,
                "interval_score": np.nan, "err_sigma_corr": np.nan,
                "cond_width": np.nan, "coverage_area": np.nan}

    # error-sigma correlation: does sigma track |error|?
    rho, _ = spearmanr(np.abs(err), sigma)
    rho = 0.0 if np.isnan(rho) else float(rho)

    # coverage-conditioned sharpness: mean sigma among covered points at the
    # chosen band width (default +/-2 sigma; set k_sigma=3 for ~99.7%).
    covered = np.abs(err) <= k_sigma * sigma
    cond_w = float(np.mean(sigma[covered])) if covered.any() else float(np.mean(sigma))

    # coverage area: fraction of the test-grid domain where the truth falls
    # inside the +/- k_sigma band (the same quantity the band-violation plot
    # reports as "inside band"). This is the area-based analogue of coverage.
    cov_area = float(np.mean(covered))

    return {
        "nll":            _gaussian_nll(err, sigma),
        "mae":            float(np.mean(np.abs(err))),
        "rmse":           float(np.sqrt(np.mean(err ** 2))),
        "mean_sigma":     float(np.mean(sigma)),
        "crps":           _crps_gaussian(err, sigma),
        "ece":            _ece(err, sigma),
        "interval_score": _interval_score(err, sigma),
        "err_sigma_corr": rho,
        "cond_width":     cond_w,
        "coverage_area":  cov_area,
    }


def _fit_and_eval_at_N(sys_mod, case, obs, n_train, epochs, seed_offset=0,
                       k_sigma=None, grid_n=None, grid_pad=0.0):
    """Train all methods at a given N_TRAIN and return (metrics, uq, bundle).

    `metrics` is the standard trajectory-level evaluate_all dict (RMSE /
    coverage / width — unchanged). `uq[name]` is the extended metric dict for
    that method, now computed on a REGULAR 2-D grid over the physically
    reachable input box (the GLOBAL function-approximation surface) rather than
    at trajectory time points. `bundle` carries the trained models and the data
    needed to re-load and re-plot them later.

    `seed_offset` makes each repeat distinct: it perturbs both the model
    initialization (torch.manual_seed) and the training-data draw, so repeats
    measure genuine run-to-run variability rather than re-running the identical
    fit. The TEST set is held FIXED across repeats (same seed) so every repeat
    is scored on the same data — and hence on the same global grid."""
    import random
    if k_sigma is None:
        k_sigma = config.K_SIGMA
    if grid_n is None:
        grid_n = config.GRID_N
    config.OBS_MODE = obs
    config.N_TRAIN = n_train
    config.N_EPOCHS = epochs

    # Seed model init + any training stochasticity for this repeat.
    seed = config.SEED + 1000 * seed_offset
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_range, test_range = _train_ranges(case)
    # Training data varies per repeat (different draw); test data is fixed.
    x0_train, y_noisy_train, _, full_train = \
        sys_mod.generate_data(n_train, train_range, config.SEED + seed_offset)
    x0_test, _, _, full_test = \
        sys_mod.generate_data(config.N_TEST, test_range, config.SEED + 1000)
    x0_batch, y_obs, t_grid, obs_indices = \
        sys_mod.build_traj_tensors(x0_train, y_noisy_train)
    x0_test_t = torch.tensor(np.stack(x0_test, 0), dtype=torch.float32)
    noise_var_diag = sys_mod.get_noise_var_diag()

    trained, traj_fns, pt_fns, loss_hist, fit_secs = {}, {}, {}, {}, {}
    for name in TRAIN_ORDER:
        train_fn, traj_pred_fn, pt_pred_fn = ALL_MODELS[name]
        print(f"      training {plotting.MD.get(name, name)}...")
        # Re-seed before every fit so each model's initialization depends only
        # on the repeat, not on how many models were trained before it. Without
        # this, a model's starting weights depend on its position in
        # TRAIN_ORDER, and adding or removing a baseline silently changes the
        # others. It also lets the sensitivity studies, which train DKL on its
        # own, reproduce the DKL fit obtained here.
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model, secs, losses = train_fn(
            sys_mod, x0_batch, y_obs, t_grid, obs_indices,
            noise_var_diag, full_states_train=full_train, epochs=epochs)
        loss_hist[name] = list(losses) if losses is not None else []
        fit_secs[name] = float(secs)
        trained[name] = model
        traj_fns[name] = traj_pred_fn
        pt_fns[name] = pt_pred_fn

    # Standard trajectory-level metrics still use the trajectory-aligned grids.
    surf_grids, surf_targets = sys_mod.build_test_grids(full_test)
    metrics = evaluate_all(trained, traj_fns, pt_fns, sys_mod, x0_test_t,
                           t_grid, full_test, noise_var_diag,
                           surf_grids=surf_grids, surf_targets=surf_targets)

    # ── Extended UQ metrics on the GLOBAL 2-D grid ──────────────────────────
    # NLL and coverage-area now score how well each method recovers g* across
    # the whole physically reachable input box, not just where trajectories
    # happened to pass. _uq_metrics_on_surface aggregates over a LIST of grids,
    # so we hand it a single-element list holding the global grid.
    global_grid, global_target, grid_meta = _build_global_grid(
        sys_mod, case, full_test, grid_n=grid_n, pad=grid_pad)
    uq = {}
    for name in MODEL_ORDER:
        uq[name] = _uq_metrics_on_surface(pt_fns[name], trained[name],
                                          [global_grid], [global_target],
                                          k_sigma=k_sigma)

    bundle = {"trained": trained, "full_train": full_train,
              "full_test": full_test, "grid_meta": grid_meta,
              "seed": seed, "seed_offset": seed_offset,
              "loss_hist": loss_hist, "fit_secs": fit_secs}
    return metrics, uq, bundle


def _save_repeat_models(models_root, N, rep, bundle, cfg, case, obs, sys_label):
    """Persist every trained model for one (N, repeat) fit so the sweep can be
    reloaded and dissected later. Layout (mirrors run_benchmark's checkpoint
    naming so the existing loaders work unchanged):

        models_root/N<N>_rep<rep>/
            mlp.pt
            bnn.pt
            vgp_part0.pt  vgp_part1.pt
            dkl_part0.pt  dkl_part1.pt  dkl_part2.pt
            meta.pkl                      # full_train (-> standardizer), seed, sizes

    A single sweep-level `sweep_meta.pkl` (written once at models_root) holds
    the FIXED test trajectories and the global-grid spec, so the re-run script
    can rebuild the identical grid and recompute the metrics from the weights.
    """
    rep_dir = os.path.join(models_root, f"N{N}_rep{rep}")
    os.makedirs(rep_dir, exist_ok=True)
    for name, model in bundle["trained"].items():
        if isinstance(model, tuple):
            for i, m in enumerate(model):
                if hasattr(m, "state_dict"):
                    torch.save(m.state_dict(),
                               os.path.join(rep_dir, f"{name}_part{i}.pt"))
        elif hasattr(model, "state_dict"):
            torch.save(model.state_dict(), os.path.join(rep_dir, f"{name}.pt"))
    # Loss histories for every model in this fit, for the convergence figure.
    with open(os.path.join(rep_dir, "loss_history.pkl"), "wb") as f:
        pickle.dump({"loss_hist": bundle.get("loss_hist", {}),
                     "fit_secs": bundle.get("fit_secs", {}),
                     "n_train": int(N), "repeat": int(rep),
                     "case": case, "obs": obs}, f)
    # full_train is all the re-run script needs to rebuild the standardizer
    # (compute_standardizer(full_train, sys_mod)) exactly as training did.
    with open(os.path.join(rep_dir, "meta.pkl"), "wb") as f:
        pickle.dump({
            "full_train": bundle["full_train"],
            "seed": bundle["seed"], "n_train": int(N), "repeat": int(rep),
            "case": case, "obs": obs,
            # model sizes, captured so reconstruction does not depend on the
            # current config.py if it later changes.
            "input_dim": _import_system(case).INPUT_DIM,
            "hidden_dim": config.HIDDEN_DIM,
            "bnn_hidden_dim": config.BNN_HIDDEN_DIM,
            "dkl_feature_dim": config.DKL_FEATURE_DIM,
            "n_inducing": config.N_INDUCING,
            "output_dim": _import_system(case).OUTPUT_DIM,
        }, f)


def _save_sweep_meta(models_root, bundle, cfg, case, obs, sys_label, n_values):
    """Write the one-time sweep-level metadata (fixed test set + grid spec)."""
    with open(os.path.join(models_root, "sweep_meta.pkl"), "wb") as f:
        pickle.dump({
            "full_test": bundle["full_test"],       # FIXED across all fits
            "grid_meta": bundle["grid_meta"],        # box / grid_n / pad / floor
            "grid_n": int(cfg.get("grid_n", 60)),
            "grid_pad": 0.0,
            "k_sigma": float(cfg.get("k_sigma", config.K_SIGMA)),
            "n_values": list(n_values),
            "n_repeats": int(cfg.get("n_repeats", 1)),
            "case": case, "obs": obs, "system": sys_label,
            "model_order": MODEL_ORDER,
        }, f)



def _plot_uq_metrics(all_uq, n_values, out_dir, sys_label):
    """Grid of the extended UQ metrics vs N_train, one panel per metric,
    one curve per method."""
    ncols = 3
    nrows = int(np.ceil(len(UQ_METRICS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.4 * nrows),
                             squeeze=False)
    for idx, (key, label, lower_better) in enumerate(UQ_METRICS):
        ax = axes[idx // ncols][idx % ncols]
        for name in MODEL_ORDER:
            ys = [all_uq[N][name].get(key, np.nan) for N in n_values]
            if np.all(np.isnan(ys)):
                continue   # e.g. deterministic MLP has no UQ metrics
            ax.plot(n_values, ys, "-o", color=plotting.MC.get(name, "#888"),
                    lw=2.2, markersize=6, label=plotting.MD.get(name, name))
        arrow = "↓ lower better" if lower_better else "↑ higher better"
        ax.set_title(f"{label}  ({arrow})")
        ax.set_xlabel("# training trajectories $N$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=9, framealpha=0.95)
    for j in range(len(UQ_METRICS), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Extended uncertainty-quantification metrics vs. data size — "
                 f"{sys_label}", fontsize=15, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, f"ablation_uq_metrics_{sys_label}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def _plot_metric_vs_N_repeats(all_uq_rep, n_values, metric_key, ylabel,
                              title, out_dir, sys_label, fname,
                              lower_better=True, log_if_wide=True,
                              scale=1.0, percent=False):
    """One clean figure: <metric> vs N_train, one line per method, with the
    mean over repeats as the line and +/- 1 std as error bars / shaded band.
    Deterministic methods (all-NaN for this metric) are skipped.

    `scale` multiplies every value before plotting (e.g. 100 to turn a 0–1
    fraction into a percentage). `percent=True` is a convenience that sets
    scale=100, forces a linear 0–100 axis, and never uses log scaling — used
    for the constraint-satisfaction figure.

    If `log_if_wide` (and not `percent`) and the across-method value range
    spans more than ~30x, the y-axis is set to log scale so one badly-scoring
    method (e.g. BNN's large NLL) does not crush the others onto the floor."""
    if percent:
        scale = 100.0
        log_if_wide = False
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    any_plotted = False
    all_means = []
    for name in MODEL_ORDER:
        means, stds = [], []
        for N in n_values:
            vals = np.array(all_uq_rep[N][name].get(metric_key, []), dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                means.append(np.nan); stds.append(np.nan)
            else:
                means.append(float(vals.mean()))
                stds.append(float(vals.std()))
        means = np.array(means) * scale
        stds = np.array(stds) * scale
        if np.all(np.isnan(means)):
            continue
        any_plotted = True
        all_means.append(means[~np.isnan(means)])
        col = plotting.MC.get(name, "#888")
        ax.plot(n_values, means, "-o", color=col, lw=2.4, markersize=7,
                label=plotting.MD.get(name, name), zorder=3)
        ax.fill_between(n_values, means - stds, means + stds, color=col,
                        alpha=0.18, zorder=1)
        ax.errorbar(n_values, means, yerr=stds, fmt="none", ecolor=col,
                    elinewidth=1.4, capsize=4, alpha=0.8, zorder=2)
    if not any_plotted:
        plt.close(fig); return

    # Decide on log scale: only if all plotted means are positive and the
    # spread across methods is large enough that a linear axis would squash
    # the smaller curves.
    flat = np.concatenate(all_means)
    flat_pos = flat[flat > 0]
    use_log = (log_if_wide and flat_pos.size > 0
               and flat_pos.min() > 0
               and (flat_pos.max() / flat_pos.min() > 30)
               and (flat <= 0).sum() == 0)
    if use_log:
        ax.set_yscale("log")
    if percent:
        ax.set_ylim(-3, 103)
        ax.axhline(100, color="0.6", ls=":", lw=1.0, zorder=0)

    arrow = "↓ lower is better" if lower_better else "↑ higher is better"
    scale_note = "  [log scale]" if use_log else ""
    ax.set_xlabel("Number of training trajectories  $N$", fontsize=12)
    ax.set_ylabel(ylabel + scale_note, fontsize=12)
    ax.set_title(f"{title}\n{sys_label}  ({arrow}; mean ± 1 s.d. over repeats)",
                 fontsize=13)
    ax.set_xticks(n_values)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=11, framealpha=0.95)
    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def main():
    cfg = ABLATION_CONFIG
    case, obs = cfg["case"], cfg["obs"]
    n_values = sorted(cfg["n_values"])
    sys_mod = _import_system(case)
    cname, sys_label = CASE_INFO[case]

    out_dir = f"results/ablation_{cname}_{obs}"
    ckpt = os.path.join(out_dir, "checkpoint")
    os.makedirs(ckpt, exist_ok=True)
    pkl_path = os.path.join(ckpt, "ablation_results.pkl")
    models_root = os.path.join(ckpt, "models")

    if cfg["plot_only"]:
        print(f"Plot-only: loading {pkl_path}")
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
        all_results = {int(k): v for k, v in saved["all_results"].items()}
        all_uq = {int(k): v for k, v in saved.get("all_uq", {}).items()}
        all_uq_rep = {int(k): v for k, v in saved.get("all_uq_rep", {}).items()}
        # Use the k_sigma the metrics were actually computed with, so the
        # coverage-area plot label matches the stored data (not whatever the
        # config currently says).
        if "k_sigma" in saved:
            cfg["k_sigma"] = saved["k_sigma"]
        loss_curves = {int(k): v
                       for k, v in saved.get("loss_curves", {}).items()}
        n_values = sorted(all_results.keys())
    else:
        n_rep = int(cfg.get("n_repeats", 1))
        grid_n = int(cfg.get("grid_n", config.GRID_N))
        grid_pad = 0.0  # fixed grid; padding disabled
        save_models = bool(cfg.get("save_models", True))
        if save_models:
            os.makedirs(models_root, exist_ok=True)
        print(f"\n{'='*70}\nABLATION: {sys_label} ({obs}) | "
              f"N_TRAIN sweep = {n_values} | repeats={n_rep} | "
              f"epochs={cfg['epochs']} | N_TEST={config.N_TEST}\n"
              f"  Surface UQ metrics (NLL, coverage-area) computed on a "
              f"{grid_n}x{grid_n} global 2-D grid; save_models={save_models}\n"
              f"{'='*70}")
        all_results, all_uq = {}, {}
        # Per-repeat store: all_uq_rep[N][name][metric] = [value per repeat]
        all_uq_rep = {N: {nm: {} for nm in MODEL_ORDER} for N in n_values}
        # loss_curves[N][name] = list of per-repeat loss histories
        loss_curves = {N: {nm: [] for nm in TRAIN_ORDER} for N in n_values}
        sweep_meta_written = False
        for N in n_values:
            print(f"\n  --- N_TRAIN = {N} ---")
            for rep in range(n_rep):
                print(f"    [repeat {rep+1}/{n_rep}]")
                metrics, uq, bundle = _fit_and_eval_at_N(
                    sys_mod, case, obs, N, cfg["epochs"], seed_offset=rep,
                    k_sigma=float(cfg.get("k_sigma", config.K_SIGMA)),
                    grid_n=grid_n, grid_pad=grid_pad)
                # keep the LAST repeat for the standard tables/built-in plots
                all_results[N] = metrics
                all_uq[N] = uq
                # accumulate every metric across repeats
                for nm in MODEL_ORDER:
                    for k, v in uq[nm].items():
                        all_uq_rep[N][nm].setdefault(k, []).append(v)
                for nm, curve in bundle.get("loss_hist", {}).items():
                    loss_curves[N].setdefault(nm, []).append(curve)
                # persist the trained models for this fit
                if save_models:
                    _save_repeat_models(models_root, N, rep, bundle, cfg,
                                        case, obs, sys_label)
                    if not sweep_meta_written:
                        _save_sweep_meta(models_root, bundle, cfg, case, obs,
                                         sys_label, n_values)
                        sweep_meta_written = True
        with open(pkl_path, "wb") as f:
            pickle.dump({"all_results": all_results, "all_uq": all_uq,
                         "all_uq_rep": all_uq_rep,
                         "loss_curves": loss_curves,
                         "n_values": n_values, "n_repeats": n_rep,
                         "k_sigma": float(cfg.get("k_sigma", config.K_SIGMA)),
                         "grid_n": grid_n, "grid_pad": grid_pad,
                         "metrics_grid": "global_2d",
                         "case": case, "obs": obs, "system": sys_label}, f)
        print(f"\n  Raw ablation metrics saved to {pkl_path}")
        if save_models:
            print(f"  Trained models for every (N, repeat) saved under "
                  f"{models_root}/")

    # ── Summary table to console (standard metrics) ──
    print(f"\n  {'N':<5}{'method':<10}{'RMSE':<12}{'coverage':<12}{'width':<10}")
    for N in n_values:
        for n in MODEL_ORDER:
            r = np.mean(all_results[N]["rmse"][n])
            c = np.mean(all_results[N]["coverage"][n])
            w = np.mean(all_results[N]["width"][n])
            print(f"  {N:<5}{plotting.MD.get(n, n):<10}{r:<12.4f}{c:<12.3f}{w:<10.4f}")

    # ── Summary table to console (extended UQ metrics) ──
    if all_uq:
        print(f"\n  Extended UQ metrics:")
        print(f"  {'N':<5}{'method':<10}{'NLL':<10}{'CRPS':<10}{'ECE':<9}"
              f"{'IntScore':<11}{'err-σ ρ':<10}{'condσ':<9}")
        for N in n_values:
            for n in MODEL_ORDER:
                u = all_uq[N][n]
                def f(x): return "  nan  " if (x is None or np.isnan(x)) else f"{x:<9.4f}"
                print(f"  {N:<5}{plotting.MD.get(n, n):<10}"
                      f"{f(u['nll'])}{f(u['crps'])}{f(u['ece'])[:8]:<9}"
                      f"{f(u['interval_score'])[:10]:<11}"
                      f"{f(u['err_sigma_corr'])}{f(u['cond_width'])[:8]:<9}")

    # ── Plots ──
    print(f"\n  Generating ablation plots -> {out_dir}/")
    plotting.plot_ablation_lines(all_results, n_values, out_dir, sys_label)
    plotting.plot_ablation_heatmap(all_results, n_values, out_dir, sys_label)
    if all_uq:
        _plot_uq_metrics(all_uq, n_values, out_dir, sys_label)
    # ── The two MAIN ablation figures ──────────────────────────────────────
    #   (1) average NLL of g* over the 2-D grid vs N
    #   (2) % constraint satisfaction over the 2-D grid vs N
    if all_uq_rep:
        ks = float(cfg.get('k_sigma', 2.0))
        print("  [main figures]")
        _plot_metric_vs_N_repeats(
            all_uq_rep, n_values, "nll",
            "Average NLL over 2-D grid",
            "Average NLL of $g^*$ over the 2-D grid vs. number of training "
            "trajectories",
            out_dir, sys_label, f"ablation_avg_nll_vs_N_{sys_label}.png",
            lower_better=True)
        _plot_metric_vs_N_repeats(
            all_uq_rep, n_values, "coverage_area",
            f"Constraint satisfaction (% of 2-D grid)",
            f"Constraint satisfaction over the 2-D grid "
            f"($g^*$ inside $\\pm{ks:g}\\sigma$ band) vs. number of training "
            f"trajectories",
            out_dir, sys_label,
            f"ablation_constraint_satisfaction_vs_N_{sys_label}.png",
            lower_better=False, percent=True)
    if loss_curves:
        print("  Loss curves saved; run plot_loss_curves.py to draw them.")
    print(f"\n{'='*70}\nABLATION COMPLETE — figures in {out_dir}/\n{'='*70}")


if __name__ == "__main__":
    main()