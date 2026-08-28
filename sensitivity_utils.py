# -*- coding: utf-8 -*-
"""
sensitivity_utils.py — shared harness for the sensitivity studies.

Provides a single `fit_dkl_once(...)` entry point that trains one DKL closure
under a specified configuration and scores it on the FIXED evaluation grid
(config.GRID_BOUNDS, Eq. 14), returning both the grid NLL and the mean absolute
error of the predictive mean against the withheld generating kinetics.

Every sensitivity script uses this same function, so results across the studies
are directly comparable to one another and to the main ablation.
"""
import os
import time

import numpy as np
import torch

import config
from surrogates import dkl
from surrogates.traj_train import compute_standardizer


def import_system(case):
    if case == "cstr":
        from systems import cstr as m
    elif case == "bioreactor":
        from systems import bioreactor as m
    elif case == "lv":
        from systems import lotka_volterra as m
    else:
        raise ValueError(f"Unknown case '{case}'")
    return m


def system_label(case):
    return {"cstr": "CSTR", "bioreactor": "Fed-batch bioreactor",
            "lv": "Lotka-Volterra"}[case]


def build_grid(case, sys_mod, grid_n=None):
    """FIXED evaluation grid (Eq. 14) and the true closure on it."""
    (x0, x1), (y0, y1) = config.get_grid_bounds(case)
    grid_n = int(grid_n if grid_n is not None else config.GRID_N)
    gx = np.linspace(x0, x1, grid_n)
    gy = np.linspace(y0, y1, grid_n)
    XX, YY = np.meshgrid(gx, gy)
    grid = torch.tensor(np.stack([XX.ravel(), YY.ravel()], 1),
                        dtype=torch.float32)
    g_true = sys_mod.true_residual(XX.ravel(), YY.ravel()).astype(np.float32)
    return grid, torch.tensor(g_true), gx, gy


def set_obs_per_traj(sys_mod, case, k_obs, seed=None):
    """Re-draw the measurement-time indices so each trajectory carries `k_obs`
    observations, and rebind them on the system module.

    OBS_INDICES is a module-level constant fixed at import from
    config.N_OBS_PER_TRAJ, so it must be rebound (not just re-configured) for a
    K sweep. Returns the previous value so the caller can restore it.
    """
    n_int = {"cstr": config.CSTR_N_INTEGRATION,
             "bioreactor": config.BIOREACTOR_N_INTEGRATION,
             "lv": config.LV_N_INTEGRATION}[case]
    prev = sys_mod.OBS_INDICES
    seed = config.SEED if seed is None else seed
    sys_mod.OBS_INDICES = config.get_obs_indices(n_int, int(k_obs), seed=seed)
    return prev


def grid_metrics(model, grid, g_true):
    """Score a trained DKL closure on the fixed grid.

    Returns dict with:
        nll  — average Gaussian negative log-likelihood of the withheld
               kinetics under the predictive distribution (Eq. 15)
        mae  — mean absolute error of the predictive MEAN vs the withheld
               kinetics
        rmse — root mean squared error of the predictive mean
        mean_sigma — average predictive standard deviation over the grid
    """
    with torch.no_grad():
        mu, var, _ = dkl.predict(model, grid)
    mu = mu.cpu().numpy().ravel()
    sigma = np.sqrt(np.maximum(var.cpu().numpy().ravel(), 1e-12))
    g = g_true.cpu().numpy().ravel()
    err = g - mu
    nll = np.mean(0.5 * np.log(2.0 * np.pi * sigma ** 2)
                  + err ** 2 / (2.0 * sigma ** 2))
    return {"nll": float(nll),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mean_sigma": float(np.mean(sigma))}


def save_dkl_model(save_dir, tag, model, full_train, meta_extra=None):
    """Checkpoint one trained DKL closure, mirroring the layout run_ablation
    uses so the same loaders in figure_utils can read it back.

        save_dir/<tag>/dkl_part0.pt   sparse variational GP
                      /dkl_part1.pt   feature extractor
                      /meta.pkl       full_train (-> standardizer) + sizes
    """
    import pickle as _pkl
    d = os.path.join(save_dir, tag)
    os.makedirs(d, exist_ok=True)
    svgp, feat, wrapper = model
    torch.save(svgp.state_dict(), os.path.join(d, "dkl_part0.pt"))
    torch.save(feat.state_dict(), os.path.join(d, "dkl_part1.pt"))
    meta = {"full_train": full_train,
            "input_dim": feat.net[0].in_features
            if hasattr(feat, "net") else None,
            "hidden_dim": config.HIDDEN_DIM,
            "bnn_hidden_dim": config.BNN_HIDDEN_DIM,
            "n_inducing": config.N_INDUCING,
            "output_dim": 1}
    if meta_extra:
        meta.update(meta_extra)
    with open(os.path.join(d, "meta.pkl"), "wb") as f:
        _pkl.dump(meta, f)
    return d


def fit_dkl_once(case, n_train, epochs, seed_offset=0,
                 feat_dim=None, n_mc=None, k_obs=None, grid_n=None,
                 verbose=False, save_dir=None, tag=None):
    """Train ONE DKL closure under the given configuration and score it.

    Parameters
    ----------
    n_train   : number of training trajectories, N
    feat_dim  : DKL latent feature dimension (None -> config.DKL_FEATURE_DIM)
    n_mc      : Monte Carlo rollouts per gradient step, M_S
                (None -> config.TRAJ_TRAIN_MC)
    k_obs     : measurement times per trajectory, K
                (None -> config.N_OBS_PER_TRAJ)
    seed_offset : distinguishes repeats; perturbs the data draw and the model
                initialization exactly as run_ablation does.

    Returns (metrics dict, wall-clock training seconds).
    """
    import random

    feat_dim = int(feat_dim if feat_dim is not None else config.DKL_FEATURE_DIM)
    n_mc = int(n_mc if n_mc is not None else config.TRAJ_TRAIN_MC)

    sys_mod = import_system(case)
    prev_obs = None
    if k_obs is not None:
        prev_obs = set_obs_per_traj(sys_mod, case, k_obs)

    try:
        seed = config.SEED + 1000 * seed_offset
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        x0_batch, y_obs, t_grid, obs_indices, noise_var_diag, full_train = \
            _make_training_data(sys_mod, case, n_train, seed_offset)

        t0 = time.perf_counter()
        model, _, losses = dkl.train_trajectory(
            sys_mod, x0_batch, y_obs, t_grid, obs_indices, noise_var_diag,
            full_states_train=full_train, epochs=epochs,
            feat_dim=feat_dim, n_mc=n_mc)
        train_s = time.perf_counter() - t0

        if save_dir is not None:
            save_dkl_model(save_dir, tag or f"N{n_train}_rep{seed_offset}",
                           model, full_train,
                           meta_extra={"input_dim": sys_mod.INPUT_DIM,
                                       "output_dim": sys_mod.OUTPUT_DIM,
                                       "dkl_feature_dim": feat_dim,
                                       "n_mc": n_mc, "n_train": int(n_train),
                                       "case": case, "obs": "full",
                                       "seed": seed, "repeat": seed_offset})

        grid, g_true, _, _ = build_grid(case, sys_mod, grid_n)
        m = grid_metrics(model, grid, g_true)
        m.update({"loss_curve": list(losses) if losses is not None else [],
                  "final_loss": float(losses[-1]) if losses else float("nan"),
                  "feat_dim": feat_dim, "n_mc": n_mc,
                  "k_obs": int(k_obs) if k_obs is not None
                  else int(config.N_OBS_PER_TRAJ),
                  "n_train": int(n_train), "train_s": train_s})
        if verbose:
            print(f"      NLL={m['nll']:+.4f}  MAE={m['mae']:.4f}  "
                  f"({train_s:.1f}s)")
        return m, train_s
    finally:
        if prev_obs is not None:
            sys_mod.OBS_INDICES = prev_obs


def _train_ranges(case):
    if case == "cstr":
        return config.CSTR_X0_TRAIN
    if case == "bioreactor":
        return config.BIOREACTOR_X0_TRAIN
    return config.LV_X0_TRAIN_RANGE


def _make_training_data(sys_mod, case, n_train, seed_offset):
    """Draw one training set, mirroring run_ablation's data generation
    exactly (same generator, same seeding convention)."""
    train_range = _train_ranges(case)
    x0_train, y_noisy_train, _, full_train = sys_mod.generate_data(
        int(n_train), train_range, config.SEED + seed_offset)
    x0_batch, y_obs, t_grid, obs_indices = \
        sys_mod.build_traj_tensors(x0_train, y_noisy_train)
    noise_var_diag = sys_mod.get_noise_var_diag()
    return (x0_batch, y_obs, t_grid, obs_indices, noise_var_diag, full_train)


def summarize(rows, keys, metric):
    """Group `rows` by the tuple of `keys` and return {key_tuple: (mean, std)}."""
    from collections import defaultdict
    acc = defaultdict(list)
    for r in rows:
        acc[tuple(r[k] for k in keys)].append(r[metric])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}
