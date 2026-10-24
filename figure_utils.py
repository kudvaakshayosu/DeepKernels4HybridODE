# -*- coding: utf-8 -*-
"""
figure_utils.py — shared helpers for the paper figure scripts.

Loads trained surrogates back from the ablation checkpoints written by
`run_ablation.py` and rebuilds the FIXED evaluation grid defined in
`config.GRID_BOUNDS` (Eq. 14). Every figure script imports from here, so all
figures are produced from the same models, on the same grid, with the same
band half-width — by construction rather than by coincidence.
"""
import os
import pickle

import numpy as np
import torch

import config
from surrogates.traj_train import compute_standardizer


# Display names, in the order used by the paper figures.
METHOD_LABEL = {
    "mlp": "DNN",
    "bnn": "BNN",
    "vgp": "VGP",
    "dkl": "DKL",
}
PROB_METHODS = ["bnn", "vgp", "dkl"]     # methods with a predictive variance
ALL_METHODS = ["mlp"] + PROB_METHODS


# ── system import ────────────────────────────────────────────────────────────
def import_system(case):
    if case == "lv":
        from systems import lotka_volterra as sys_mod
    elif case == "cstr":
        from systems import cstr as sys_mod
    elif case == "bioreactor":
        from systems import bioreactor as sys_mod
    else:
        raise ValueError(f"Unknown case '{case}'")
    return sys_mod


def case_axis_labels(case):
    """(xlabel, ylabel) for the two surrogate inputs."""
    if case == "cstr":
        return r"$C_A$ (mol/L)", r"$C_B$ (mol/L)"
    if case == "bioreactor":
        return r"$X$ (g/L)", r"$S$ (g/L)"
    return r"$x_1$", r"$x_2$"


def case_closure_label(case):
    if case == "cstr":
        return r"$r(C_A, C_B)$"
    if case == "bioreactor":
        return r"$g(X, S)$"
    return r"$g(x_1, x_2)$"


def system_label(case):
    return {"cstr": "CSTR", "bioreactor": "Bioreactor",
            "lv": "Lotka-Volterra"}[case]


# ── fixed evaluation grid (Eq. 14) ───────────────────────────────────────────
def build_grid(case, sys_mod, grid_n=None):
    """Rebuild the FIXED evaluation grid and the true closure on it.

    Identical to `run_ablation._build_global_grid`: bounds come verbatim from
    config.GRID_BOUNDS with no padding.

    Returns (grid[T,2] tensor, g_true[n,n] array, gx, gy, box).
    """
    (x0, x1), (y0, y1) = config.get_grid_bounds(case)
    grid_n = int(grid_n if grid_n is not None else config.GRID_N)
    gx = np.linspace(x0, x1, grid_n)
    gy = np.linspace(y0, y1, grid_n)
    XX, YY = np.meshgrid(gx, gy)
    grid = torch.tensor(np.stack([XX.ravel(), YY.ravel()], 1),
                        dtype=torch.float32)
    g_true = sys_mod.true_residual(XX.ravel(), YY.ravel()).reshape(XX.shape)
    return grid, g_true, gx, gy, (x0, x1, y0, y1)


# ── checkpoint loaders ───────────────────────────────────────────────────────
def _load_mlp(rep_dir, std, meta):
    from surrogates.mlp import MLPNet
    from surrogates import mlp as mlp_mod
    path = os.path.join(rep_dir, "mlp.pt")
    if not os.path.exists(path):
        return None
    net = MLPNet(meta["input_dim"], meta["hidden_dim"],
                 output_dim=meta["output_dim"], standardizer=std)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return lambda grid: mlp_mod.predict(net, grid)


def _load_bnn(rep_dir, std, meta):
    from surrogates.bnn import BayesianNet
    from surrogates import bnn as bnn_mod
    path = os.path.join(rep_dir, "bnn.pt")
    if not os.path.exists(path):
        return None
    net = BayesianNet(meta["input_dim"], meta["bnn_hidden_dim"],
                      output_dim=meta["output_dim"], standardizer=std)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return lambda grid: bnn_mod.predict(net, grid)


def _load_vgp(rep_dir, std, meta):
    from surrogates.vgp import SVGP, GPWrapper
    from surrogates import vgp as vgp_mod
    p0 = os.path.join(rep_dir, "vgp_part0.pt")
    if not os.path.exists(p0):
        return None
    svgp = SVGP(torch.zeros(meta["n_inducing"], meta["input_dim"]),
                meta["input_dim"])
    svgp.load_state_dict(torch.load(p0, map_location="cpu"))
    svgp.eval()
    wrapper = GPWrapper(svgp, standardizer=std)
    return lambda grid: vgp_mod.predict((svgp, wrapper), grid)


def _load_dkl(rep_dir, std, meta):
    from surrogates.dkl import DKLFeat, FeatureGP, DKLWrapper
    from surrogates import dkl as dkl_mod
    p0 = os.path.join(rep_dir, "dkl_part0.pt")
    if not os.path.exists(p0):
        return None
    feat = DKLFeat(meta["input_dim"], meta["hidden_dim"],
                   meta["dkl_feature_dim"], standardizer=std)
    svgp = FeatureGP(torch.zeros(meta["n_inducing"], meta["dkl_feature_dim"]),
                     meta["dkl_feature_dim"])
    svgp.load_state_dict(torch.load(p0, map_location="cpu"))
    feat.load_state_dict(torch.load(os.path.join(rep_dir, "dkl_part1.pt"),
                                    map_location="cpu"))
    feat.eval()
    svgp.eval()
    wrapper = DKLWrapper(feat, svgp, standardizer=std)
    return lambda grid: dkl_mod.predict((svgp, feat, wrapper), grid)


_LOADERS = {"mlp": _load_mlp, "bnn": _load_bnn,
            "vgp": _load_vgp, "dkl": _load_dkl}


def models_root(case, obs="full", results_dir="results"):
    sys_name = {"cstr": "cstr", "bioreactor": "bioreactor",
                "lv": "lotka_volterra"}[case]
    return os.path.join(results_dir, f"ablation_{sys_name}_{obs}",
                        "checkpoint", "models")


def load_predictors(case, n_train, rep, methods, obs="full",
                    results_dir="results", n_mc=None):
    """Load the trained surrogates for one (N_TRAIN, repeat) fit.

    `n_mc` overrides the number of Monte Carlo weight samples the BNN uses at
    prediction time (the GP-based methods return closed-form moments).

    Returns (predictors dict name -> predict(grid), meta dict).
    """
    root = models_root(case, obs, results_dir)
    rep_dir = os.path.join(root, f"N{n_train}_rep{rep}")
    if not os.path.isdir(rep_dir):
        raise FileNotFoundError(
            f"No checkpoint at {rep_dir}.\n"
            f"Run run_ablation.py for case='{case}' with save_models=True "
            f"and N_TRAIN={n_train} in n_values first.")
    with open(os.path.join(rep_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    sys_mod = import_system(case)
    std = compute_standardizer(meta["full_train"], sys_mod)

    if n_mc is not None:
        from surrogates import bnn as bnn_mod
        bnn_mod.MC_SAMPLES = int(n_mc)

    predictors = {}
    for m in methods:
        fn = _LOADERS[m](rep_dir, std, meta)
        if fn is None:
            print(f"    (no checkpoint for '{m}' in {rep_dir}, skipping)")
        else:
            predictors[m] = fn
    if not predictors:
        raise FileNotFoundError(f"No usable checkpoints found in {rep_dir}.")
    return predictors, meta


def available_repeats(case, n_train, obs="full", results_dir="results"):
    """List the repeat indices checkpointed for this (case, N_TRAIN)."""
    root = models_root(case, obs, results_dir)
    if not os.path.isdir(root):
        return []
    reps = []
    for d in os.listdir(root):
        if d.startswith(f"N{n_train}_rep"):
            try:
                reps.append(int(d.split("rep")[-1]))
            except ValueError:
                pass
    return sorted(reps)


def predict_on_grid(fn, grid, shape):
    """Evaluate a loaded predictor -> (mu, sigma) reshaped to `shape`."""
    with torch.no_grad():
        mu, var, _ = fn(grid)
    mu = mu.cpu().numpy().ravel().reshape(shape)
    sigma = np.sqrt(np.maximum(var.cpu().numpy().ravel(), 0.0)).reshape(shape)
    return mu, sigma


def training_footprint(meta, sys_mod):
    """(x, y) state-space locations visited by the training trajectories,
    at the sparse measurement times only — the grey/maroon points overlaid on
    the paper figures."""
    full_train = meta["full_train"]
    ox = np.concatenate([s[0, sys_mod.OBS_INDICES] for s in full_train])
    oy = np.concatenate([s[1, sys_mod.OBS_INDICES] for s in full_train])
    return ox, oy


def ensure_outdir(case, obs="full", results_dir="results"):
    sys_name = {"cstr": "cstr", "bioreactor": "bioreactor",
                "lv": "lotka_volterra"}[case]
    out = os.path.join(results_dir, f"ablation_{sys_name}_{obs}", "figures")
    os.makedirs(out, exist_ok=True)
    return out
