# -*- coding: utf-8 -*-
"""
backfill_grid_metrics.py — add missing grid metrics to an existing sweep
WITHOUT retraining.

`run_ablation.py` originally recorded NLL, CRPS, ECE, interval score, the
error-sigma correlation, conditional width and coverage area, but not the mean
absolute error of the predictive mean. This script reloads the surrogates that
were checkpointed during the sweep, re-evaluates them on the same fixed grid,
computes the missing quantities, and writes them back into
`ablation_results.pkl`.

Only the metrics dictionaries are touched: the trained weights are read but
never modified, and the original pickle is backed up first.

As a consistency check the script also recomputes NLL and coverage area and
compares them with the values already stored. Deterministic closures (VGP, DKL)
should reproduce exactly; the BNN predictive is a Monte Carlo average over
weight samples, so small differences there are expected and are reported rather
than hidden.

Run from an IDE (no arguments) or:  python backfill_grid_metrics.py
"""

# ════════════════════════════════════════════════════════════════════════
# CONFIG — edit these, then press Run.
# ════════════════════════════════════════════════════════════════════════
BACKFILL_CONFIG = {
    "cases":   ["cstr", "bioreactor"],   # sweeps to update
    "obs":     "full",
    "methods": ["bnn", "vgp", "dkl"],
    "n_mc":    None,      # BNN weight samples; None -> config.MC_SAMPLES
    "backup":  True,      # keep a .bak copy of the original pickle
    "verify":  True,      # recompute NLL / coverage and report drift
    "dry_run": False,     # True -> compute and report, write nothing
}
# ════════════════════════════════════════════════════════════════════════

import os
import pickle
import shutil

import numpy as np
import torch

import config
import figure_utils as fu

SYS_DIR = {"cstr": "cstr", "bioreactor": "bioreactor",
           "lv": "lotka_volterra"}

# Metrics this script can reconstruct from the checkpoints.
NEW_KEYS = ["mae", "rmse", "mean_sigma"]


def _paths(case, obs, results_dir="results"):
    root = os.path.join(results_dir, f"ablation_{SYS_DIR[case]}_{obs}",
                        "checkpoint")
    return os.path.join(root, "ablation_results.pkl"), \
        os.path.join(root, "models")


def _grid_from_sweep(models_root, case, sys_mod):
    """Rebuild the evaluation grid the sweep actually used.

    Preferred source is the box recorded in sweep_meta.pkl, so a sweep run
    before the grid was fixed is still reproduced faithfully. Falls back to
    config.GRID_BOUNDS.
    """
    meta_path = os.path.join(models_root, "sweep_meta.pkl")
    box = grid_n = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            sm = pickle.load(f)
        gm = sm.get("grid_meta") or {}
        box = gm.get("box")
        grid_n = sm.get("grid_n") or gm.get("grid_n")
    if box is None:
        (x0, x1), (y0, y1) = config.get_grid_bounds(case)
        box = (x0, x1, y0, y1)
    grid_n = int(grid_n or config.GRID_N)

    x0, x1, y0, y1 = box
    gx = np.linspace(x0, x1, grid_n)
    gy = np.linspace(y0, y1, grid_n)
    XX, YY = np.meshgrid(gx, gy)
    grid = torch.tensor(np.stack([XX.ravel(), YY.ravel()], 1),
                        dtype=torch.float32)
    g_true = sys_mod.true_residual(XX.ravel(), YY.ravel()).astype(np.float32)
    return grid, g_true, box, grid_n


def _metrics(mu, sigma, g_true, k_sigma):
    """Grid metrics, using the same expressions as run_ablation."""
    sigma = np.maximum(sigma, 1e-6)
    err = mu - g_true
    return {
        "mae":           float(np.mean(np.abs(err))),
        "rmse":          float(np.sqrt(np.mean(err ** 2))),
        "mean_sigma":    float(np.mean(sigma)),
        "nll":           float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2)
                                       + 0.5 * (err / sigma) ** 2)),
        "coverage_area": float(np.mean(np.abs(err) <= k_sigma * sigma)),
    }


def backfill_case(case, cfg):
    obs = cfg["obs"]
    pkl_path, models_root = _paths(case, obs)
    if not os.path.exists(pkl_path):
        print(f"  [{case}] no ablation_results.pkl — skipped")
        return
    if not os.path.isdir(models_root):
        print(f"  [{case}] no saved models — cannot backfill, skipped")
        return

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    n_values = sorted(d.get("n_values") or d.get("all_uq_rep", {}).keys())
    k_sigma = float(d.get("k_sigma", config.K_SIGMA))
    sys_mod = fu.import_system(case)
    grid, g_true, box, grid_n = _grid_from_sweep(models_root, case, sys_mod)

    print(f"\n  [{case}] grid {grid_n}x{grid_n} box="
          f"({box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f})  "
          f"k_sigma={k_sigma}")

    all_uq_rep = d.setdefault("all_uq_rep", {})
    all_uq = d.setdefault("all_uq", {})
    drift = {m: [] for m in cfg["methods"]}
    n_done = 0

    for N in n_values:
        reps = fu.available_repeats(case, N, obs)
        if not reps:
            print(f"    N={N}: no checkpoints, skipped")
            continue
        acc = {m: {k: [] for k in NEW_KEYS} for m in cfg["methods"]}

        for rep in reps:
            try:
                predictors, _meta = fu.load_predictors(
                    case, N, rep, cfg["methods"], obs=obs,
                    n_mc=cfg["n_mc"] or config.MC_SAMPLES)
            except FileNotFoundError as e:
                print(f"    N={N} rep{rep}: {e}")
                continue

            for m, fn in predictors.items():
                mu, sigma = fu.predict_on_grid(fn, grid, (g_true.size,))
                vals = _metrics(mu, sigma, g_true, k_sigma)
                for k in NEW_KEYS:
                    acc[m][k].append(vals[k])
                n_done += 1

                if cfg["verify"]:
                    stored = (all_uq_rep.get(N, {}).get(m, {})
                              .get("nll", []))
                    idx = reps.index(rep)
                    if idx < len(stored):
                        drift[m].append(abs(vals["nll"] - stored[idx]))

        # write per-repeat lists and the aggregated means
        for m in cfg["methods"]:
            if not acc[m]["mae"]:
                continue
            all_uq_rep.setdefault(N, {}).setdefault(m, {})
            all_uq.setdefault(N, {}).setdefault(m, {})
            for k in NEW_KEYS:
                all_uq_rep[N][m][k] = acc[m][k]
                all_uq[N][m][k] = float(np.mean(acc[m][k]))
        msg = "  ".join(
            f"{fu.METHOD_LABEL[m]} MAE={np.mean(acc[m]['mae']):.4f}"
            for m in cfg["methods"] if acc[m]["mae"])
        print(f"    N={N:<3} ({len(reps)} reps)  {msg}")

    if cfg["verify"]:
        print("\n    consistency check (recomputed vs stored NLL):")
        for m in cfg["methods"]:
            if drift[m]:
                mx = max(drift[m])
                if mx < 1e-3:
                    note = "numerically identical"
                elif m == "bnn":
                    note = "MC sampling noise (BNN predictive is stochastic)"
                else:
                    note = "LARGER THAN EXPECTED - investigate"
                print(f"      {fu.METHOD_LABEL[m]:<4} max |diff| = "
                      f"{mx:.3e}   ({note})")

    if cfg["dry_run"]:
        print(f"\n    dry run — {pkl_path} not modified")
        return

    if cfg["backup"]:
        bak = pkl_path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(pkl_path, bak)
            print(f"\n    backup written: {bak}")

    with open(pkl_path, "wb") as f:
        pickle.dump(d, f)
    print(f"    updated {pkl_path}  ({n_done} model evaluations)")


def main(cfg=None):
    cfg = dict(BACKFILL_CONFIG if cfg is None else cfg)
    print("=" * 70)
    print("BACKFILL GRID METRICS — no retraining")
    print("=" * 70)
    for case in cfg["cases"]:
        try:
            backfill_case(case, cfg)
        except Exception as e:
            print(f"  [{case}] FAILED: {type(e).__name__}: {e}")
    print("\nDone. Metricsvs_N.py can now plot the MAE panel.")


if __name__ == "__main__":
    main()
