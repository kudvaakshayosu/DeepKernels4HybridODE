# -*- coding: utf-8 -*-
"""evaluate.py — Trajectory-level evaluation for integrator-in-the-loop models.

We evaluate each surrogate by rolling out the hybrid ODE forward from the
test initial conditions and comparing the predicted observed channels
(C @ x_t) against the clean test trajectories.

Metrics:
    rmse_obs  : RMSE on observed states, per trajectory
    coverage  : fraction of clean test points inside the predictive 95% CI
                of the rollout (epistemic + aleatoric obs noise)
    width     : mean predictive 95% CI width on observed channels
    surf_rmse : RMSE of the surrogate g(.) on the (state-space) test grid
"""
import torch
import numpy as np
import time
import pickle
import os
from collections import defaultdict
from config import INFERENCE_GRID_SIZES, INFERENCE_REPEATS


def evaluate_trajectory(name, predict_traj_fn, model, sys_mod, x0_test, t_grid,
                         full_test, noise_var_diag, C_obs, n_mc=100):
    """Roll out the hybrid model from x0_test, compare to clean trajectories
    on observed channels. Returns per-trajectory dicts of rmse/coverage/width."""
    rmse_list, cov_list, wid_list = [], [], []
    n_test = x0_test.shape[0]
    with torch.no_grad():
        trajs = predict_traj_fn(model, sys_mod, x0_test, t_grid, n_mc=n_mc)
        # trajs shape: [n_mc_or_ensemble, T, n_test, n_x]
        # Project to observation space
        obs_pred = trajs @ C_obs.t()                      # [.., T, n_test, n_y]
        pred_mean = obs_pred.mean(0)                       # [T, n_test, n_y]
        pred_var_epi = obs_pred.var(0)                     # [T, n_test, n_y]
        # Add aleatoric (observation noise) on top
        pred_var_total = pred_var_epi #+ noise_var_diag.view(1, 1, -1) # get rid of the aleotoric coz noise free trajectory
        pred_std = torch.sqrt(pred_var_total)

    # Ground truth observed trajectory
    for i in range(n_test):
        truth_full = torch.tensor(full_test[i], dtype=torch.float32)   # [n_x, T]
        truth_obs = (truth_full.t() @ C_obs.t())                       # [T, n_y]
        diff = (pred_mean[:, i] - truth_obs)                           # [T, n_y]
        rmse_list.append(torch.sqrt((diff ** 2).mean()).item())
        std_i = pred_std[:, i]
        inside = ((truth_obs >= pred_mean[:, i] - 1.96 * std_i) &
                  (truth_obs <= pred_mean[:, i] + 1.96 * std_i))
        cov_list.append(inside.float().mean().item())
        wid_list.append((2 * 1.96 * std_i).mean().item())
    return rmse_list, cov_list, wid_list


def evaluate_surface(predict_fn, model, grids, targets):
    """RMSE of surrogate g(.) on the state-space test grid (for plotting)."""
    rmse = []
    with torch.no_grad():
        for g, t in zip(grids, targets):
            mu, _, _ = predict_fn(model, g)
            rmse.append(torch.sqrt(((mu - t) ** 2).mean()).item())
    return rmse


def evaluate_all(trained, predict_traj_fns, predict_pt_fns, sys_mod,
                  x0_test, t_grid, full_test, noise_var_diag,
                  surf_grids=None, surf_targets=None):
    """Run trajectory + surface evaluation for every trained model."""
    rmse = defaultdict(list); cov = defaultdict(list); wid = defaultdict(list)
    surf_rmse = defaultdict(list)
    C_obs = sys_mod.get_C()
    for name, model in trained.items():
        print(f"  Evaluating {name}...")
        r, c, w = evaluate_trajectory(
            name, predict_traj_fns[name], model, sys_mod, x0_test, t_grid,
            full_test, noise_var_diag, C_obs)
        rmse[name].extend(r); cov[name].extend(c); wid[name].extend(w)
        if surf_grids is not None and surf_targets is not None:
            sr = evaluate_surface(predict_pt_fns[name], model,
                                   surf_grids, surf_targets)
            surf_rmse[name].extend(sr)
    return {'rmse': dict(rmse), 'coverage': dict(cov), 'width': dict(wid),
            'surf_rmse': dict(surf_rmse)}


def benchmark_inference_cost(trained, predict_pt_fns, input_dim=2,
                              sizes=INFERENCE_GRID_SIZES, n_rep=INFERENCE_REPEATS):
    print("\n  Benchmarking inference cost (pointwise g(.))...")
    cost = {}
    for name, model in trained.items():
        pf = predict_pt_fns[name]; timings = {}
        for N in sizes:
            g = torch.randn(N, input_dim)
            for _ in range(2):
                try: pf(model, g)
                except Exception: pass
            times = []
            for _ in range(n_rep):
                t0 = time.perf_counter()
                try: pf(model, g)
                except Exception: continue
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
            if times:
                timings[N] = {'median_ms': float(np.median(times)),
                              'mean_ms': float(np.mean(times)),
                              'std_ms': float(np.std(times))}
        cost[name] = timings
        if 100 in timings:
            print(f"    {name}: {timings[100]['median_ms']:.2f} ms @ N=100")
    return cost


def save_all_results(output_dir, metrics, results, inference_cost, model_order,
                     train_data, test_data, config_dict):
    ckpt = os.path.join(output_dir, "checkpoint")
    os.makedirs(ckpt, exist_ok=True)
    with open(os.path.join(ckpt, "all_results.pkl"), 'wb') as f:
        pickle.dump({
            'metrics': metrics, 'results': results,
            'inference_cost': inference_cost,
            'model_order': model_order, 'config': config_dict,
            'train_data': train_data, 'test_data': test_data,
        }, f)
    print(f"  All results saved to {ckpt}/all_results.pkl")


def load_results(output_dir):
    with open(os.path.join(output_dir, "checkpoint", "all_results.pkl"), 'rb') as f:
        return pickle.load(f)
