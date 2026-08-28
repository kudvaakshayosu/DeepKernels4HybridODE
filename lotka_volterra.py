# -*- coding: utf-8 -*-
"""
systems/lotka_volterra.py — Lotka-Volterra with Holling Type III residual.

Parallel hybrid model:
    dx1/dt = alpha*x1 - beta*x1*x2
    dx2/dt = -gamma*x2 + delta*x1*x2 + g(x1, x2)

True residual: g*(x1,x2) = -c x1^2 / (1+x1^2) * x2  (Holling Type III).

Observation mode is controlled by config.OBS_MODE:
    'partial' → only x2 measured; x1 latent
    'full'    → both states measured with their own noise variance
"""
import numpy as np
from scipy.integrate import solve_ivp
import torch
from config import (LV_T_SPAN, LV_N_INTEGRATION, LV_NOISE_STD,
                    LV_X0_TRAIN_RANGE, LV_X0_TEST_RANGE, N_OBS_PER_TRAJ,
                    SEED, get_obs_indices, get_obs_config)

ALPHA, BETA, GAMMA, DELTA = 1.0, 0.5, 0.8, 0.3
C_TRUE = 0.3
T_INTEGRATION = np.linspace(LV_T_SPAN[0], LV_T_SPAN[1], LV_N_INTEGRATION)
OBS_INDICES = get_obs_indices(LV_N_INTEGRATION, N_OBS_PER_TRAJ)
SYSTEM_NAME = "Lotka-Volterra"
CASE_KEY = "lv"
INPUT_DIM = 2
STATE_DIM = 2
OUTPUT_DIM = 1
ARCHITECTURE = "parallel"
B_PARALLEL = torch.tensor([[0.0], [1.0]])


# ─── True / known dynamics ───────────────────────────────────────────────────

def true_residual(x1, x2):
    return -C_TRUE * x1 ** 2 / (1.0 + x1 ** 2) * x2

def true_dynamics(t, x):
    x1, x2 = x
    return [ALPHA * x1 - BETA * x1 * x2,
            -GAMMA * x2 + DELTA * x1 * x2 + true_residual(x1, x2)]

def f_known(t, x):
    x1 = x[..., 0]; x2 = x[..., 1]
    dx1 = ALPHA * x1 - BETA * x1 * x2
    dx2 = -GAMMA * x2 + DELTA * x1 * x2
    return torch.stack([dx1, dx2], dim=-1)


# ─── Data generation: observation slice driven by mode ───────────────────────

def generate_data(n_traj, x0_range, seed=SEED, mode=None):
    """Generate trajectories of the true system and observe a subset of states.

    Returns:
        x0s     : list of [STATE_DIM] initial conditions
        y_noisy : list of [n_obs, n_y] noisy observations (n_y depends on mode)
        y_clean : list of [n_obs, n_y] clean observations
        full    : list of [STATE_DIM, T] complete state trajectories
    """
    _, noise_var, obs_idx, _ = get_obs_config(CASE_KEY, mode)
    noise_std = torch.sqrt(noise_var).numpy()                     # [n_y]
    rng = np.random.RandomState(seed)
    x0s, y_noisy, y_clean, full = [], [], [], []
    att = 0
    while len(x0s) < n_traj and att < n_traj * 30:
        att += 1
        x0 = [rng.uniform(*x0_range), rng.uniform(*x0_range)]
        sol = solve_ivp(true_dynamics, LV_T_SPAN, x0, t_eval=T_INTEGRATION,
                        method='RK45', rtol=1e-8, atol=1e-10)
        if (sol.success and sol.y.shape[1] == LV_N_INTEGRATION and
                np.all(np.isfinite(sol.y)) and np.max(np.abs(sol.y)) < 100):
            x0s.append(x0)
            obs_clean = sol.y[obs_idx][:, OBS_INDICES].T              # [n_obs, n_y]
            y_clean.append(obs_clean)
            y_noisy.append(obs_clean + rng.normal(
                0, noise_std[None, :], obs_clean.shape))
            full.append(sol.y)
    assert len(x0s) >= n_traj, f"Only got {len(x0s)}/{n_traj}"
    return x0s[:n_traj], y_noisy[:n_traj], y_clean[:n_traj], full[:n_traj]


def build_traj_tensors(x0s, y_noisy_list):
    """Stack observations into [n_obs, N, n_y] regardless of mode."""
    x0_batch = torch.tensor(np.stack(x0s, 0), dtype=torch.float32)    # [N, 2]
    y_arr = np.stack(y_noisy_list, axis=1)                             # [n_obs, N, n_y]
    y_obs = torch.tensor(y_arr, dtype=torch.float32)
    t_grid = torch.tensor(T_INTEGRATION, dtype=torch.float32)
    obs_indices = torch.tensor(OBS_INDICES, dtype=torch.long)
    return x0_batch, y_obs, t_grid, obs_indices


def build_test_grids(full_states):
    grids, targets = [], []
    for s in full_states:
        grids.append(torch.tensor(np.stack([s[0], s[1]], 1), dtype=torch.float32))
        targets.append(torch.tensor(true_residual(s[0], s[1]), dtype=torch.float32))
    return grids, targets


def get_noise_std():
    return LV_NOISE_STD


def get_noise_var_diag(mode=None):
    _, noise_var, _, _ = get_obs_config(CASE_KEY, mode)
    return noise_var


def get_C(mode=None):
    C, _, _, _ = get_obs_config(CASE_KEY, mode)
    return C


def get_state_names():
    _, _, _, names = get_obs_config(CASE_KEY)
    return names


# ─── Hybrid factory ──────────────────────────────────────────────────────────

def make_hybrid_factory():
    # Local import avoids a circular dependency (surrogates import systems).
    from surrogates.traj_train import HybridParallel
    def factory(g_net):
        return HybridParallel(f_known, g_net, B_PARALLEL)
    return factory
