# -*- coding: utf-8 -*-
"""
systems/cstr.py — Isothermal CSTR with unknown rate expression.

Series hybrid with NO prior kinetic model. The surrogate g(Ca, Cb) learns
the full reaction rate r(Ca, Cb) from scratch, and is fed directly into
the KNOWN mass-balance equations:

        r_used(Ca, Cb) = g(Ca, Cb),
        r_true(Ca, Cb) = k0 · Ca / (1 + K1·Ca + K2·Cb)         (Langmuir-
                                                                Hinshelwood
                                                                truth).

Dynamics (mass balances are known; kinetics are entirely learned):

    A -> B in a CSTR with FIXED dilution rate D = F/V (1/min)

    dCa/dt = D·(Ca_feed − Ca) − r(Ca, Cb)
    dCb/dt = -D·Cb           + r(Ca, Cb)

The rate constant k(Ca, Cb) = k0/(1 + K1·Ca + K2·Cb) is a smooth nonlinear
function of *both* species (Cb inhibits the surface reaction via
adsorption), so the surrogate must capture a genuine 2-D constitutive
relation. The dynamics are bounded and well-behaved, which makes this
case study friendlier to GP-style priors than the unbounded fed-batch
bioreactor it replaces.

State order: (Ca, Cb). Observation mode driven by config.OBS_MODE.
"""
import numpy as np
from scipy.integrate import solve_ivp
import torch
from config import (CSTR_T_SPAN, CSTR_N_INTEGRATION,
                    CSTR_NOISE_CA, CSTR_NOISE_CB,
                    CSTR_X0_TRAIN, CSTR_X0_TEST, N_OBS_PER_TRAJ, SEED,
                    get_obs_indices, get_obs_config)

# ─── Mechanistic constants (known) ───────────────────────────────────────────
D_DIL = 0.3        # 1/min, FIXED dilution rate
CA_FEED = 1.0      # mol/L, feed concentration of A

# ─── Kinetic constants (the surrogate is learning this expression) ──────────
K0 = 1.5
K1 = 0.5
K2 = 2.0

T_INTEGRATION = np.linspace(CSTR_T_SPAN[0], CSTR_T_SPAN[1], CSTR_N_INTEGRATION)
OBS_INDICES = get_obs_indices(CSTR_N_INTEGRATION, N_OBS_PER_TRAJ)
SYSTEM_NAME = "CSTR"
CASE_KEY = "cstr"
INPUT_DIM = 2          # surrogate takes (Ca, Cb)
STATE_DIM = 2
OUTPUT_DIM = 1         # surrogate returns a scalar rate
ARCHITECTURE = "series"


# ─── True kinetics (used only for data generation and evaluation) ────────────

def rate_true_np(Ca, Cb):
    """Langmuir-Hinshelwood rate expression with product inhibition.

    r*(Ca, Cb) = k0 · Ca / (1 + K1·Ca + K2·Cb).
    """
    Ca = np.maximum(Ca, 0.0)
    Cb = np.maximum(Cb, 0.0)
    return K0 * Ca / (1.0 + K1 * Ca + K2 * Cb)


def true_residual(Ca, Cb):
    """Target the surrogate should recover: r_true(Ca, Cb) itself.

    Named `true_residual` for API uniformity with the LV case (used by the
    force-surface plotting code, which calls `sys_mod.true_residual`)."""
    return rate_true_np(Ca, Cb)


def true_dynamics(t, state):
    """SciPy RHS for data generation (uses the TRUE rate expression)."""
    Ca, Cb = state
    r = rate_true_np(Ca, Cb)
    return [D_DIL * (CA_FEED - Ca) - r,
            -D_DIL * Cb            + r]


# ─── Hybrid RHS: surrogate provides the full reaction rate ───────────────────

def f_with_g(t, x, g_val):
    """Hybrid RHS with the surrogate g(Ca, Cb) providing the full rate.

    r_used(Ca, Cb) = g(Ca, Cb).
    """
    Ca = x[..., 0].clamp_min(0)        # concentrations cannot be negative
    Cb = x[..., 1].clamp_min(0)
    r = g_val.squeeze(-1)
    dCa = D_DIL * (CA_FEED - Ca) - r
    dCb = -D_DIL * Cb            + r
    return torch.stack([dCa, dCb], dim=-1)


def input_selector(x):
    """The surrogate g(.) takes the full state (Ca, Cb)."""
    return x[..., :2]


# ─── Data generation: noisy observations of a subset of states ───────────────

def generate_data(n_traj, x0_ranges=None, seed=SEED, mode=None):
    """Generate true CSTR trajectories and observe a subset of states."""
    if x0_ranges is None: x0_ranges = CSTR_X0_TRAIN
    _, noise_var, obs_idx, _ = get_obs_config(CASE_KEY, mode)
    noise_std = torch.sqrt(noise_var).numpy()
    rng = np.random.RandomState(seed)
    x0s, y_noisy, y_clean, full = [], [], [], []
    att = 0
    while len(x0s) < n_traj and att < n_traj * 30:
        att += 1
        x0 = [rng.uniform(*x0_ranges['Ca']),
              rng.uniform(*x0_ranges['Cb'])]
        sol = solve_ivp(true_dynamics, CSTR_T_SPAN, x0, t_eval=T_INTEGRATION,
                        method='RK45', rtol=1e-8, atol=1e-10)
        if (sol.success and sol.y.shape[1] == CSTR_N_INTEGRATION and
                np.all(np.isfinite(sol.y))):
            sol.y[0] = np.maximum(sol.y[0], 0)
            sol.y[1] = np.maximum(sol.y[1], 0)
            x0s.append(x0)
            obs_clean = sol.y[obs_idx][:, OBS_INDICES].T          # [n_obs, n_y]
            y_clean.append(obs_clean)
            y_noisy.append(obs_clean + rng.normal(
                0, noise_std[None, :], obs_clean.shape))
            full.append(sol.y)
    assert len(x0s) >= n_traj, f"Only got {len(x0s)}/{n_traj}"
    return x0s[:n_traj], y_noisy[:n_traj], y_clean[:n_traj], full[:n_traj]


def build_traj_tensors(x0s, y_noisy_list):
    x0_batch = torch.tensor(np.stack(x0s, 0), dtype=torch.float32)
    y_arr = np.stack(y_noisy_list, axis=1)                         # [n_obs, N, n_y]
    y_obs = torch.tensor(y_arr, dtype=torch.float32)
    t_grid = torch.tensor(T_INTEGRATION, dtype=torch.float32)
    obs_indices = torch.tensor(OBS_INDICES, dtype=torch.long)
    return x0_batch, y_obs, t_grid, obs_indices


def build_test_grids(full_states):
    """For surface evaluation, the surrogate consumes (Ca, Cb) (INPUT_DIM=2).
    Targets are the true reaction rate r_true(Ca, Cb) at the state values
    from the true trajectory."""
    grids, targets = [], []
    for s in full_states:
        grids.append(torch.tensor(np.stack([s[0], s[1]], 1),
                                  dtype=torch.float32))                  # [T, 2]
        targets.append(torch.tensor(rate_true_np(s[0], s[1]),
                                    dtype=torch.float32))
    return grids, targets


def get_noise_std():
    return (CSTR_NOISE_CA + CSTR_NOISE_CB) / 2


def get_noise_var_diag(mode=None):
    _, noise_var, _, _ = get_obs_config(CASE_KEY, mode)
    return noise_var


def get_C(mode=None):
    C, _, _, _ = get_obs_config(CASE_KEY, mode)
    return C


def get_state_names():
    _, _, _, names = get_obs_config(CASE_KEY)
    return names


def make_hybrid_factory():
    # Local import avoids a circular dependency (surrogates import systems).
    from surrogates.traj_train import HybridSeries
    def factory(g_net):
        return HybridSeries(f_with_g, g_net, input_selector)
    return factory
