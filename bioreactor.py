# -*- coding: utf-8 -*-
"""
systems/bioreactor.py — Fed-batch bioreactor with Haldane substrate inhibition
and logistic biomass-density limitation. Series hybrid (CSTR-style).

This case study is designed to stress test stationary GP kernels. The unknown
component is a specific biomass formation rate g*(X, S) that combines:
  - Haldane kinetics in substrate: mu(S) = mu_max * S / (K_s + S + S^2/K_I)
    which rises sharply from 0 at low S, peaks near S* = sqrt(K_s * K_I),
    and decays at high S due to substrate inhibition.
  - Logistic biomass-density limitation: (1 - X/X_max), which captures the
    self-crowding effect (oxygen limitation, nutrient depletion, etc).
  - Mass-action: multiplied by X (biomass concentration) because growth is
    a per-cell process aggregated over the population.

Combined:
    g*(X, S) = mu_max * X * S * (1 - X/X_max) / (K_s + S + S^2/K_I)

WHY THIS IS HARD FOR A STATIONARY GP
====================================
The function g* has TWO operating regimes coexisting in (X, S) space:
  (A) Substrate-limited regime (low S, moderate X): g* varies SHARPLY in S,
      slowly in X. The dominant lengthscale is small in S, large in X.
  (B) Inhibition/biomass-limited regime (high S OR high X): g* varies
      slowly in S (it's already past the Haldane peak), and the dominant
      sensitivity is to (1 - X/X_max). The dominant lengthscale is large
      in S, small in X.

A stationary ARD-RBF kernel must commit to a SINGLE pair of ARD
lengthscales for the entire input space, so it ends up:
  - either short in both → over-fitting (high variance everywhere),
  - or long in both → over-smoothing the Haldane peak,
  - or a compromise that wins nothing.

DKL's neural feature extractor can warp the input space differently in
each regime — features can be sensitive to S in regime A and to X in
regime B — so the kernel becomes effectively non-stationary in the
original input space. This is exactly the structural advantage the paper
claims for DKL.

Dynamics (mass balances known, kinetics learned):
    dX/dt = g(X, S) - D*X                    # growth minus dilution
    dS/dt = D*(S_feed - S) - g(X, S)/Y_XS    # feed minus consumption

Constants chosen so that:
  - Haldane peak is at S* = sqrt(K_s * K_I) ~ 1.58 g/L,
  - Biomass cap at X_max = 10 g/L,
  - Training trajectories visit BOTH regimes over t in [0, 24] h.
"""
import numpy as np
from scipy.integrate import solve_ivp
import torch
from config import (BIOREACTOR_T_SPAN, BIOREACTOR_N_INTEGRATION,
                    BIOREACTOR_NOISE_X, BIOREACTOR_NOISE_S,
                    N_OBS_PER_TRAJ, SEED, get_obs_indices, get_obs_config)

# ─── Mechanistic / kinetic constants ────────────────────────────────────────
# Yield, dilution, feed: known.
Y_XS    = 0.5                     # biomass yield per substrate consumed
D_DIL   = 0.10                    # dilution rate (1/h)
S_FEED  = 4.0                     # feed substrate concentration (g/L)

# Unknown-kinetics parameters (used only to GENERATE data; the surrogate
# must learn the resulting g*(X, S) without seeing these).
MU_MAX  = 0.6                     # max specific growth rate (1/h)
K_S     = 0.5                     # half-saturation in substrate (g/L)
K_I     = 5.0                     # substrate-inhibition constant (g/L)
X_MAX   = 10.0                    # biomass saturation (g/L)
# Haldane peak occurs at S* = sqrt(K_S * K_I); peak value mu(S*) is below mu_max
# because of the inhibition term. For our values: S* ~ 1.58, mu(S*) ~ 0.27.

T_INTEGRATION = np.linspace(BIOREACTOR_T_SPAN[0],
                             BIOREACTOR_T_SPAN[1],
                             BIOREACTOR_N_INTEGRATION)
OBS_INDICES   = get_obs_indices(BIOREACTOR_N_INTEGRATION, N_OBS_PER_TRAJ)
SYSTEM_NAME, CASE_KEY = "Bioreactor", "bioreactor"
INPUT_DIM, STATE_DIM, OUTPUT_DIM = 2, 2, 1
ARCHITECTURE = "series"


# ─── Ground-truth rate (numpy and torch flavors) ────────────────────────────
def rate_true_np(X, S):
    """Specific biomass formation rate g*(X, S). Smooth but non-stationary."""
    X = np.maximum(X, 0.0)
    S = np.maximum(S, 0.0)
    denom = K_S + S + S * S / K_I                                # > 0 always
    growth = MU_MAX * X * S * np.maximum(1.0 - X / X_MAX, 0.0) / denom
    return growth

def true_residual(X, S):           # kept for plotting-API uniformity
    return rate_true_np(X, S)

def true_dynamics(t, state):
    X, S = state
    r = rate_true_np(X, S)
    return [r - D_DIL * X,
            D_DIL * (S_FEED - S) - r / Y_XS]


# ─── Hybrid RHS (series form) ───────────────────────────────────────────────
def f_with_g(t, x, g_val):
    """Hybrid RHS where the surrogate g(X, S) supplies the rate.

    Note: we intentionally do NOT clamp negativity inside f_with_g (just like
    cstr.py); the soft-clamp wrappers in surrogates/core.py provide the
    stability margin. We DO clamp X and S to >= 0 before feeding to g, since
    the integrator can briefly take x slightly negative during transient
    early-training rollouts."""
    X = x[..., 0].clamp_min(0.0)
    S = x[..., 1].clamp_min(0.0)
    r = g_val.squeeze(-1)
    return torch.stack([r - D_DIL * X,
                         D_DIL * (S_FEED - S) - r / Y_XS],
                        dim=-1)


def input_selector(x):
    """g(.) takes the full state (X, S)."""
    return x[..., :2]


# ─── Data generation ────────────────────────────────────────────────────────
def generate_data(n_traj, x0_ranges, seed=SEED, mode=None):
    """Generate n_traj trajectories with x0 sampled uniformly from x0_ranges.

    Returns (x0s, y_noisy, y_clean, full_states) — same interface as cstr.py.
    Trajectories that diverge or go non-finite are rejected and resampled."""
    _, noise_var, obs_idx, _ = get_obs_config(CASE_KEY, mode)
    noise_std = torch.sqrt(noise_var).numpy()
    rng = np.random.RandomState(seed)
    x0s, y_noisy, y_clean, full = [], [], [], []
    att = 0
    while len(x0s) < n_traj and att < n_traj * 30:
        att += 1
        x0 = [rng.uniform(*x0_ranges['X']),
              rng.uniform(*x0_ranges['S'])]
        sol = solve_ivp(true_dynamics, BIOREACTOR_T_SPAN, x0,
                         t_eval=T_INTEGRATION, method='RK45',
                         rtol=1e-8, atol=1e-10)
        if (sol.success
              and sol.y.shape[1] == BIOREACTOR_N_INTEGRATION
              and np.all(np.isfinite(sol.y))):
            sol.y[0] = np.maximum(sol.y[0], 0)
            sol.y[1] = np.maximum(sol.y[1], 0)
            x0s.append(x0)
            obs_clean = sol.y[obs_idx][:, OBS_INDICES].T
            y_clean.append(obs_clean)
            y_noisy.append(obs_clean + rng.normal(0, noise_std[None, :],
                                                    obs_clean.shape))
            full.append(sol.y)
    assert len(x0s) >= n_traj, (
        f"Could not generate {n_traj} valid trajectories after {att} attempts")
    return x0s[:n_traj], y_noisy[:n_traj], y_clean[:n_traj], full[:n_traj]


# ─── Tensor packaging (uniform interface) ──────────────────────────────────
def build_traj_tensors(x0s, y_noisy_list):
    x0_batch = torch.tensor(np.stack(x0s, 0), dtype=torch.float32)
    y_obs    = torch.tensor(np.stack(y_noisy_list, axis=1), dtype=torch.float32)
    t_grid   = torch.tensor(T_INTEGRATION, dtype=torch.float32)
    obs_idx  = torch.tensor(OBS_INDICES, dtype=torch.long)
    return x0_batch, y_obs, t_grid, obs_idx


def build_test_grids(full_states):
    """Trajectory-aligned test grids for surface-RMSE: at every integration
    grid point along every test trajectory, return (X, S) and the true rate."""
    grids, targets = [], []
    for s in full_states:
        grids.append(torch.tensor(np.stack([s[0], s[1]], 1), dtype=torch.float32))
        targets.append(torch.tensor(rate_true_np(s[0], s[1]), dtype=torch.float32))
    return grids, targets


# ─── Observation-model accessors ──────────────────────────────────────────
def get_noise_var_diag(mode=None):
    _, noise_var, _, _ = get_obs_config(CASE_KEY, mode)
    return noise_var

def get_C(mode=None):
    C, _, _, _ = get_obs_config(CASE_KEY, mode)
    return C

def get_state_names():
    _, _, _, names = get_obs_config(CASE_KEY)
    return names


# ─── Hybrid factory (consumed by train_elbo / train_deterministic) ─────────
def make_hybrid_factory():
    # Local import avoids a circular dependency (surrogates import systems).
    from surrogates.traj_train import HybridSeries
    def factory(g_net):
        return HybridSeries(f_with_g, g_net, input_selector)
    return factory
