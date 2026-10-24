# -*- coding: utf-8 -*-
"""
config.py — Central configuration for the simplified hybrid-ODE UQ benchmark.

Three case studies, each available in two observation modes:

   - 'partial' : only the channels listed in *_OBS_IDX_PARTIAL are observed
                 (e.g. predator only for LV; product C_B only for CSTR;
                 biomass X only for the bioreactor). Other states are LATENT.
   - 'full'    : every state is observed independently with its own noise
                 variance.

Four surrogate methods are compared, all trained the SAME way (integrator-in-
the-loop trajectory likelihood):
   - MLP : deterministic baseline (no uncertainty)
   - BNN : Bayesian neural net (variational weight posteriors)
   - VGP : sparse variational GP in input space
   - DKL : deep kernel learning (NN feature extractor + sparse variational GP)

Model sizes are the "generous" values that give well-behaved DKL uncertainty;
they are NOT matched to a tiny parameter budget (matching crushes the GP).
"""
import numpy as np
import torch

# ── Observation mode ─────────────────────────────────────────────────────────
OBS_MODE = 'full'         # 'partial' or 'full'; override with --obs at the CLI

N_TRAIN = 5
N_TEST = 100
N_EPOCHS = 600
N_OBS_PER_TRAJ = 5
SEED = 1000

# ── Model hyperparameters ────────────────────────────────────────────────────
HIDDEN_DIM = 16           # MLP / DKL feature-extractor hidden width
BNN_HIDDEN_DIM = 16       # BNN hidden width
MC_SAMPLES = 1500          # MC rollouts at evaluation time
L2_LAMBDA = 1e-4
LR = 0.003                # NN learning rate
LR_GP = 0.01              # GP learning rate
N_INDUCING = 20           # inducing points for VGP / DKL
DKL_FEATURE_DIM = 6       # DKL latent feature dimension

# ── Integrator-in-the-loop training settings ─────────────────────────────────
ODE_SOLVER = 'euler'
TRAJ_TRAIN_MC = 3         # MC rollouts per gradient step for stochastic methods

# ── Inference benchmark ──────────────────────────────────────────────────────
INFERENCE_GRID_SIZES = [10, 50, 100, 500, 1000]
INFERENCE_REPEATS = 20

# ═══════════════════════════════════════════════════════════════════════════
# Case 1: Lotka-Volterra (parallel hybrid, Holling-III residual)
# ═══════════════════════════════════════════════════════════════════════════
LV_T_SPAN = [0, 12]
LV_N_INTEGRATION = 60
LV_NOISE_STD = 0.15
LV_X0_TRAIN_RANGE = (1.0, 2.5)
LV_X0_TEST_RANGE = (1.0, 2.5)   # same box as train
LV_C_PARTIAL = torch.tensor([[0.0, 1.0]])
LV_C_FULL    = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
LV_NOISE_VAR_PARTIAL = torch.tensor([LV_NOISE_STD ** 2])
LV_NOISE_VAR_FULL    = torch.tensor([LV_NOISE_STD ** 2, LV_NOISE_STD ** 2])
LV_OBS_IDX_PARTIAL = [1]
LV_OBS_IDX_FULL    = [0, 1]
LV_STATE_NAMES = ['Prey $x_1$', 'Predator $x_2$']

# ═══════════════════════════════════════════════════════════════════════════
# Case 2: CSTR (series hybrid, Langmuir-Hinshelwood rate)
# ═══════════════════════════════════════════════════════════════════════════
CSTR_T_SPAN = [0, 8]
CSTR_N_INTEGRATION = 60
CSTR_NOISE_CA = 0.03
CSTR_NOISE_CB = 0.03
CSTR_X0_TRAIN = {'Ca': (0.6, 1.0), 'Cb': (0.0, 0.2)}
CSTR_X0_TEST  = {'Ca': (0.6, 1.0), 'Cb': (0.0, 0.2)}   # same box as train
CSTR_C_PARTIAL = torch.tensor([[0.0, 1.0]])
CSTR_C_FULL    = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
CSTR_NOISE_VAR_PARTIAL = torch.tensor([CSTR_NOISE_CB ** 2])
CSTR_NOISE_VAR_FULL    = torch.tensor([CSTR_NOISE_CA ** 2, CSTR_NOISE_CB ** 2])
CSTR_OBS_IDX_PARTIAL = [1]
CSTR_OBS_IDX_FULL    = [0, 1]
CSTR_STATE_NAMES = ['$C_A$ (mol/L)', '$C_B$ (mol/L)']

# ═══════════════════════════════════════════════════════════════════════════
# Case 3: Bioreactor (series hybrid, Haldane-logistic kinetics)
# Non-stationary unknown with two coexisting regimes; stresses stationary GP
# kernels and showcases DKL's learned feature warping.
# ═══════════════════════════════════════════════════════════════════════════
BIOREACTOR_T_SPAN = [0.0, 24.0]
BIOREACTOR_N_INTEGRATION = 60
BIOREACTOR_NOISE_X = 0.10
BIOREACTOR_NOISE_S = 0.05
BIOREACTOR_X0_TRAIN = {'X': (0.3, 1.2), 'S': (1.5, 3.0)}
BIOREACTOR_X0_TEST  = {'X': (0.3, 1.2), 'S': (1.5, 3.0)}   # same box as train
BIOREACTOR_C_PARTIAL = torch.tensor([[1.0, 0.0]])
BIOREACTOR_C_FULL    = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
BIOREACTOR_NOISE_VAR_PARTIAL = torch.tensor([BIOREACTOR_NOISE_X ** 2])
BIOREACTOR_NOISE_VAR_FULL    = torch.tensor([BIOREACTOR_NOISE_X ** 2,
                                              BIOREACTOR_NOISE_S ** 2])
BIOREACTOR_OBS_IDX_PARTIAL = [0]
BIOREACTOR_OBS_IDX_FULL    = [0, 1]
BIOREACTOR_STATE_NAMES = ['Biomass $X$ (g/L)', 'Substrate $S$ (g/L)']


# ── Helper to fetch observation setup by case + mode ─────────────────────────
def get_obs_config(case, mode=None):
    """Return (C_matrix, noise_var_diag, observed_state_indices, state_names)."""
    mode = mode or OBS_MODE
    assert mode in ('partial', 'full'), f"OBS_MODE must be partial|full, got {mode}"
    if case == 'lv':
        if mode == 'partial':
            return LV_C_PARTIAL, LV_NOISE_VAR_PARTIAL, LV_OBS_IDX_PARTIAL, LV_STATE_NAMES
        return LV_C_FULL, LV_NOISE_VAR_FULL, LV_OBS_IDX_FULL, LV_STATE_NAMES
    if case == 'cstr':
        if mode == 'partial':
            return CSTR_C_PARTIAL, CSTR_NOISE_VAR_PARTIAL, CSTR_OBS_IDX_PARTIAL, CSTR_STATE_NAMES
        return CSTR_C_FULL, CSTR_NOISE_VAR_FULL, CSTR_OBS_IDX_FULL, CSTR_STATE_NAMES
    if case == 'bioreactor':
        if mode == 'partial':
            return (BIOREACTOR_C_PARTIAL, BIOREACTOR_NOISE_VAR_PARTIAL,
                    BIOREACTOR_OBS_IDX_PARTIAL, BIOREACTOR_STATE_NAMES)
        return (BIOREACTOR_C_FULL, BIOREACTOR_NOISE_VAR_FULL,
                BIOREACTOR_OBS_IDX_FULL, BIOREACTOR_STATE_NAMES)
    raise ValueError(f"Unknown case {case}")


def get_obs_indices(n_int, n_obs, seed=SEED):
    """Sample observation time indices, leaving t=0 unobserved."""
    return np.sort(np.random.RandomState(seed).choice(
        np.arange(1, n_int), n_obs, replace=False))

# ═══════════════════════════════════════════════════════════════════════════
# EXPLICIT EVALUATION-GRID BOUNDS  (Eq. 14 of the manuscript)
# ───────────────────────────────────────────────────────────────────────────
# The evaluation grid is now a FIXED, hard-coded box per case rather than a
# data-dependent bounding rectangle inferred from trajectories. This makes the
# reported grid identical across every script, seed, N and repeat, and makes it
# match the domain stated in the paper by construction rather than by
# coincidence. No padding is applied.
#
#   CSTR       : (C_A, C_B) in [0, 1.1] x [0, 1.2]   mol/L
#   Bioreactor : (X,   S)   in [0, 2.5] x [0, 3.0]   g/L
# ═══════════════════════════════════════════════════════════════════════════
GRID_BOUNDS = {
    'cstr':       {'x': (0.0, 1.1), 'y': (0.0, 1.2)},
    'bioreactor': {'x': (0.0, 2.5), 'y': (0.0, 3.0)},
    'lv':         {'x': (0.0, 3.0), 'y': (0.0, 3.0)},
}

GRID_N = 60        # grid resolution per axis -> GRID_N**2 evaluation points
GRID_PAD = 0.0     # no padding; bounds above are used verbatim
K_SIGMA = 3.0      # band half-width in sigmas for the coverage-area metric


def get_grid_bounds(case):
    """Return ((x0, x1), (y0, y1)) for the fixed evaluation grid of `case`."""
    if case not in GRID_BOUNDS:
        raise ValueError(f"No GRID_BOUNDS defined for case '{case}'")
    b = GRID_BOUNDS[case]
    return b['x'], b['y']
