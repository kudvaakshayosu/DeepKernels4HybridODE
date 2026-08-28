# -*- coding: utf-8 -*-
"""
surrogates/mlp.py — Deterministic MLP surrogate, integrator-in-the-loop training.

The simplest baseline: a plain feed-forward network g(.) trained by rolling the
hybrid ODE forward and minimizing the Gaussian observation NLL. It has NO
epistemic uncertainty (a single deterministic function), so its predictive
sigma is reported as zero. It exists to show the best achievable point
accuracy at this parameter budget — the bar the probabilistic methods must
match while also producing calibrated uncertainty.
"""
import torch
import torch.nn as nn
from config import HIDDEN_DIM, LR, N_EPOCHS, L2_LAMBDA
from surrogates.traj_train import (train_traj_deterministic, rollout,
                                    predict_traj_mean)


class MLPNet(nn.Module):
    """2-hidden-layer MLP. Standardizes the raw input and un-standardizes the
    output, so the network learns a ~unit-scale target on ~unit-scale inputs.
    Output head initialized small so g(x) ~ 0 at the start, which keeps the
    integrator stable in early training."""
    def __init__(self, input_dim, hidden=HIDDEN_DIM, output_dim=1,
                 standardizer=None):
        super().__init__()
        self.std = standardizer
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, output_dim))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, x):
        if self.std is not None:
            x = self.std.encode(x)
        out = self.net(x)
        if self.std is not None:
            out = self.std.decode_g(out)
        return out


def train_trajectory(sys_mod, x0_batch, y_obs, t_grid, obs_indices,
                     noise_var_diag, full_states_train=None,
                     epochs=N_EPOCHS, lr=LR,
                     hidden=HIDDEN_DIM):
    from surrogates.traj_train import compute_standardizer
    std = (compute_standardizer(full_states_train, sys_mod)
           if full_states_train is not None else None)
    g_net = MLPNet(sys_mod.INPUT_DIM, hidden, output_dim=sys_mod.OUTPUT_DIM,
                   standardizer=std)
    factory = sys_mod.make_hybrid_factory()
    C_obs = sys_mod.get_C()
    return train_traj_deterministic(
        g_net, factory, x0_batch, t_grid, y_obs, obs_indices,
        C_obs, noise_var_diag, epochs=epochs, lr=lr, l2=L2_LAMBDA,
        verbose_tag="MLP")


def predict_trajectory(g_net, sys_mod, x0, t_grid, n_mc=1):
    """Deterministic: a single rollout, returned with a leading sample axis of
    size 1 so the evaluation code can treat all methods uniformly."""
    factory = sys_mod.make_hybrid_factory()
    with torch.no_grad():
        traj = rollout(factory(g_net), x0, t_grid)
    return traj.unsqueeze(0)


def predict(model, grid):
    """Pointwise prediction of g(.) for force-surface plots. No epistemic
    uncertainty for a deterministic net, so variance is zero."""
    model.eval()
    with torch.no_grad():
        mean = model(grid).squeeze(-1)
    zero = torch.zeros_like(mean)
    return mean, zero, zero
