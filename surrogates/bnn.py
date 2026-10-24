# -*- coding: utf-8 -*-
"""
surrogates/bnn_weights.py — BNN with integrator-in-the-loop ELBO training.

The surrogate g(.) is a Bayesian neural net with Gaussian variational
posteriors over weights. Training maximizes a trajectory ELBO:

   ELBO = E_q[log p(y | rollout(g))] - β KL(q || p)

The expectation is approximated by drawing TRAJ_TRAIN_MC weight samples
per gradient step, integrating the hybrid ODE through each, and averaging
the Gaussian observation NLL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from config import (BNN_HIDDEN_DIM, LR, N_EPOCHS, MC_SAMPLES, TRAJ_TRAIN_MC)
from surrogates.traj_train import (train_traj_stochastic, rollout,
                                    predict_traj_mc)

KL_WARMUP_FRAC = 0.3


class BayesLinear(nn.Module):
    def __init__(self, in_f, out_f, prior_scale=1.0, is_output=False):
        super().__init__()
        self.prior_std = prior_scale / np.sqrt(in_f)
        # Output layer: initialize tiny so g(x) ≈ 0 at start (stable rollout)
        mu_scale = 0.01 if is_output else (1.0 / np.sqrt(in_f))
        self.w_mu = nn.Parameter(torch.randn(out_f, in_f) * mu_scale)
        self.b_mu = nn.Parameter(torch.zeros(out_f))
        # Posterior σ initialized to small fraction of prior σ; output layer tighter
        sigma_frac = 0.01 if is_output else 0.05
        init_logsigma = np.log(sigma_frac * self.prior_std)
        self.w_logsigma = nn.Parameter(torch.full((out_f, in_f), init_logsigma))
        self.b_logsigma = nn.Parameter(torch.full((out_f,), init_logsigma))

    def forward(self, x):
        w_sigma = F.softplus(self.w_logsigma)
        b_sigma = F.softplus(self.b_logsigma)
        act_mu = F.linear(x, self.w_mu, self.b_mu)
        act_var = F.linear(x.pow(2), w_sigma.pow(2), b_sigma.pow(2))
        act_std = torch.sqrt(act_var + 1e-8)
        return act_mu + act_std * torch.randn_like(act_mu)

    def kl(self):
        w_sigma = F.softplus(self.w_logsigma)
        b_sigma = F.softplus(self.b_logsigma)
        ps2 = self.prior_std ** 2
        kl_w = 0.5 * (w_sigma.pow(2) / ps2 + self.w_mu.pow(2) / ps2
                      - 1 - 2 * torch.log(w_sigma / self.prior_std + 1e-8))
        kl_b = 0.5 * (b_sigma.pow(2) / ps2 + self.b_mu.pow(2) / ps2
                      - 1 - 2 * torch.log(b_sigma / self.prior_std + 1e-8))
        return kl_w.sum() + kl_b.sum()


class BayesianNet(nn.Module):
    def __init__(self, input_dim, hidden=32, output_dim=1, noise_std=0.1,
                 standardizer=None):
        super().__init__()
        self.std = standardizer
        # Modest prior scale to avoid wide initial outputs that break the integrator
        prior_scale = 1.0
        self.fc1 = BayesLinear(input_dim, hidden, prior_scale, is_output=False)
        self.fc2 = BayesLinear(hidden, hidden, prior_scale, is_output=False)
        self.fc3 = BayesLinear(hidden, output_dim, prior_scale, is_output=True)

    def forward(self, x):
        if self.std is not None:
            x = self.std.encode(x)
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        out = self.fc3(h)
        if self.std is not None:
            out = self.std.decode_g(out)
        return out

    def kl(self):
        return self.fc1.kl() + self.fc2.kl() + self.fc3.kl()


def train_trajectory(sys_mod, x0_batch, y_obs, t_grid, obs_indices,
                     noise_var_diag, full_states_train=None,
                     epochs=N_EPOCHS, lr=LR,
                     hidden=BNN_HIDDEN_DIM, n_mc=TRAJ_TRAIN_MC):
    from surrogates.traj_train import compute_standardizer
    std = (compute_standardizer(full_states_train, sys_mod)
           if full_states_train is not None else None)
    noise_std = float(torch.sqrt(noise_var_diag.mean()).item())
    g_net = BayesianNet(sys_mod.INPUT_DIM, hidden,
                         output_dim=sys_mod.OUTPUT_DIM, noise_std=noise_std,
                         standardizer=std)
    factory = sys_mod.make_hybrid_factory()
    C_obs = sys_mod.get_C()
    # Scale KL by an "effective" sample size that reflects the information
    # content of trajectory training. Each rolled-out trajectory contains
    # information at every integration step, not just at observation times,
    # so a small n_data (= obs count) over-weights the KL and prevents the
    # surrogate mean from moving. We approximate this with the number of
    # integration time points * number of trajectories.
    n_traj = x0_batch.shape[0]
    n_t = t_grid.shape[0]
    n_eff = int(n_traj * n_t)
    return train_traj_stochastic(
        g_net, factory, x0_batch, t_grid, y_obs, obs_indices,
        C_obs, noise_var_diag, epochs=epochs, lr=lr,
        kl_fn=lambda: g_net.kl(), n_data=n_eff,
        n_mc=n_mc, verbose_tag="BNN", kl_warmup_frac=KL_WARMUP_FRAC)


def predict_trajectory(g_net, sys_mod, x0, t_grid, n_mc=MC_SAMPLES):
    factory = sys_mod.make_hybrid_factory()
    return predict_traj_mc(g_net, factory, x0, t_grid, n_mc=n_mc)


def predict(model, grid):
    """Pointwise prediction of g(.) for force-surface plots."""
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            preds.append(model(grid).squeeze(-1))
    preds = torch.stack(preds)
    epi = preds.var(0)
    ale = torch.zeros_like(epi)
    return preds.mean(0), epi, ale

