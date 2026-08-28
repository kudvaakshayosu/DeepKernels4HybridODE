# -*- coding: utf-8 -*-
"""
surrogates/vgp.py — Sparse Variational GP with integrator-in-the-loop training.

Implemented with GPyTorch primitives:
    * `gpytorch.models.ApproximateGP` for the model class
    * `VariationalStrategy` with `CholeskyVariationalDistribution` for the
      whitened inducing-point variational posterior
    * `ScaleKernel(RBFKernel(ard_num_dims=...))` for the ARD-RBF kernel
    * `VariationalELBO` exposes a closed-form KL divergence; we use it
      only for the KL component since our likelihood lives at the
      ROLLOUT level, not at the GP-output level.

Per gradient step we draw one MC sample from the variational posterior
of the GP, evaluate the conditional GP mean at each integration step
inside the hybrid ODE, and average the trajectory NLL across
`TRAJ_TRAIN_MC` such draws. The KL term comes directly from
`gpytorch.variational.CholeskyVariationalDistribution`.
"""
import torch
import torch.nn as nn
import numpy as np
import time
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (CholeskyVariationalDistribution,
                                    VariationalStrategy)
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import GammaPrior

from config import N_INDUCING, LR_GP, N_EPOCHS, MC_SAMPLES, TRAJ_TRAIN_MC
from surrogates.traj_train import rollout, gaussian_obs_nll


class SVGP(ApproximateGP):
    """GPyTorch ApproximateGP with whitened CholeskyVariationalDistribution.

    `whiten=True` (the default) uses the standard whitened
    parameterization u = L_K η with η ~ q(η); this is numerically
    stable and what we want for ODE training.
    """
    def __init__(self, inducing_points, input_dim):
        var_dist = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0))
        var_strategy = VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True)
        super().__init__(var_strategy)
        #self.mean_module = ConstantMean()
        self.mean_module = LinearMean(input_size=input_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=input_dim))
        #self.covar_module = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=input_dim,lengthscale_prior=GammaPrior(3.0, 2.0)),outputscale_prior=GammaPrior(2.0, 0.15))

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class GPWrapper(nn.Module):
    """Evaluate the GP inside the integrator. Inputs are standardized before
    the kernel sees them (the inducing points live in standardized space too),
    and the output is mapped back to the physical residual scale."""
    def __init__(self, svgp, standardizer=None):
        super().__init__()
        self.svgp = svgp
        self.std = standardizer

    def forward(self, x):
        if self.std is not None:
            x = self.std.encode(x)
        dist = self.svgp(x)
        if self.training:
            out = dist.rsample()
        else:
            out = dist.mean
        out = out.unsqueeze(-1) if out.dim() == 1 else out
        if self.std is not None:
            out = self.std.decode_g(out)
        return out


def _init_inducing(x0_batch, sys_mod, n_inducing, standardizer=None):
    """Pick inducing inputs in the (standardized) input space of g(.).

    Initial conditions seed the inducing points; if a standardizer is given,
    they are mapped into standardized coordinates so they live in the same
    space the kernel operates in.
    """
    base = x0_batch[:, :sys_mod.INPUT_DIM]
    if standardizer is not None:
        base = standardizer.encode(base)
    if base.shape[0] >= n_inducing:
        idx = torch.randperm(base.shape[0])[:n_inducing]
        Z = base[idx].clone()
    else:
        reps = (n_inducing + base.shape[0] - 1) // base.shape[0]
        Z = base.repeat(reps, 1)[:n_inducing].clone()
        Z = Z + 0.1 * torch.randn_like(Z)
    return Z


def train_trajectory(sys_mod, x0_batch, y_obs, t_grid, obs_indices,
                     noise_var_diag, full_states_train=None,
                     epochs=N_EPOCHS, lr=LR_GP,
                     n_inducing=N_INDUCING, n_mc=TRAJ_TRAIN_MC):

    from surrogates.traj_train import compute_standardizer
    std = (compute_standardizer(full_states_train, sys_mod)
           if full_states_train is not None else None)
    Z = _init_inducing(x0_batch, sys_mod, n_inducing, standardizer=std)
    svgp = SVGP(Z, sys_mod.INPUT_DIM)
    wrapper = GPWrapper(svgp, standardizer=std)
    factory = sys_mod.make_hybrid_factory()
    hybrid = factory(wrapper)
    C_obs = sys_mod.get_C()
    n_traj = x0_batch.shape[0]; n_t = t_grid.shape[0]
    n_data = int(n_traj * n_t * 10)             # scaled effective sample size

    # Two parameter groups: slow inducing-point motion, normal speed for the rest
    inducing_params = [svgp.variational_strategy.inducing_points]
    other_params = [p for n, p in svgp.named_parameters()
                    if 'inducing_points' not in n]
    opt = torch.optim.Adam([
        {'params': other_params, 'lr': lr},
        {'params': inducing_params, 'lr': lr},  #* 0.01
    ])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    warmup_eps = int(epochs * 0.3)
    svgp.train(); wrapper.train()
    losses = []; t0 = time.time()
    with gpytorch.settings.cholesky_jitter(float_value=1e-4, double_value=1e-6):
        for ep in range(epochs):
            opt.zero_grad()
            beta = min(1.0, ep / max(warmup_eps, 1))
            total_nll = 0.0
            for _ in range(n_mc):
                traj = rollout(hybrid, x0_batch, t_grid)
                total_nll = total_nll + gaussian_obs_nll(
                    traj, y_obs, obs_indices, C_obs, noise_var_diag)
            total_nll = total_nll / n_mc
            # KL term provided by the variational strategy in whitened form
            kl = svgp.variational_strategy.kl_divergence().sum() / n_data
            loss = total_nll + beta * kl
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    VGP ep {ep}: NaN/Inf loss; skipping step")
                opt.zero_grad(); continue
            loss.backward()
            bad_grad = any(p.grad is not None and torch.isnan(p.grad).any()
                            for p in svgp.parameters())
            if bad_grad:
                opt.zero_grad(); print(f"    VGP ep {ep}: NaN gradient; skipping step")
                continue
            torch.nn.utils.clip_grad_norm_(svgp.parameters(), 5.0)
            opt.step(); sch.step()
            losses.append(loss.item())
            if ep % 200 == 0:
                print(f"    VGP ep {ep:4d} | loss {loss.item():.4f} | "
                      f"β={beta:.2f} | KL/N={kl.item():.4f}")
    return (svgp, wrapper), time.time() - t0, losses


def predict_trajectory(model_tuple, sys_mod, x0, t_grid, n_mc=MC_SAMPLES):
    """MC rollout: keep variational sampling active to capture epistemic uncertainty."""
    svgp, wrapper = model_tuple
    # Keep .train() so rsample() is used inside forward; gpytorch handles the rest
    svgp.train(); wrapper.train()
    factory = sys_mod.make_hybrid_factory()
    hybrid = factory(wrapper)
    trajs = []
    with torch.no_grad():
        for _ in range(n_mc):
            trajs.append(rollout(hybrid, x0, t_grid))
    return torch.stack(trajs, 0)


def predict(model_tuple, grid):
    """Pointwise prediction of g(.) for force-surface plots.

    The GP kernel operates in standardized input space and on a standardized
    target, so we encode the query grid and decode the returned moments back
    to the physical residual scale (variance by g_scale^2)."""
    svgp, wrapper = model_tuple
    svgp.eval()
    std = getattr(wrapper, 'std', None)
    g = std.encode(grid) if std is not None else grid
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        dist = svgp(g)
        mean = dist.mean
        var = dist.variance
    if std is not None:
        gs = std.g_scale
        mean = mean * gs
        var = var * gs * gs
    ale = torch.zeros_like(var)
    return mean, var, ale

