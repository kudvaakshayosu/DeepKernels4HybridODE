# -*- coding: utf-8 -*-
"""
surrogates/dkl.py — Deep Kernel Learning with integrator-in-the-loop training.

NN feature extractor + sparse variational GP (GPyTorch) in feature space.
Joint training of NN weights, inducing points (in feature space), kernel
hyperparameters, and variational parameters via the trajectory ELBO.

Key implementation choices:
    * Feature extractor learns at a slower rate than the GP variational
      parameters (10x slower). Without this, the NN tends to collapse the
      feature representation into something the GP prior can match trivially
      ("posterior collapse"), and the GP contributes nothing.
    * Inducing points are initialized in feature space by passing the
      training x0 inputs through the (untrained) feature extractor.
    * Kernel uses a ScaleKernel(RBFKernel(ard_num_dims=feat_dim)).
"""
import torch
import torch.nn as nn
import time
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (CholeskyVariationalDistribution,
                                    VariationalStrategy)
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal

from config import (HIDDEN_DIM, DKL_FEATURE_DIM, N_INDUCING, LR_GP,
                    N_EPOCHS, MC_SAMPLES, TRAJ_TRAIN_MC)
from surrogates.traj_train import rollout, gaussian_obs_nll


class DKLFeat(nn.Module):
    """Neural feature extractor h_w: x -> z for the deep kernel.

    Standardizes the raw physical input with a fixed Standardizer (so the
    first layer sees ~zero-mean/unit-variance inputs) and uses a smooth,
    non-saturating GELU activation instead of Tanh. Tanh saturates on
    unscaled inputs (vanishing gradients) and squashes the feature space into
    a bounded box that makes the inducing geometry awkward; GELU on
    standardized inputs keeps gradients alive and lets the feature
    coordinates spread out naturally."""
    def __init__(self, in_dim, hidden=64, out_dim=6, standardizer=None):
        super().__init__()
        self.std = standardizer
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim))

    def forward(self, x):
        if self.std is not None:
            x = self.std.encode(x)
        return self.net(x)


class FeatureGP(ApproximateGP):
    """ApproximateGP that lives in the feature space produced by phi(x; W)."""
    def __init__(self, inducing_points, feat_dim):
        var_dist = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0))
        var_strategy = VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True)
        super().__init__(var_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=feat_dim))

    def forward(self, z):
        return MultivariateNormal(self.mean_module(z), self.covar_module(z))


class DKLWrapper(nn.Module):
    """phi(x; W) → SVGP. Used as g(.) inside the hybrid ODE. The SVGP output
    is on a standardized (~unit) scale; we map it back to the physical
    residual scale via the standardizer before returning."""
    def __init__(self, feat, svgp, standardizer=None):
        super().__init__()
        self.feat = feat
        self.svgp = svgp
        self.std = standardizer

    def forward(self, x):
        z = self.feat(x)
        dist = self.svgp(z)
        out = dist.rsample() if self.training else dist.mean
        out = out.unsqueeze(-1) if out.dim() == 1 else out
        if self.std is not None:
            out = self.std.decode_g(out)
        return out


def _init_inducing_dkl(x0_batch, sys_mod, feat, n_inducing):
    """Inducing locations IN FEATURE SPACE.

    Take the training initial conditions, push them through the (untrained)
    feature extractor, and select n_inducing rows. If fewer ICs than
    inducing points are available, replicate with small jitter.
    """
    base_x = x0_batch[:, :sys_mod.INPUT_DIM]
    with torch.no_grad():
        base_z = feat(base_x)
    if base_z.shape[0] >= n_inducing:
        idx = torch.randperm(base_z.shape[0])[:n_inducing]
        Z = base_z[idx].clone()
    else:
        reps = (n_inducing + base_z.shape[0] - 1) // base_z.shape[0]
        Z = base_z.repeat(reps, 1)[:n_inducing].clone()
        Z = Z + 0.1 * torch.randn_like(Z)
    return Z

def train_trajectory(sys_mod, x0_batch, y_obs, t_grid, obs_indices,
                     noise_var_diag, full_states_train=None,
                     epochs=N_EPOCHS, lr=LR_GP,
                     hidden=HIDDEN_DIM, feat_dim=DKL_FEATURE_DIM,
                     n_inducing=N_INDUCING, n_mc=TRAJ_TRAIN_MC):

    from surrogates.traj_train import compute_standardizer
    std = (compute_standardizer(full_states_train, sys_mod)
           if full_states_train is not None else None)
    feat = DKLFeat(sys_mod.INPUT_DIM, hidden, feat_dim, standardizer=std)
    Z = _init_inducing_dkl(x0_batch, sys_mod, feat, n_inducing)
    svgp = FeatureGP(Z, feat_dim)
    wrapper = DKLWrapper(feat, svgp, standardizer=std)
    factory = sys_mod.make_hybrid_factory()
    hybrid = factory(wrapper)
    C_obs = sys_mod.get_C()
    n_traj = x0_batch.shape[0]; n_t = t_grid.shape[0]
    n_data = int(n_traj * n_t * 10)

    # Three parameter groups: slow NN, slow inducing pts, normal-speed GP
    inducing_params = [svgp.variational_strategy.inducing_points]
    gp_params = [p for n, p in svgp.named_parameters()
                 if 'inducing_points' not in n]
    opt = torch.optim.Adam([
        {'params': feat.parameters(), 'lr': lr * 0.1},
        {'params': inducing_params, 'lr': lr * 0.01},
        {'params': gp_params, 'lr': lr},
    ])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    warmup_eps = int(epochs * 0.3)
    feat.train(); svgp.train(); wrapper.train()
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
                print(f"    DKL ep {ep:4d} | loss {loss.item():.4f} | "
                      f"β={beta:.2f} | KL/N={kl.item():.4f}")
    return (svgp, feat, wrapper), time.time() - t0, losses


def predict_trajectory(model_tuple, sys_mod, x0, t_grid, n_mc=MC_SAMPLES):
    svgp, feat, wrapper = model_tuple
    feat.train(); svgp.train(); wrapper.train()
    factory = sys_mod.make_hybrid_factory()
    hybrid = factory(wrapper)
    trajs = []
    with torch.no_grad():
        for _ in range(n_mc):
            trajs.append(rollout(hybrid, x0, t_grid))
    return torch.stack(trajs, 0)


def predict(model_tuple, grid):
    """Pointwise prediction in feature space for surface plots. The SVGP
    operates on the standardized residual scale; map mean and variance back
    to the physical scale (variance scales by g_scale^2)."""
    svgp, feat, wrapper = model_tuple
    feat.eval(); svgp.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        z = feat(grid)
        dist = svgp(z)
        mean = dist.mean
        var = dist.variance
    std = getattr(wrapper, 'std', None)
    if std is not None:
        gs = std.g_scale
        mean = mean * gs
        var = var * gs * gs
    ale = torch.zeros_like(var)
    return mean, var, ale

