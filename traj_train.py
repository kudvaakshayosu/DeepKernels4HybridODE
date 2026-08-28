# -*- coding: utf-8 -*-
"""
surrogates/traj_train.py — Integrator-in-the-loop training utilities.

This module provides the shared infrastructure for training a data-driven
component g(.) of a hybrid ODE by maximizing the likelihood of the *observed*
(partially measured, noisy) trajectories. Unlike supervised regression on
ground-truth residuals, the surrogate here only sees:

   1. The known initial conditions x0 of each training trajectory.
   2. Noisy observations y_t = C x_t + v_t at sparse times t_k.
   3. The known mechanistic component f(.) and the observation operator C.

Training proceeds by rolling out the hybrid ODE
       dx/dt = f_hybrid(x; g_theta)
forward from x0, comparing C @ x_t to y_t at observed times, and back-
propagating the Gaussian log-likelihood through the integrator to update
the surrogate parameters.

For methods with explicit posteriors (BNN, VGP, DKL), the rollout is repeated
TRAJ_TRAIN_MC times per gradient step, each time drawing a fresh sample from
the posterior, and the loss is averaged. KL terms are added on top.
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
from config import ODE_SOLVER


# ─── Hybrid dynamics wrappers ────────────────────────────────────────────────

# Soft bound on dx/dt magnitude: scale_softclamp(z, K) = K * tanh(z/K) keeps
# the derivative within [-K, K] while remaining smooth and differentiable.
# This is critical for trajectory training because a poorly initialized
# stochastic surrogate (BNN, GP sample) can otherwise push the integrator
# into NaN territory before training has a chance to recover.
DXDT_SOFTCLAMP = 50.0      # |dx/dt| <= ~50; well above realistic dynamics
G_SOFTCLAMP    = 20.0      # |g(x)| <= ~20

def soft_clamp(z, K):
    return K * torch.tanh(z / K)


class Standardizer(nn.Module):
    """Holds fixed input mean/std and an output (residual) scale as buffers,
    estimated ONCE from the measured training trajectories. A surrogate net
    standardizes its raw physical input x -> (x - mu)/sd before its first
    layer, and un-standardizes its raw output g_raw -> g_raw * g_scale on the
    way out. This keeps inputs ~zero-mean/unit-variance (so Tanh / RBF
    lengthscales operate in a sensible range) and lets the network learn a
    unit-scale target while the physical residual recovers its true magnitude.

    The statistics are computed from the *measured* training-trajectory states
    (not the test set, and not the integrator's transient excursions), and are
    held fixed for the entire run so train-time and predict-time scaling are
    identical."""
    def __init__(self, in_dim, x_mean=None, x_std=None, g_scale=1.0):
        super().__init__()
        if x_mean is None: x_mean = torch.zeros(in_dim)
        if x_std is None:  x_std = torch.ones(in_dim)
        self.register_buffer('x_mean', x_mean.float())
        self.register_buffer('x_std', x_std.float().clamp_min(1e-6))
        self.register_buffer('g_scale', torch.tensor(float(g_scale)))

    def encode(self, x):
        """Raw physical input -> standardized input."""
        return (x - self.x_mean) / self.x_std

    def decode_g(self, g_std):
        """Network output (unit scale) -> physical residual scale."""
        return g_std * self.g_scale


def compute_standardizer(full_states_train, sys_mod):
    """Estimate input mean/std and a residual output scale from the MEASURED
    training trajectories.

    full_states_train : list of [n_x, T] arrays (clean integrated states the
                        observations were sampled from).
    Inputs to the surrogate are the system states the surrogate is called on
    (sys_mod.input_selector picks them); we standardize over the measured
    state cloud. The residual scale is a robust estimate of |g*| magnitude
    over that cloud, so the network output starts at ~unit scale."""
    import numpy as np
    # Stack all measured state points: [N*T, n_x]
    pts = np.concatenate([s.T for s in full_states_train], axis=0)  # [M, STATE_DIM]
    # The surrogate input is whatever input_selector extracts; for all current
    # systems that is the full state (INPUT_DIM == STATE_DIM), but we guard:
    in_dim = sys_mod.INPUT_DIM
    pts_in = pts[:, :in_dim]
    x_mean = torch.tensor(pts_in.mean(0), dtype=torch.float32)
    x_std = torch.tensor(pts_in.std(0), dtype=torch.float32)

    # Residual scale: evaluate the TRUE residual over the measured cloud and
    # take its RMS as the characteristic magnitude. This uses only training
    # data (states that were actually measured) via the known true_residual,
    # which is available for these synthetic benchmarks; if it were not, the
    # state-derivative scale would be the fallback.
    try:
        g_true = sys_mod.true_residual(pts_in[:, 0], pts_in[:, 1])
        g_scale = float(np.sqrt(np.mean(np.square(g_true))) + 1e-8)
        if not np.isfinite(g_scale) or g_scale < 1e-6:
            g_scale = 1.0
    except Exception:
        g_scale = 1.0
    return Standardizer(in_dim, x_mean, x_std, g_scale)



class HybridParallel(nn.Module):
    """Wraps a parallel hybrid model: dx/dt = f(x) + B @ g(x).

    f(x)       : known mechanistic dynamics, returns shape [n_x]
    g(x)       : learned residual surrogate, returns shape [n_g]
    B          : selection matrix [n_x, n_g] that places the residual into the
                 appropriate state equations (e.g. [[0],[1]] for LV).
    """
    def __init__(self, f_known, g_net, B):
        super().__init__()
        self.f = f_known
        self.g = g_net
        self.register_buffer('B', B.float())

    def forward(self, t, x):
        # x: [batch, n_x] or [n_x]. Make it 2D for batching, then squeeze back.
        single = (x.dim() == 1)
        if single: x = x.unsqueeze(0)
        f_val = self.f(t, x)                          # [batch, n_x]
        g_val = self.g(x)                             # [batch, n_g]
        if g_val.dim() == 1: g_val = g_val.unsqueeze(-1)
        g_val = soft_clamp(g_val, G_SOFTCLAMP)        # bound residual
        contrib = g_val @ self.B.t()                  # [batch, n_x]
        out = f_val + contrib
        out = soft_clamp(out, DXDT_SOFTCLAMP)         # bound RHS
        return out.squeeze(0) if single else out


class HybridSeries(nn.Module):
    """Wraps a series hybrid model: dx/dt = f(x, g(x_input)).

    f_with_g(x, g_val) : known mechanistic dynamics that consumes g(x_input)
                         as a constitutive relation. Returns shape [n_x].
    g(x_input)         : learned constitutive surrogate.
    input_selector     : callable that takes the full state and returns the
                         subset of states that g(.) takes as input.
    """
    def __init__(self, f_with_g, g_net, input_selector):
        super().__init__()
        self.f_with_g = f_with_g
        self.g = g_net
        self.input_selector = input_selector

    def forward(self, t, x):
        single = (x.dim() == 1)
        if single: x = x.unsqueeze(0)
        g_input = self.input_selector(x)              # [batch, n_g_in]
        g_val = self.g(g_input)                       # [batch, n_g] or [batch]
        if g_val.dim() == 1: g_val = g_val.unsqueeze(-1)
        g_val = soft_clamp(g_val, G_SOFTCLAMP)
        out = self.f_with_g(t, x, g_val)              # [batch, n_x]
        out = soft_clamp(out, DXDT_SOFTCLAMP)
        return out.squeeze(0) if single else out


# ─── Trajectory rollout ──────────────────────────────────────────────────────

def rollout(hybrid_dyn, x0, t_grid, solver=None, max_deriv=1e3):
    """Integrate the hybrid dynamics from x0 over t_grid.

    Args:
        hybrid_dyn: HybridParallel or HybridSeries module
        x0: initial state, shape [batch, n_x] or [n_x]
        t_grid: 1D tensor of times, shape [T]
        solver: torchdiffeq solver name
        max_deriv: clip dx/dt magnitude to this value to prevent integrator blowup
                    during early training when surrogates may produce extreme values
    Returns:
        trajectory of shape [T, batch, n_x] (or [T, n_x] if x0 was 1D)
    """
    solver = solver or ODE_SOLVER

    if max_deriv is not None:
        # Wrap dynamics with a soft clamp (preserves gradients via tanh)
        class _Clamped(nn.Module):
            def __init__(self, dyn): super().__init__(); self.dyn = dyn
            def forward(self, t, x):
                d = self.dyn(t, x)
                return max_deriv * torch.tanh(d / max_deriv)
        hybrid_dyn = _Clamped(hybrid_dyn)

    options = {}
    if solver == 'rk4':
        dt = float(t_grid[1] - t_grid[0]) / 4.0
        options = {'step_size': dt}
    return odeint(hybrid_dyn, x0, t_grid, method=solver, options=options)


# ─── Gaussian negative log-likelihood on observations ────────────────────────

def gaussian_obs_nll(traj, y_obs, obs_indices, C, noise_var_diag):
    """Compute Gaussian NLL of observations under the rolled-out trajectory.

    Args:
        traj            : [T, batch, n_x]  predicted state trajectory
        y_obs           : [n_obs, batch, n_y]  noisy observations
        obs_indices     : 1D array of length n_obs, indices into the T-axis
        C               : [n_y, n_x] observation matrix
        noise_var_diag  : [n_y] per-channel noise variance
    Returns:
        scalar NLL (mean over obs times, batch, channels)
    """
    pred_full = traj[obs_indices]                       # [n_obs, batch, n_x]
    pred_obs = pred_full @ C.t()                        # [n_obs, batch, n_y]
    sq = (pred_obs - y_obs) ** 2                        # [n_obs, batch, n_y]
    nv = noise_var_diag.view(1, 1, -1).clamp_min(1e-8)
    nll = 0.5 * (sq / nv + torch.log(2 * np.pi * nv))
    return nll.mean()


# ─── Generic trajectory-NLL training loop for deterministic surrogates ───────

def train_traj_deterministic(g_net, hybrid_factory, x0s, t_grid, y_obs_list,
                              obs_indices, C, noise_var_diag,
                              epochs, lr, l2=0.0, verbose_tag="model",
                              extra_loss_fn=None, grad_clip=5.0):
    """Train a deterministic g(.) by integrator-in-the-loop NLL.

    Args:
        g_net           : nn.Module mapping inputs to residual / constitutive value
        hybrid_factory  : callable(g_net) -> HybridParallel/HybridSeries instance
        x0s             : [N, n_x] training initial conditions
        t_grid          : [T] integration time grid
        y_obs_list      : [n_obs, N, n_y] observations at observation indices
        obs_indices     : indices into t_grid where observations are recorded
        C               : observation matrix
        noise_var_diag  : per-channel noise variance
        epochs, lr      : training hyperparameters
        l2              : optional L2 regularization
        extra_loss_fn   : optional callable(g_net) returning extra loss term
                          (e.g. KL for BNN; weight prior for ensembles)
    Returns:
        (g_net, losses)
    """
    hybrid = hybrid_factory(g_net)
    opt = torch.optim.Adam(g_net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    losses = []; t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        traj = rollout(hybrid, x0s, t_grid)
        nll = gaussian_obs_nll(traj, y_obs_list, obs_indices, C, noise_var_diag)
        loss = nll
        if l2 > 0:
            loss = loss + l2 * sum(p.pow(2).sum() for p in g_net.parameters())
        if extra_loss_fn is not None:
            loss = loss + extra_loss_fn(g_net)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(g_net.parameters(), grad_clip)
        opt.step(); sch.step()
        losses.append(loss.item())
        if ep % 200 == 0:
            print(f"    {verbose_tag} ep {ep:4d} | loss {loss.item():.4f}")
    return g_net, time.time() - t0, losses


# ─── MC trajectory-ELBO training loop (for stochastic surrogates) ────────────

def train_traj_stochastic(g_net, hybrid_factory, x0s, t_grid, y_obs_list,
                           obs_indices, C, noise_var_diag,
                           epochs, lr, kl_fn, n_data,
                           n_mc=3, verbose_tag="model",
                           kl_warmup_frac=0.3, grad_clip=5.0,
                           extra_params=None):
    """Trajectory-level ELBO training for stochastic g(.).

    At each gradient step:
      1. Roll out the hybrid ODE n_mc times, each with a fresh draw of g's
         stochasticity (dropout mask, weight sample, GP inducing draw).
      2. Average the NLLs.
      3. Add KL term, β-annealed.

    Args:
        g_net           : nn.Module (must be stochastic in forward())
        kl_fn           : callable() -> scalar KL (closures over g_net)
        n_data          : effective dataset size for ELBO scaling
        extra_params    : optional list of additional parameters to optimize
                          (e.g. likelihood noise for GP/DKL)
    """
    hybrid = hybrid_factory(g_net)
    params = list(g_net.parameters())
    if extra_params is not None:
        params = params + list(extra_params)
    opt = torch.optim.Adam(params, lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    warmup_eps = int(epochs * kl_warmup_frac)
    losses = []; t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        beta = min(1.0, ep / max(warmup_eps, 1))
        # MC estimate of expected NLL
        total_nll = 0.0
        for _ in range(n_mc):
            traj = rollout(hybrid, x0s, t_grid)
            total_nll = total_nll + gaussian_obs_nll(
                traj, y_obs_list, obs_indices, C, noise_var_diag)
        total_nll = total_nll / n_mc
        kl = kl_fn() / n_data
        loss = total_nll + beta * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step(); sch.step()
        losses.append(loss.item())
        if ep % 200 == 0:
            print(f"    {verbose_tag} ep {ep:4d} | loss {loss.item():.4f} | "
                  f"β={beta:.2f} | KL/N={kl.item():.4f}")
    return g_net, time.time() - t0, losses


# ─── Predictive trajectory rollout (for evaluation) ──────────────────────────

def predict_traj_mean(g_net_or_callable, hybrid_factory, x0, t_grid):
    """Single deterministic rollout. Returns [T, n_x] trajectory."""
    hybrid = hybrid_factory(g_net_or_callable)
    with torch.no_grad():
        if x0.dim() == 1:
            return rollout(hybrid, x0, t_grid)
        return rollout(hybrid, x0, t_grid)


def predict_traj_mc(g_net, hybrid_factory, x0, t_grid, n_mc=200,
                     stochastic_setup=None):
    """MC rollout for stochastic surrogates. Returns [n_mc, T, n_x]."""
    if stochastic_setup is not None: stochastic_setup(g_net)
    hybrid = hybrid_factory(g_net)
    trajs = []
    with torch.no_grad():
        for _ in range(n_mc):
            trajs.append(rollout(hybrid, x0, t_grid))
    return torch.stack(trajs, 0)
