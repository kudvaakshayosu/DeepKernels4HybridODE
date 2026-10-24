# -*- coding: utf-8 -*-
"""plotting.py — Publication-quality figures for the DKL benchmark.

Style choices
-------------
- Journal-style sizes: 18pt axis labels, 16pt ticks, 16pt legend (bold titles).
- Tol's BRIGHT palette — colorblind-safe (deuteranopia, protanopia, tritanopia),
  prints distinguishably in greyscale. Reference: https://personal.sron.nl/~pault/
- Larger markers (markersize=9), thicker lines (linewidth=2.4).
- Wider figures so axis labels never get clipped on the journal page.

Time/cost normalization
-----------------------
Several plots accept a `normalize=False` flag. When True, training-time and
inference-cost values are divided by the MLP reference at the same x-value
and labels are switched to "× MLP". When False (default), absolute units
(seconds, ms) are used.

Regenerating plots only
-----------------------
None of the plotting routines train; they all consume in-memory metric/result
dicts that main.py produces. To re-plot without retraining, set PLOT_ONLY=True
in main.py — that path calls load_results() and then re-invokes every plot
function with the same arguments. So edits here take effect on next run.
"""
import os
import numpy as np
import torch
import matplotlib, matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
# Global style — applied as soon as the module is imported.
# ═════════════════════════════════════════════════════════════════════════════
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 16,
    'axes.labelsize': 18, 'axes.titlesize': 18,
    'axes.titleweight': 'bold', 'axes.labelweight': 'normal',
    'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'legend.fontsize': 14, 'legend.title_fontsize': 15,
    'legend.frameon': True, 'legend.framealpha': 0.95,
    'legend.edgecolor': '0.3',
    'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.18,
    'axes.grid': True, 'grid.alpha': 0.22, 'grid.linestyle': '--',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.4, 'lines.markersize': 9,
})


# ═════════════════════════════════════════════════════════════════════════════
# Colorblind-safe palette (Tol bright) — distinguishable in greyscale too.
# Reference: Paul Tol's "Colour Schemes" technical note, SRON/AS-MGS-005, 2021.
# ═════════════════════════════════════════════════════════════════════════════
MC = {'mlp': '#BBBBBB',    # neutral grey — deterministic baseline
      'bnn': '#EE6677',    # red          — parametric Bayesian
      'vgp': '#228833',    # green        — non-parametric Bayesian
      'dkl': '#4477AA'}    # blue         — main contribution

MD = {'mlp': 'MLP (det.)',
      'bnn': 'LL-BNN',
      'vgp': 'Variational GP',
      'dkl': 'Deep Kernel Learning'}

C_TRUTH     = '#000000'
C_MEAN      = '#1f77b4'
C_EPI       = '#88CCEE'
C_TOTAL     = '#FFAABB'
C_SCATTER   = '#8B0000'
C_OBSBORDER = '#8B0000'


def _save(fig, d, name):
    fig.savefig(os.path.join(d, f"{name}.pdf"))
    fig.savefig(os.path.join(d, f"{name}.png"))
    print(f"  Saved: {name}"); plt.close(fig)


# Helper used by both training-time and inference-cost plots so the
# normalization logic is identical.
def _maybe_normalize(values, names, normalize, ref_name='mlp'):
    """If normalize is True, divide each value by the value at ref_name.
    Returns (values, unit_string)."""
    if not normalize:
        return values, None
    if ref_name not in names:
        return values, None
    ref = values[names.index(ref_name)]
    if ref == 0 or not np.isfinite(ref):
        return values, None
    return [v / ref for v in values], f'× {MD.get(ref_name, ref_name)}'


# ═════════════════════════════════════════════════════════════════════════════
# 1. Force-surface plots (2-D input g(.))
# ═════════════════════════════════════════════════════════════════════════════
def plot_force_surfaces(trained, pred_fns, names, true_fn, obs_x, obs_y,
                          gx, gy, output_dir,
                          xlabel='$x_1$', ylabel='$x_2$',
                          truth_title='Truth: $g^*(x_1, x_2)$'):
    """Top row: TRUTH | predicted residual surfaces for each method, all on
    a shared color scale so methods are visually comparable to ground truth.
    Bottom row: epistemic σ for each method on a shared scale (the truth
    column for σ is blank since the truth has no uncertainty)."""
    n = len(names)
    # (n+1) columns: column 0 is truth/legend; columns 1..n are methods.
    # constrained_layout handles per-row colorbars more reliably than
    # tight_layout when colorbar labels are long.
    fig, axes = plt.subplots(2, n + 1,
                               figsize=(4.4 * (n + 1) + 2.4, 8.8),
                               squeeze=False,
                               constrained_layout=True)
    XX, YY = np.meshgrid(gx, gy)
    grid = torch.tensor(np.stack([XX.ravel(), YY.ravel()], 1), dtype=torch.float32)
    true_grid = true_fn(XX, YY)

    means, epis = {}, {}
    for name in names:
        with torch.no_grad():
            mu, ep, _ = pred_fns[name](trained[name], grid)
        means[name] = mu.cpu().numpy().reshape(XX.shape)
        epis[name]  = np.sqrt(np.maximum(ep.cpu().numpy(), 0)).reshape(XX.shape)

    # SHARED TOP-ROW SCALE: symmetric about zero, covers truth AND all
    # predicted means, so a method that systematically over-shoots is
    # immediately visible against the truth panel.
    all_top = np.concatenate([true_grid.ravel()] + [m.ravel() for m in means.values()])
    top_abs = float(np.abs(all_top).max())
    top_lv  = np.linspace(-top_abs, top_abs, 30)

    # SHARED BOTTOM-ROW SCALE for epistemic σ. If σ spans more than 2
    # decades across methods (overconfident-BNN vs uncertainty-inflated-DKL),
    # switch to log scale so all methods are simultaneously readable;
    # otherwise use linear.
    sigma_min = min(e.min() for e in epis.values())
    sigma_max = max(e.max() for e in epis.values())
    use_log = (sigma_max > 0) and (sigma_min > 0) and (sigma_max / sigma_min > 100)
    if use_log:
        bot_lv = np.geomspace(max(sigma_min, sigma_max * 1e-4), sigma_max, 30)
        bot_norm = matplotlib.colors.LogNorm(vmin=bot_lv[0], vmax=bot_lv[-1])
        sigma_label = 'Epistemic $\\sigma(x)$ (log scale)'
    else:
        bot_lv = np.linspace(sigma_min, sigma_max, 30)
        bot_norm = None
        sigma_label = 'Epistemic $\\sigma(x)$'

    # ROW 0 COLUMN 0: ground truth
    ax = axes[0, 0]
    im_t = ax.contourf(XX, YY, true_grid, levels=top_lv, cmap='RdBu_r')
    ax.scatter(obs_x, obs_y, c=C_SCATTER, s=22, edgecolors='black',
                linewidths=0.6, zorder=3, label='Training points')
    ax.set_title(truth_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Predicted $g(x)$\n' + ylabel, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)

    # ROW 0 COLUMNS 1..n: predicted means
    for i, name in enumerate(names):
        ax = axes[0, i + 1]
        ax.contourf(XX, YY, means[name], levels=top_lv, cmap='RdBu_r')
        ax.scatter(obs_x, obs_y, c=C_SCATTER, s=22, edgecolors='black',
                    linewidths=0.6, zorder=3)
        ax.set_title(MD.get(name, name))
        ax.set_xlabel(xlabel)

    # ROW 1 COLUMN 0: blank (truth has no uncertainty), but mark the row.
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5,
        '(no uncertainty\nfor ground truth)',
        ha='center', va='center', fontsize=14, style='italic', color='#666',
        transform=axes[1, 0].transAxes)

    # ROW 1 COLUMNS 1..n: epistemic σ
    im_b = None
    for i, name in enumerate(names):
        ax = axes[1, i + 1]
        if use_log:
            im_b = ax.contourf(XX, YY, np.maximum(epis[name], bot_lv[0]),
                                levels=bot_lv, cmap='viridis', norm=bot_norm)
        else:
            im_b = ax.contourf(XX, YY, epis[name], levels=bot_lv, cmap='viridis')
        ax.scatter(obs_x, obs_y, c=C_SCATTER, s=22, edgecolors='black',
                    linewidths=0.6, zorder=3)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(sigma_label.replace(' (log scale)', '') + '\n' + ylabel,
                           fontweight='bold')

    # ONE colorbar per row, anchored to the right edge.
    cb1 = fig.colorbar(im_t, ax=axes[0, :].tolist(), fraction=0.030, pad=0.02,
                        shrink=0.92)
    cb1.set_label('Predicted $g(x)$ (truth + methods share scale)',
                    fontsize=14)
    cb2 = fig.colorbar(im_b, ax=axes[1, :].tolist(), fraction=0.030, pad=0.02,
                        shrink=0.92)
    cb2.set_label(sigma_label + ' (methods share scale)', fontsize=14)
    fig.suptitle('Recovered surface and epistemic uncertainty '
                  '(all methods on shared scales; left column = ground truth)',
                  fontsize=18)
    _save(fig, output_dir, 'force_surfaces')


# ═════════════════════════════════════════════════════════════════════════════
# 2. Rollout: trajectory prediction for one test point
# ═════════════════════════════════════════════════════════════════════════════
def plot_rollout_all_states(trained, predict_traj_fns, names, sys_mod, x0_test,
                              t_grid, full_test, noise_var_diag, C_obs,
                              observed_idx, state_names, output_dir,
                              traj_idx=0, n_mc=100):
    """Per-method row × per-state column grid. Columns share the y-axis
    across method rows so the reader can visually compare interval widths
    method-to-method for the same state."""
    n_methods, n_states = len(names), sys_mod.STATE_DIM
    fig, axes = plt.subplots(n_methods, n_states,
                              figsize=(6.0 * n_states, 3.8 * n_methods),
                              squeeze=False,
                              sharex=True,
                              sharey='col',                 # <-- shared y per state column
                              constrained_layout=True)
    tg = t_grid.cpu().numpy()
    truth = torch.tensor(full_test[traj_idx], dtype=torch.float32).cpu().numpy()
    x0_single = x0_test[traj_idx:traj_idx + 1]
    observed_set = set(observed_idx)

    aleatoric = np.zeros(n_states)
    for k, idx in enumerate(observed_idx):
        aleatoric[idx] = float(noise_var_diag[k])

    for row, name in enumerate(names):
        with torch.no_grad():
            trajs = predict_traj_fns[name](trained[name], sys_mod, x0_single, t_grid, n_mc=n_mc)
        trajs = trajs.squeeze(2).cpu().numpy()
        pm   = trajs.mean(0)
        sd_e = np.sqrt(np.maximum(trajs.var(0), 0))
        sd_t = np.sqrt(np.maximum(trajs.var(0) + aleatoric[None, :], 0))

        for col in range(n_states):
            ax = axes[row, col]
            # ax.fill_between(tg, pm[:, col] - 2 * sd_t[:, col], pm[:, col] + 2 * sd_t[:, col],
            #                  color=C_TOTAL, alpha=0.30, label=r'Total $\pm 2\sigma$')
            ax.fill_between(tg, pm[:, col] - 2 * sd_e[:, col], pm[:, col] + 2 * sd_e[:, col],
                             color=C_EPI, alpha=0.55, label=r'Epistemic $\pm 2\sigma$')
            ax.plot(tg, truth[col], color=C_TRUTH, lw=2.6, label='True')
            ax.plot(tg, pm[:, col], color=C_MEAN, lw=2.0, ls='--', label='Predicted mean')
            if col in observed_set:
                for sp in ax.spines.values():
                    sp.set_edgecolor(C_OBSBORDER); sp.set_linewidth(2.4); sp.set_visible(True)
            if row == 0:
                tag = '(observed)' if col in observed_set else '(latent)'
                ax.set_title(f"{state_names[col]} {tag}")
            if col == 0:
                # Generous labelpad so the method name doesn't overlap tick labels.
                ax.set_ylabel(MD.get(name, name), fontweight='bold', labelpad=24)
            if row == n_methods - 1: ax.set_xlabel('Time')
            if row == 0 and col == n_states - 1:
                ax.legend(loc='best', fontsize=12, framealpha=0.92)

    fig.suptitle(f'Rolled-out predictions vs truth — trajectory {traj_idx} '
                  '(maroon border = observed channel; columns share y-axis)',
                  fontsize=18)
    _save(fig, output_dir, f'rollout_traj{traj_idx}')


# ═════════════════════════════════════════════════════════════════════════════
# 3. Summary metric plots
# ═════════════════════════════════════════════════════════════════════════════
def plot_rmse_box(metrics, names, d):
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(names) + 4), 6))
    bp = ax.boxplot([metrics['rmse'][n] for n in names], patch_artist=True,
                     widths=0.55, medianprops={'color': 'black', 'linewidth': 2.4})
    for patch, n in zip(bp['boxes'], names):
        patch.set_facecolor(MC.get(n, '#888')); patch.set_alpha(0.85)
        patch.set_edgecolor('black'); patch.set_linewidth(1.1)
    ax.set_xticklabels([MD.get(n, n) for n in names], rotation=18, ha='right')
    ax.set_ylabel('Trajectory RMSE (observed channels)')
    ax.set_title('Trajectory RMSE on test set')
    plt.tight_layout(); _save(fig, d, 'rmse_box')


def plot_training_time(results, names, d, normalize=False):
    """Bar plot of training time. If normalize=True, expressed as multiples of
    the MLP reference (so curves are hardware-invariant)."""
    raw = [results[n]['time'] for n in names]
    values, unit = _maybe_normalize(raw, names, normalize, ref_name='mlp')
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(names) + 4), 6))
    bars = ax.bar([MD.get(n, n) for n in names], values,
                   color=[MC.get(n, '#888') for n in names],
                   alpha=0.88, edgecolor='black', linewidth=1.0)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2,
                 v * 1.01 + (max(values) * 0.005 if v >= 0 else 0),
                 (f'{v:.2f}{unit}' if unit else f'{v:.1f}s'),
                 ha='center', va='bottom', fontsize=14)
    ax.set_ylabel(f'Training time ({unit})' if unit else 'Training time (s)')
    title_suffix = ' — normalized to MLP' if normalize else ''
    ax.set_title(f'Wall-clock training cost{title_suffix}')
    plt.xticks(rotation=18, ha='right')
    if normalize:
        ax.axhline(1.0, color='gray', ls=':', lw=1.5, alpha=0.6)
    plt.tight_layout(); _save(fig, d, 'training_time' + ('_normalized' if normalize else ''))


def plot_loss_curves(results, names, d):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for n in names:
        ax.plot(results[n]['losses'], color=MC.get(n, '#888'),
                 label=MD.get(n, n), lw=2.2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Training loss')
    ax.set_title('Trajectory-NLL / ELBO loss curves')
    ax.legend(loc='best', framealpha=0.92)
    ax.set_yscale('symlog', linthresh=1.0)
    plt.tight_layout(); _save(fig, d, 'loss_curves')


def plot_calibration(metrics, names, d):
    """Scatter of (interval width, empirical coverage) per method. Method
    identity is shown by color via the legend; we deliberately do NOT add
    per-point text labels, which pile up and overlap when several methods land
    at similar coordinates (the legend already disambiguates)."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for n in names:
        c = float(np.mean(metrics['coverage'][n]))
        w = float(np.mean(metrics['width'][n]))
        ax.scatter(w, c, s=340, color=MC.get(n, '#888'), edgecolors='black',
                   linewidth=1.2, alpha=0.92, label=MD.get(n, n), zorder=3)

    ax.axhline(0.95, color='gray', ls='--', lw=1.4, label='Nominal 95%')
    ax.set_xlabel(r'Mean predictive interval width ($\pm 2\sigma$)')
    ax.set_ylabel('Empirical 95% coverage')
    ax.set_title('Calibration: coverage vs width')
    ax.set_ylim(-0.05, 1.10)
    ax.legend(loc='lower right', fontsize=13, framealpha=0.95)
    plt.tight_layout(); _save(fig, d, 'calibration')


def plot_summary_table(metrics, results, names, d, title="Summary"):
    """Tabular summary panel — kept on absolute units so it conveys real
    compute realism. (Normalized counterpart is plot_training_time(normalize=True).)"""
    fig, ax = plt.subplots(figsize=(max(14, 2.4 * len(names) + 5), 3.8))
    rows = []
    has_params = all('n_params' in results[n] for n in names)
    for n in names:
        r = np.array(metrics['rmse'][n]); c = np.array(metrics['coverage'][n])
        w = np.array(metrics['width'][n])
        row = [f"{r.mean():.4f} ± {r.std():.4f}",
               f"{c.mean():.3f} ± {c.std():.3f}",
               f"{w.mean():.4f}",
               f"{results[n]['time']:.1f} s"]
        if has_params:
            row.append(f"{results[n]['n_params']}")
        rows.append(row)
    col_labels = ['Trajectory RMSE', 'Coverage (target 0.95)',
                  '95% interval width', 'Train time']
    if has_params:
        col_labels.append('# params')
    tbl = ax.table(cellText=rows,
                    colLabels=col_labels,
                    rowLabels=[MD.get(n, n) for n in names],
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(13); tbl.scale(1.1, 2.0)
    ncol = len(col_labels)
    for i in range(ncol):
        tbl[0, i].set_facecolor('#dddddd'); tbl[0, i].set_text_props(weight='bold')
    ax.axis('off')
    ax.set_title(title, pad=18, fontsize=16)
    plt.tight_layout()
    _save(fig, d, 'summary_table')


def plot_inference_cost(ic, names, d, normalize=False):
    """Latency vs grid size. If normalize=True, divides every method's latency
    by the MLP latency at the same grid size, so curves are hardware-invariant
    and the MLP baseline becomes a flat line at 1.0."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    if normalize and 'mlp' in ic:
        ref = {s: ic['mlp'][s]['median_ms'] for s in ic['mlp']}
    else:
        ref = None
    for n in names:
        t = ic.get(n, {})
        if not t: continue
        sizes = sorted(t.keys())
        if ref is not None:
            ys = [t[s]['median_ms'] / ref[s] if ref.get(s, 0) else float('nan')
                  for s in sizes]
        else:
            ys = [t[s]['median_ms'] for s in sizes]
        ax.plot(sizes, ys, '-o', color=MC.get(n, '#888'),
                 label=MD.get(n, n), lw=2.4, markersize=10)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Number of evaluation points')
    ax.set_ylabel(f'Latency (× {MD["mlp"]})' if normalize else 'Latency (ms, median)')
    if normalize:
        ax.axhline(1.0, color='gray', ls=':', lw=1.5, alpha=0.6)
        ax.set_title('Pointwise inference cost — normalized to MLP')
    else:
        ax.set_title('Pointwise inference cost')
    ax.legend(loc='best', framealpha=0.92)
    plt.tight_layout(); _save(fig, d, 'inference_cost' + ('_normalized' if normalize else ''))


# ═════════════════════════════════════════════════════════════════════════════
# 4. Ablation: sweep N_TRAIN
# ═════════════════════════════════════════════════════════════════════════════
def plot_ablation_heatmap(all_results, n_values, output_dir, system_name=""):
    metrics_show = [('rmse', 'RMSE (lower better)'),
                    ('coverage', 'Coverage (target 0.95)'),
                    ('width', 'Width (lower better)')]
    fig, axes = plt.subplots(len(metrics_show), 1,
                              figsize=(1.7 * len(n_values) + 6, 4.5 * len(metrics_show)))
    if len(metrics_show) == 1: axes = [axes]
    method_names = list(all_results[n_values[0]]['rmse'].keys())
    for k, (mk, mlabel) in enumerate(metrics_show):
        mat = np.zeros((len(method_names), len(n_values)))
        for j, N in enumerate(n_values):
            for i, n in enumerate(method_names):
                mat[i, j] = np.mean(all_results[N][mk][n])
        ax = axes[k]
        im = ax.imshow(mat, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(n_values))); ax.set_xticklabels(n_values)
        ax.set_yticks(range(len(method_names)))
        ax.set_yticklabels([MD.get(n, n) for n in method_names])
        ax.set_xlabel('$N_{\\mathrm{train}}$')
        ax.set_title(f'{system_name}: {mlabel}', fontsize=17)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha='center', va='center',
                         color='white' if mat[i, j] > mat.mean() else 'black',
                         fontsize=13)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); _save(fig, output_dir, f'ablation_heatmap_{system_name}')


def plot_ablation_lines(all_results, n_values, output_dir, system_name=""):
    """Line plot of metric vs N_TRAIN, one curve per method."""
    metrics_show = [('rmse', 'RMSE', True),
                    ('coverage', 'Coverage (target=0.95)', False),
                    ('width', '95% Interval width', True)]
    method_names = list(all_results[n_values[0]]['rmse'].keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (mk, label, log_y) in zip(axes, metrics_show):
        for n in method_names:
            ys = [np.mean(all_results[N][mk][n]) for N in n_values]
            es = [np.std(all_results[N][mk][n]) /
                  np.sqrt(max(len(all_results[N][mk][n]), 1))
                  for N in n_values]
            ax.errorbar(n_values, ys, yerr=es, color=MC.get(n, '#888'),
                         label=MD.get(n, n), lw=2.4, marker='o',
                         markersize=10, capsize=5)
        ax.set_xlabel('# Training trajectories ($N_{\\mathrm{train}}$)')
        ax.set_ylabel(label); ax.set_title(label)
        if mk == 'coverage':
            ax.axhline(0.95, color='gray', ls='--', lw=1.4)
        if log_y: ax.set_yscale('log')
        ax.set_xticks(n_values)
    axes[0].legend(loc='best', fontsize=14, framealpha=0.92)
    fig.suptitle(f'{system_name}: data-efficiency ablation', fontsize=19, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, output_dir, f'ablation_lines_{system_name}')


# ═════════════════════════════════════════════════════════════════════════════
# 5. INTERPRETABILITY: epistemic-vs-distance plots
# ═════════════════════════════════════════════════════════════════════════════
# These figures answer the "WHY is THIS region uncertain?" question. The
# canonical Bayesian-surrogate diagnostic is: epistemic uncertainty should
# grow as the query point moves away from training data. If it does, the
# model's uncertainty has the expected geometric structure; if it doesn't,
# something has gone wrong (e.g., over-confident extrapolation, or a
# stationary kernel that can't model the data density).

def _grid_with_training_overlay(full_train, sys_mod, grid_size=70,
                                  shrink_fraction=0.05):
    """Build a 2-D evaluation grid covering the training-trajectory support
    (with a small inward shrink), and return both the grid and the training
    points used as overlay/distance reference."""
    pts = np.concatenate([s.T for s in full_train], axis=0)[:, :sys_mod.INPUT_DIM]
    x1_min, x1_max = pts[:, 0].min(), pts[:, 0].max()
    x2_min, x2_max = pts[:, 1].min(), pts[:, 1].max()
    span1, span2   = x1_max - x1_min, x2_max - x2_min
    x1_lo, x1_hi = x1_min + shrink_fraction * span1, x1_max - shrink_fraction * span1
    x2_lo, x2_hi = x2_min + shrink_fraction * span2, x2_max - shrink_fraction * span2
    g1 = np.linspace(x1_lo, x1_hi, grid_size)
    g2 = np.linspace(x2_lo, x2_hi, grid_size)
    G1, G2 = np.meshgrid(g1, g2)
    return (G1, G2, pts, (x1_lo, x1_hi, x2_lo, x2_hi))


def _epistemic_sigma_on_grid(model, pred_fn, G1, G2):
    """Returns sqrt(epistemic variance) reshaped to G1.shape."""
    grid = torch.tensor(np.stack([G1.ravel(), G2.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        mu, ep, _ = pred_fn(model, grid)
    return np.sqrt(np.maximum(ep.cpu().numpy(), 0)).reshape(G1.shape)


def plot_uncertainty_vs_distance(trained, pred_fns, names, sys_mod, full_train,
                                    output_dir, xlabel='$x_1$', ylabel='$x_2$',
                                    cbar_label='Epistemic $\\sigma$',
                                    grid_size=70):
    """Per-method epistemic-σ heatmap with training points + distance-to-data
    contours overlaid. The eye reads: where contours and σ agree → uncertainty
    is geometrically calibrated; where they diverge → something is off.

    Three rows of panels per method are not needed; one row per method is
    enough because each panel carries both the σ heatmap AND the distance
    contour overlay."""
    probabilistic = [n for n in names if n != 'mlp']
    if not probabilistic:
        print("  Skipping uncertainty_vs_distance (no probabilistic methods)")
        return
    G1, G2, pts, (x1_lo, x1_hi, x2_lo, x2_hi) = _grid_with_training_overlay(
        full_train, sys_mod, grid_size=grid_size)

    # Distance-to-nearest-training-point on the grid. Standardize coordinates
    # first so the distance is comparable across axes with different units.
    train_xy = pts.copy()
    mu = train_xy.mean(0); sd = train_xy.std(0); sd[sd < 1e-3] = 1.0
    train_z = (train_xy - mu) / sd
    grid_xy = np.stack([G1.ravel(), G2.ravel()], 1)
    grid_z  = (grid_xy - mu) / sd
    # min-distance from each grid point to any training point (in standardized coords)
    # Broadcast: [Gx*Gy, 1, 2] - [1, N, 2] -> [Gx*Gy, N, 2]
    dists = np.sqrt(((grid_z[:, None, :] - train_z[None, :, :]) ** 2).sum(-1))
    min_d = dists.min(1).reshape(G1.shape)

    n = len(probabilistic)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n + 1.4, 6.4), squeeze=False,
                               constrained_layout=True)

    # Common color scale across all probabilistic methods. Explicitly passing
    # the SAME `levels` array (not just vmin/vmax) is what actually forces
    # contourf to share the scale — otherwise each panel auto-bins its data.
    sigmas = {nm: _epistemic_sigma_on_grid(trained[nm], pred_fns[nm], G1, G2)
              for nm in probabilistic}
    sigma_min = min(s.min() for s in sigmas.values())
    sigma_max = max(s.max() for s in sigmas.values())
    # Use log spacing if σ spans >2 decades across methods (e.g. overconfident
    # LL-BNN at ~1e-3 vs inflated DKL at ~10).
    use_log = (sigma_max > 0) and (sigma_min > 0) and (sigma_max / sigma_min > 100)
    if use_log:
        levels = np.geomspace(max(sigma_min, sigma_max * 1e-4), sigma_max, 24)
        norm = matplotlib.colors.LogNorm(vmin=levels[0], vmax=levels[-1])
        cbar_label_full = cbar_label + ' (log scale, shared)'
    else:
        levels = np.linspace(sigma_min, sigma_max, 24)
        norm = None
        cbar_label_full = cbar_label + ' (shared scale)'

    im_last = None
    for col, name in enumerate(probabilistic):
        ax = axes[0, col]
        z = sigmas[name]
        if use_log:
            z = np.maximum(z, levels[0])
        im = ax.contourf(G1, G2, z, levels=levels, cmap='viridis', norm=norm)
        im_last = im
        # Distance-to-data contours (dashed, semi-transparent)
        cs = ax.contour(G1, G2, min_d, levels=6, colors='white',
                         linewidths=1.0, alpha=0.7, linestyles='--')
        ax.clabel(cs, inline=True, fontsize=10, fmt='%.2f')
        # Training trajectory points
        ax.scatter(pts[:, 0], pts[:, 1], s=10, c='black', alpha=0.5,
                    edgecolors='none', label='Training points')
        ax.set_xlim(x1_lo, x1_hi); ax.set_ylim(x2_lo, x2_hi)
        ax.set_xlabel(xlabel)
        if col == 0:
            ax.set_ylabel(ylabel)
            ax.legend(loc='lower right', fontsize=12)
        ax.set_title(MD.get(name, name))

    # ONE shared colorbar to the right of all panels (instead of one per panel)
    # so the reader can read all three heatmaps against the same scale.
    cb = fig.colorbar(im_last, ax=axes[0, :].tolist(), fraction=0.020, pad=0.02,
                       shrink=0.92)
    cb.set_label(cbar_label_full, fontsize=14)

    fig.suptitle('Where does each surrogate report high uncertainty?\n'
                  'Heatmap: epistemic $\\sigma(x)$ (shared scale).  '
                  'Dashed contours: standardized distance to nearest training point.',
                  fontsize=17)
    _save(fig, output_dir, 'uncertainty_vs_distance')


def plot_uncertainty_vs_distance_scatter(trained, pred_fns, names, sys_mod,
                                            full_train, output_dir,
                                            grid_size=50):
    """Scatter plot: x-axis = standardized distance to nearest training point,
    y-axis = epistemic σ at that point. One panel per probabilistic method.
    A well-behaved Bayesian model produces a monotone increasing point cloud
    in this plot. A flat cloud → uncertainty insensitive to data density
    (over- or under-confident extrapolation)."""
    probabilistic = [n for n in names if n != 'mlp']
    if not probabilistic:
        return
    G1, G2, pts, _ = _grid_with_training_overlay(full_train, sys_mod,
                                                    grid_size=grid_size)
    mu = pts.mean(0); sd = pts.std(0); sd[sd < 1e-3] = 1.0
    grid_z = (np.stack([G1.ravel(), G2.ravel()], 1) - mu) / sd
    train_z = (pts - mu) / sd
    min_d = np.sqrt(((grid_z[:, None, :] - train_z[None, :, :]) ** 2)
                     .sum(-1)).min(1)

    fig, axes = plt.subplots(1, len(probabilistic),
                              figsize=(5.8 * len(probabilistic), 6.0),
                              squeeze=False, sharex=True,
                              constrained_layout=True)

    # Determine global σ range so the y-axis can hint whether one method is
    # systematically more uncertain than another. We use a LOG y-axis when
    # the σ range spans >1.5 decades — this is the only way to keep an
    # overconfident BNN (σ ~ 1e-3) and an inflated DKL (σ ~ 10) both
    # readable in the same figure.
    all_sigmas = []
    for name in probabilistic:
        all_sigmas.append(
            _epistemic_sigma_on_grid(trained[name], pred_fns[name], G1, G2).ravel())
    sigma_glo_min = max(min(s.min() for s in all_sigmas), 1e-12)
    sigma_glo_max = max(s.max() for s in all_sigmas)
    use_log = (sigma_glo_max / sigma_glo_min) > 30

    for k, name in enumerate(probabilistic):
        ax = axes[0, k]
        sigma = all_sigmas[k]
        ax.scatter(min_d, sigma, s=14, alpha=0.45,
                    color=MC.get(name, '#888'), edgecolors='none')
        # Fit a smooth trend (running mean over distance bins).
        order = np.argsort(min_d)
        d_sorted, s_sorted = min_d[order], sigma[order]
        bin_edges = np.linspace(d_sorted.min(), d_sorted.max(), 12)
        bin_centers, bin_means = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (d_sorted >= lo) & (d_sorted < hi)
            if mask.sum() > 0:
                bin_centers.append(0.5 * (lo + hi))
                bin_means.append(s_sorted[mask].mean())
        ax.plot(bin_centers, bin_means, 'k-', lw=2.6,
                 label='Running mean')
        if use_log:
            ax.set_yscale('log')
            ax.set_ylim(sigma_glo_min * 0.7, sigma_glo_max * 1.4)
        else:
            ax.set_ylim(sigma_glo_min * 0.9, sigma_glo_max * 1.1)
        ax.set_xlabel('Std. distance to nearest training point')
        if k == 0:
            ax.set_ylabel('Epistemic $\\sigma(x)$' +
                            (' (log)' if use_log else ''))
        ax.set_title(MD.get(name, name))
        ax.legend(loc='upper left', fontsize=12)
    fig.suptitle('Does uncertainty grow with distance from data?\n'
                  '(monotone upward running mean = well-calibrated; '
                  'flat = uncertainty insensitive to data density)',
                  fontsize=16)
    _save(fig, output_dir, 'uncertainty_vs_distance_scatter')


def plot_sigma_vs_distance_quantified(trained, pred_fns, names, sys_mod,
                                       full_train, output_dir, grid_size=60,
                                       n_shells=6):
    """Quantitative test of the central claim: epistemic sigma should be LOW
    near the training data and grow with distance away from it.

    Three panels:
      (a) sigma vs standardized distance-to-nearest-training-point, all
          probabilistic methods overlaid as binned mean curves with a shaded
          inter-quartile band -- the clearest comparison of who tightens near
          data and widens away from it;
      (b) concentric-shell bars: the input domain is partitioned into shells
          of increasing distance from the data (the 'spheres' around the
          training set), and the mean sigma in each shell is shown per method.
          A well-behaved method has short bars in the inner shells and tall
          bars in the outer shells;
      (c) a single monotonicity score per method: the Spearman rank
          correlation between distance and sigma over the whole grid, in
          [-1, 1]. Higher = sigma more reliably increases with distance from
          data. This reduces 'which method's uncertainty is best spatially
          structured?' to one comparable number.

    Distances are computed in STANDARDIZED input coordinates (zero mean, unit
    variance per axis) so the 'spheres' are isotropic in the space the models
    actually operate on, rather than being dominated by whichever physical
    axis has the larger numerical range.
    """
    from scipy.stats import spearmanr

    probabilistic = [n for n in names if n != 'mlp']
    if not probabilistic:
        print("  Skipping sigma_vs_distance_quantified (no probabilistic methods)")
        return

    G1, G2, pts, _ = _grid_with_training_overlay(full_train, sys_mod,
                                                  grid_size=grid_size)
    mu = pts.mean(0); sd = pts.std(0); sd[sd < 1e-3] = 1.0
    grid_z = (np.stack([G1.ravel(), G2.ravel()], 1) - mu) / sd
    train_z = (pts - mu) / sd
    min_d = np.sqrt(((grid_z[:, None, :] - train_z[None, :, :]) ** 2)
                    .sum(-1)).min(1)                          # [Ngrid]

    sigmas = {nm: _epistemic_sigma_on_grid(trained[nm], pred_fns[nm], G1, G2).ravel()
              for nm in probabilistic}

    # Spearman monotonicity score (distance vs sigma) per method.
    scores = {}
    for nm in probabilistic:
        rho, _ = spearmanr(min_d, sigmas[nm])
        scores[nm] = 0.0 if np.isnan(rho) else float(rho)

    # Shell edges by distance quantiles, so each shell holds ~equal grid area.
    edges = np.quantile(min_d, np.linspace(0, 1, n_shells + 1))
    edges[-1] += 1e-9
    shell_centers = 0.5 * (edges[:-1] + edges[1:])
    shell_labels = [f"[{edges[i]:.1f},{edges[i+1]:.1f})" for i in range(n_shells)]

    use_log = False
    allvals = np.concatenate([s for s in sigmas.values()])
    pos = allvals[allvals > 0]
    if pos.size and (pos.max() / pos.min() > 30):
        use_log = True

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.2))
    fig.subplots_adjust(wspace=0.32, top=0.80, bottom=0.16, left=0.06, right=0.985)

    # ── Panel (a): overlaid binned-mean curves with IQR band ──
    axA = axes[0]
    nb = 14
    bin_edges = np.linspace(min_d.min(), min_d.max(), nb + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for nm in probabilistic:
        s = sigmas[nm]
        means, q1, q3 = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            m = (min_d >= lo) & (min_d < hi)
            if m.sum() > 0:
                means.append(s[m].mean())
                q1.append(np.percentile(s[m], 25))
                q3.append(np.percentile(s[m], 75))
            else:
                means.append(np.nan); q1.append(np.nan); q3.append(np.nan)
        means = np.array(means); q1 = np.array(q1); q3 = np.array(q3)
        col = MC.get(nm, '#888')
        axA.plot(centers, means, '-', color=col, lw=2.8, marker='o',
                 markersize=6, label=MD.get(nm, nm))
        axA.fill_between(centers, q1, q3, color=col, alpha=0.18)
    if use_log:
        axA.set_yscale('log')
    axA.set_xlabel('Standardized distance to nearest training point')
    axA.set_ylabel('Epistemic $\\sigma(x)$' + (' (log)' if use_log else ''))
    axA.set_title('(a) $\\sigma$ vs distance\n(mean line, IQR band)', fontsize=12)
    axA.legend(loc='upper left', fontsize=11, framealpha=0.95)

    # ── Panel (b): concentric-shell mean-σ bars, grouped by shell ──
    axB = axes[1]
    width = 0.8 / len(probabilistic)
    x = np.arange(n_shells)
    for j, nm in enumerate(probabilistic):
        s = sigmas[nm]
        shell_means = []
        for i in range(n_shells):
            m = (min_d >= edges[i]) & (min_d < edges[i + 1])
            shell_means.append(s[m].mean() if m.sum() > 0 else np.nan)
        axB.bar(x + j * width, shell_means, width=width,
                color=MC.get(nm, '#888'), alpha=0.9, label=MD.get(nm, nm),
                edgecolor='black', linewidth=0.5)
    if use_log:
        axB.set_yscale('log')
    axB.set_xticks(x + 0.4 - width / 2)
    axB.set_xticklabels(shell_labels, rotation=30, ha='right', fontsize=9)
    axB.set_xlabel('Distance shell (standardized)')
    axB.set_ylabel('Mean $\\sigma$ in shell' + (' (log)' if use_log else ''))
    axB.set_title('(b) Mean $\\sigma$ per shell\n(inner short, outer tall = good)', fontsize=12)
    axB.legend(loc='upper left', fontsize=11, framealpha=0.95)

    # ── Panel (c): Spearman monotonicity score per method ──
    axC = axes[2]
    order = sorted(probabilistic, key=lambda n: scores[n], reverse=True)
    vals = [scores[n] for n in order]
    cols = [MC.get(n, '#888') for n in order]
    bars = axC.bar(range(len(order)), vals, color=cols, alpha=0.9,
                   edgecolor='black', linewidth=0.8)
    for b, v in zip(bars, vals):
        axC.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.06),
                 f"{v:.2f}", ha='center', va='bottom' if v >= 0 else 'top',
                 fontsize=12, fontweight='bold')
    axC.axhline(0, color='black', lw=0.8)
    axC.set_xticks(range(len(order)))
    axC.set_xticklabels([MD.get(n, n) for n in order], rotation=20, ha='right')
    axC.set_ylim(min(0, min(vals) - 0.12), 1.05)
    axC.set_ylabel('Spearman $\\rho$(distance, $\\sigma$)')
    axC.set_title('(c) Monotonicity score\n(Spearman; higher = better)', fontsize=12)

    fig.suptitle('Is epistemic uncertainty low near data and high away from it?',
                 fontsize=15, y=0.99)
    _save(fig, output_dir, 'sigma_vs_distance_quantified')

    # Console summary so the ranking is logged, not just drawn.
    print("  Monotonicity score (Spearman rho of distance vs sigma):")
    for n in order:
        print(f"    {MD.get(n, n):<22} {scores[n]:+.3f}")
    return scores



# ═════════════════════════════════════════════════════════════════════════════
# 6. INTERPRETABILITY: rate-error decomposition on the surface
# ═════════════════════════════════════════════════════════════════════════════
def plot_rate_error_decomposition(trained, pred_fns, names, sys_mod, full_train,
                                     true_fn, output_dir,
                                     xlabel='$x_1$', ylabel='$x_2$',
                                     grid_size=60):
    """For each probabilistic method, three panels side-by-side:
      (a) absolute rate error |g_hat - g_true|
      (b) epistemic sigma
      (c) z-score = error / sigma  -- the ratio that tells you whether the
          uncertainty CORRECTLY explains the error.
    A well-calibrated method has z ≤ 2 almost everywhere; z ≫ 2 means
    confidently wrong; z ≪ 1 means uncertainty-inflated.

    Color scales are SHARED across method rows within each column, so the
    reader can compare the magnitudes across methods at a glance. Per
    column: a single colorbar on the right (not one per cell)."""
    probabilistic = [n for n in names if n != 'mlp']
    if not probabilistic: return
    G1, G2, pts, _ = _grid_with_training_overlay(
        full_train, sys_mod, grid_size=grid_size)
    true_grid = true_fn(G1, G2)

    # PASS 1: gather error and sigma for every method so we can set
    # column-wide color scales BEFORE plotting.
    n = len(probabilistic)
    grid_t = torch.tensor(np.stack([G1.ravel(), G2.ravel()], 1), dtype=torch.float32)
    err_all, sd_all, z_all = {}, {}, {}
    for name in probabilistic:
        with torch.no_grad():
            mu, ep, _ = pred_fns[name](trained[name], grid_t)
        mu_g = mu.cpu().numpy().reshape(G1.shape)
        sd_g = np.sqrt(np.maximum(ep.cpu().numpy(), 1e-12)).reshape(G1.shape)
        err_all[name] = np.abs(mu_g - true_grid)
        sd_all[name]  = sd_g
        z_all[name]   = err_all[name] / sd_g

    # Column-wide shared levels
    err_max = max(e.max() for e in err_all.values())
    err_lv  = np.linspace(0, err_max, 22)

    sd_min  = min(s.min() for s in sd_all.values())
    sd_max  = max(s.max() for s in sd_all.values())
    sd_use_log = (sd_max > 0) and (sd_min > 0) and (sd_max / sd_min > 100)
    if sd_use_log:
        sd_lv  = np.geomspace(max(sd_min, sd_max * 1e-4), sd_max, 22)
        sd_norm = matplotlib.colors.LogNorm(vmin=sd_lv[0], vmax=sd_lv[-1])
        sd_label = 'Epistemic $\\sigma$ (log, shared)'
    else:
        sd_lv  = np.linspace(sd_min, sd_max, 22)
        sd_norm = None
        sd_label = 'Epistemic $\\sigma$ (shared)'

    # Z-score scale: always [0, 4] capped; semantically meaningful threshold
    # is z=2, so we keep it identical for all methods.
    z_lv = np.linspace(0, 4, 22)

    fig, axes = plt.subplots(n, 3, figsize=(17.5, 4.8 * n), squeeze=False,
                               constrained_layout=True)

    im_err = im_sd = im_z = None
    for row, name in enumerate(probabilistic):
        # Column 0: |error|
        ax = axes[row, 0]
        im_err = ax.contourf(G1, G2, err_all[name], levels=err_lv, cmap='Reds')
        ax.scatter(pts[:, 0], pts[:, 1], s=8, c='black', alpha=0.4,
                    edgecolors='none')
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel, fontweight='bold')
        if row == 0:
            ax.set_title('$|\\hat{g}(x) - g^*(x)|$')
        ax.text(-0.32, 0.5, MD.get(name, name), transform=ax.transAxes,
                 rotation=90, va='center', ha='center',
                 fontsize=15, fontweight='bold')

        # Column 1: epistemic σ
        ax = axes[row, 1]
        z_sd = sd_all[name].copy()
        if sd_use_log:
            z_sd = np.maximum(z_sd, sd_lv[0])
        im_sd = ax.contourf(G1, G2, z_sd, levels=sd_lv, cmap='viridis',
                              norm=sd_norm)
        ax.scatter(pts[:, 0], pts[:, 1], s=8, c='black', alpha=0.4,
                    edgecolors='none')
        ax.set_xlabel(xlabel)
        if row == 0:
            ax.set_title('Epistemic $\\sigma(x)$')

        # Column 2: z-score
        ax = axes[row, 2]
        z_show = np.clip(z_all[name], 0, 4)
        im_z = ax.contourf(G1, G2, z_show, levels=z_lv, cmap='RdYlGn_r',
                            extend='max')
        cs = ax.contour(G1, G2, z_all[name], levels=[1, 2, 3], colors='black',
                         linewidths=1.2)
        ax.clabel(cs, fmt='%.0f', fontsize=11)
        ax.scatter(pts[:, 0], pts[:, 1], s=8, c='black', alpha=0.4,
                    edgecolors='none')
        ax.set_xlabel(xlabel)
        if row == 0:
            ax.set_title('$z(x) = |\\hat{g} - g^*| / \\sigma$')

    # ONE colorbar per column at the BOTTOM of each column. constrained_layout
    # handles bottom colorbars cleanly when they're attached to a multi-row
    # axes list; right colorbars in the same situation float into row gaps.
    cb_err = fig.colorbar(im_err, ax=axes[:, 0].tolist(), location='bottom',
                            fraction=0.06, pad=0.04, shrink=0.92, aspect=35)
    cb_err.set_label('Absolute error $|\\hat{g}-g^*|$ (shared)', fontsize=14)
    cb_sd = fig.colorbar(im_sd, ax=axes[:, 1].tolist(), location='bottom',
                          fraction=0.06, pad=0.04, shrink=0.92, aspect=35)
    cb_sd.set_label(sd_label, fontsize=14)
    cb_z = fig.colorbar(im_z, ax=axes[:, 2].tolist(), location='bottom',
                         fraction=0.06, pad=0.04, shrink=0.92, aspect=35)
    cb_z.set_label('$z$-score (capped at 4; threshold $z{=}2$)', fontsize=14)

    fig.suptitle('Rate-error decomposition — '
                  'does the uncertainty explain the error?\n'
                  '$z \\leq 2$ everywhere $\\Rightarrow$ well-calibrated.  '
                  '$z \\gg 2$ in some region $\\Rightarrow$ confidently wrong.  '
                  '(rows = methods; columns share color scale)',
                  fontsize=16)
    _save(fig, output_dir, 'rate_error_decomposition')


# ═════════════════════════════════════════════════════════════════════════════
# 7. METHOD COMPARISON OVERLAY (research-paper style)
# ═════════════════════════════════════════════════════════════════════════════
def plot_rollout_method_overlay(trained, predict_traj_fns, names, sys_mod,
                                  x0_test, t_grid, full_test, noise_var_diag,
                                  observed_idx, state_names, output_dir,
                                  traj_idx=0, n_mc=100):
    """All methods overlaid on the same axes per state, with each method's
    mean trace in its assigned color and the ±2σ band semi-transparent. This
    is the standard side-by-side comparison figure that papers use to show
    method differences at a glance — separate-row figures (above) are still
    saved for detailed inspection, this is the summary."""
    n_states = sys_mod.STATE_DIM
    fig, axes = plt.subplots(1, n_states,
                              figsize=(7.0 * n_states, 5.8),
                              squeeze=False,
                              constrained_layout=True)
    tg = t_grid.cpu().numpy()
    truth = torch.tensor(full_test[traj_idx], dtype=torch.float32).cpu().numpy()
    x0_single = x0_test[traj_idx:traj_idx + 1]
    observed_set = set(observed_idx)

    # Compute every method's trajectory ensemble once
    method_data = {}
    for name in names:
        with torch.no_grad():
            trajs = predict_traj_fns[name](trained[name], sys_mod, x0_single, t_grid, n_mc=n_mc)
        trajs = trajs.squeeze(2).cpu().numpy()
        method_data[name] = (trajs.mean(0),
                              np.sqrt(np.maximum(trajs.var(0), 0)))

    for col in range(n_states):
        ax = axes[0, col]
        # Truth first (background, so colored bands sit on top)
        ax.plot(tg, truth[col], color='black', lw=3.0, label='True',
                 zorder=5)
        # Each method: mean line + ±2σ band
        for name in names:
            pm, sd_e = method_data[name]
            ax.plot(tg, pm[:, col], color=MC.get(name, '#888'), lw=2.2,
                     ls='--', label=MD.get(name, name), zorder=4)
            # Show ±2σ band only for probabilistic methods (deterministic MLP
            # has sd_e=0 from MC=1 prediction)
            if name != 'mlp':
                ax.fill_between(tg, pm[:, col] - 2 * sd_e[:, col],
                                  pm[:, col] + 2 * sd_e[:, col],
                                  color=MC.get(name, '#888'), alpha=0.18,
                                  zorder=2)
        if col in observed_set:
            for sp in ax.spines.values():
                sp.set_edgecolor(C_OBSBORDER)
                sp.set_linewidth(2.4); sp.set_visible(True)
        tag = '(observed)' if col in observed_set else '(latent)'
        ax.set_title(f"{state_names[col]} {tag}")
        ax.set_xlabel('Time')
        if col == 0:
            ax.set_ylabel('State value')
            ax.legend(loc='best', fontsize=12, framealpha=0.94)
    fig.suptitle(f'All methods overlaid on the same trajectory — '
                  f'test trajectory #{traj_idx} (maroon = observed channel)',
                  fontsize=17)
    _save(fig, output_dir, f'rollout_overlay_traj{traj_idx}')


# ═════════════════════════════════════════════════════════════════════════════
# 8. MULTI-PANEL SUMMARY (research-paper style)
# ═════════════════════════════════════════════════════════════════════════════
def plot_metric_bar_comparison(metrics, names, d, title="Metric comparison"):
    """A 3-panel side-by-side bar plot of (RMSE, coverage, width) across
    methods, on a single figure. The standard "summary figure" in papers."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    metrics_show = [
        ('rmse',     'Trajectory RMSE (lower is better)',          True),
        ('coverage', 'Empirical 95% coverage',                     False),
        ('width',    '95% interval width (lower is tighter)',      True),
    ]
    for ax, (mk, label, log_y) in zip(axes, metrics_show):
        vals = [float(np.mean(metrics[mk][n])) for n in names]
        errs = [float(np.std(metrics[mk][n]) /
                       np.sqrt(max(len(metrics[mk][n]), 1))) for n in names]
        bars = ax.bar([MD.get(n, n) for n in names], vals,
                       yerr=errs, capsize=6, error_kw={'lw': 1.4},
                       color=[MC.get(n, '#888') for n in names],
                       edgecolor='black', linewidth=1.0, alpha=0.88)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                     v + 0.02 * max(vals),
                     f'{v:.3f}', ha='center', va='bottom', fontsize=12)
        if mk == 'coverage':
            ax.axhline(0.95, color='gray', ls='--', lw=1.4)
            ax.set_ylim(0, 1.10)
        if log_y:
            ax.set_yscale('log')
        ax.set_title(label, fontsize=15)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([MD.get(n, n) for n in names], rotation=18,
                            ha='right')
    fig.suptitle(title, fontsize=18)
    _save(fig, d, 'metric_comparison')
