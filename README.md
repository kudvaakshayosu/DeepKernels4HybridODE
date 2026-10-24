# Hybrid-ODE UQ Benchmark — ablation only (fixed evaluation grid)

Reduced package containing **only** what is needed to reproduce the
N_TRAIN ablation (Fig. 3 of the manuscript). Plotting scripts for figures
not in the paper have been removed.

## What changed relative to the original code

1. **The evaluation grid is now fixed and explicit.** It is defined in
   `config.py::GRID_BOUNDS` and matches Eq. (14):

       CSTR       (C_A, C_B) in [0, 1.1] x [0, 1.2]  mol/L
       Bioreactor (X,   S)   in [0, 2.5] x [0, 3.0]  g/L

   Previously the box was inferred at runtime as the bounding rectangle of
   the trajectories plus a padding of 0.5, so it varied between scripts and
   did not match the domain stated in the paper.

2. **Padding is disabled** (`GRID_PAD = 0.0`). The bounds above are used
   verbatim.

3. **Hyperparameters restored to the values reported in the paper:**
   `HIDDEN_DIM = 16`, `BNN_HIDDEN_DIM = 16` (the working copy had drifted
   to 10, which did not match the saved checkpoints or Appendix A).

4. `ABLATION_CONFIG` defaults set to the paper run: `n_values =
   [1,3,5,10,20]`, `epochs = 1000`, `n_repeats = 3`, `k_sigma = 3.0`,
   `grid_n = 60`.

## Running

    pip install -r requirements.txt
    python run_ablation.py

Set `"case"` in `ABLATION_CONFIG` at the top of `run_ablation.py` to
`"cstr"` or `"bioreactor"` and run once for each.

Outputs land in `results/ablation_<case>_full/`:

    ablation_avg_nll_vs_N_<s>.png                  average grid NLL vs N
    ablation_constraint_satisfaction_vs_N_<s>.png  % coverage area vs N
    ablation_lines_<s>.png / ablation_heatmap_<s>.png / ablation_uq_metrics_<s>.png
    checkpoint/ablation_results.pkl                raw per-N, per-repeat metrics
    checkpoint/models/N<N>_rep<r>/                 saved weights + meta

Set `"plot_only": True` to redraw from a saved `ablation_results.pkl`
without retraining.

## Paper figures

After the sweep has run (with `save_models: True`), reproduce the two
state-space figures from the saved checkpoints:

    python plot_spatial_coverage.py    # Fig. 4  (band coverage + violations)
    python plot_kinetic_surfaces.py    # Fig. 5  (recovered means + std devs)

Each has a `PLOT_CONFIG` block at the top: set `case`, `n_train`, and `rep`.
Setting `"rep": None` averages the predictions over every available repeat
instead of plotting a single one. `n_mc` controls the number of Monte Carlo
weight samples used for the BNN predictive (the GP-based methods return
closed-form moments, so it does not affect them).

Both scripts reload the models trained by `run_ablation.py` and rebuild the
grid from `config.GRID_BOUNDS`, so they use the same models, the same domain
and the same `k_sigma` as the Fig. 3 metrics. Figures are written to
`results/ablation_<case>_full/figures/` as both PNG and PDF.

Fig. 3 itself is produced directly by `run_ablation.py`. Two further scripts
read its saved results:

    python plot_ablation_4panel.py results/ablation_cstr_full/checkpoint/ablation_results.pkl
    python plot_loss_curves.py

`plot_ablation_4panel.py` is the journal-style Fig. 3, extended from three
panels to four: average grid NLL, **mean absolute error of the predictive mean
over the grid**, % coverage area, and trajectory coverage. Axis labels adapt to
the case recorded in the pickle. It takes the pickle path as an argument.

`plot_loss_curves.py` draws the training objective against epoch for every
model, with a min-max band over repeats, and prints how much each loss still
drifts over the final 10% of training — the convergence diagnostic. It has a
`PLOT_CONFIG` block and reads the same pickle. Note the objective differs
between models (the DNN minimizes the measurement NLL, the probabilistic
closures a negative ELBO), so curves are comparable across N and repeats for a
given model rather than between models.

## Sensitivity studies (appendix)

Three additional studies, all DKL-only except where noted, all scored on the
same fixed grid so they are comparable to the main results:

    python sensitivity_mc_featdim.py       # (1) M_S x feature dimension
    python sensitivity_samples_per_traj.py # (2) K at fixed N
    python sensitivity_timing.py           # (3) cost breakdown

**(1) Monte Carlo samples x DKL feature dimension.** Sweeps the MC rollouts per
gradient step, `M_S`, against the latent feature dimension `n_z`, reporting
average grid NLL and the mean absolute error of the predictive mean against the
withheld kinetics. Outputs two heat maps plus a CSV of the raw per-repeat
values. These runs train from scratch and do not use the ablation checkpoints.

**(2) Measurements per trajectory at fixed N.** Holds `N` fixed and varies `K`,
the number of measurement times per trajectory. This separates measurement
sparsity from state-space coverage: varying `N` changes which regions the
trajectories visit, whereas varying `K` changes only the temporal sampling
density. `OBS_INDICES` is fixed at system import, so the study rebinds it per
`K`; with `vary_obs_seed: True` the measurement times are also re-drawn each
repeat, which matters at small `K` where a single draw is not representative.

**(3) Cost breakdown.** Splits DKL training time into the GP posterior
(including the Cholesky factorization of `K_ZZ`), the feature extractor, the ODE
integrator, and the backward pass, across several `(M_S, n_z)` settings.
GPyTorch overrides `Module.__call__`, so PyTorch forward hooks never fire on the
SVGP; the closure's forward is therefore instrumented directly. The GP and
feature times occur inside the rollout, so the integrator figure is the rollout
time net of both. Use a small `epochs` here (the default is 100) — the split is
stable well before convergence.

Each study has a `STUDY_CONFIG` block at the top and writes to
`results/sensitivity_<case>/`: a PNG/PDF figure, a CSV of the raw per-repeat
values, and a `.pkl` of the same. With `save_models: True` (the default in
studies 1 and 2) every trained DKL closure is also checkpointed under
`results/sensitivity_<case>/checkpoint/`, in the same layout `run_ablation.py`
uses, so any configuration can be reloaded and inspected without retraining.

All three studies compute the average grid NLL with the same expression as the
main ablation (Eq. 15), on the same fixed grid, so the numbers are directly
comparable to Fig. 3. `run_ablation.py` re-seeds before every model fit, so a
model's initialization depends only on the repeat index and not on its position
in `TRAIN_ORDER`. Consequently the baseline cell of study 1 (`M_S = 3`,
`n_z = 6`) reproduces the ablation's DKL fit exactly at the same `N` and repeat
— a useful cross-check that the two entry points agree.

### A note on epochs

Grid NLL is strongly dependent on training length. For the CSTR at `N = 5`,
a single DKL fit gives roughly:

    epochs      25        75       1000
    NLL     +1.084    +0.009     -0.507

Short runs therefore produce large positive NLL that reflects undertraining
rather than a problem with the metric or the model. Keep `epochs` at 1000 for
any result that will be reported; the reduced values are only useful for
smoke-testing that a script runs.

## Files

    config.py                   hyperparameters, case definitions, GRID_BOUNDS
    run_ablation.py             the ablation sweep (entry point) -> Fig. 3
    plot_spatial_coverage.py    Fig. 4
    plot_kinetic_surfaces.py    Fig. 5
    figure_utils.py             checkpoint loaders + fixed-grid rebuild
    evaluate.py                 trajectory-level metrics
    plotting.py                 figure routines used by the ablation
    surrogates/                 mlp, bnn, vgp, dkl + integrator-in-the-loop training
    systems/                    cstr, bioreactor, lotka_volterra ODEs + true kinetics
