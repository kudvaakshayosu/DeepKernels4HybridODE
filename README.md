# [Paper Title]

**Akshay Kudva, Parth Shah, Taehyun Kim, Hangjun Cho, Joseph Sang-Il Kwon**
*Computers & Chemical Engineeering Journal, 2026*

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)]([https://www.sciencedirect.com/science/article/pii/S0098135426003881?via%3Dihub])


This is the official code for the paper *[Paper Title]*. It reproduces every figure and table in the paper, including the appendix sensitivity studies.

<p align="center">
  <img src="figures/[overview_figure].png" width="750" alt="Method overview">
</p>

---

## Overview

Hybrid ODE models combine known mechanistic structure, such as mass balances, with a data-driven **closure** that learns the terms we cannot write down, such as reaction kinetics. When these models are used for design, control or optimization, the closure must report how confident it is, not only a point prediction.

This repository benchmarks four closure surrogates. Each is trained **with the ODE integrator in the loop**, so it learns from state measurements rather than from direct observations of the unknown kinetics.

| Surrogate | Description | Predictive uncertainty |
|---|---|---|
| **DNN** | Multilayer perceptron | Point estimate |
| **BNN** | Bayesian neural network | Monte Carlo over weight samples |
| **VGP** | Sparse variational Gaussian process | Closed-form moments |
| **DKL** | Deep kernel learning: neural feature extractor + sparse variational GP | Closed-form moments |

The surrogates are compared on two case studies:

| Case | States | Evaluation domain |
|---|---|---|
| **CSTR** | $(C_A, C_B)$ | $[0, 1.1] \times [0, 1.2]$ mol/L |
| **Bioreactor** | $(X, S)$ | $[0, 2.5] \times [0, 3.0]$ g/L |

They are scored against the withheld true kinetics on a fixed evaluation grid using:
- average negative log-likelihood (NLL)
- mean absolute error of the predictive mean
- percentage of the domain where the true kinetics fall inside the $k\sigma$ band
- trajectory-level coverage

---

## Installation

```bash
git clone https://github.com/kudvaakshayosu/DeepKernels4HybridODE.git
cd DeepKernels4HybridODE

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproducing the paper

| Paper result | Command | Notes |
|---|---|---|
| **Fig. 3:** effect of training trajectories $N$ | `python run_ablation.py` | Run once per case (see below) |
| **Fig. 3:** four-panel version | `python plot_ablation_4panel.py results/ablation_<case>_full/checkpoint/ablation_results.pkl` | Reads saved results |
| **Fig. 4:** uncertainty-band coverage and violations | `python plot_spatial_coverage.py` | Needs saved models from Fig. 3 |
| **Fig. 5:** recovered kinetics (mean and std. dev.) | `python plot_kinetic_surfaces.py` | Needs saved models from Fig. 3 |
| **Appendix:** training convergence | `python plot_loss_curves.py` | Reads saved results |
| **Appendix:** MC samples × DKL feature dimension | `python sensitivity_mc_featdim.py` | Trains from scratch |
| **Appendix:** measurements per trajectory | `python sensitivity_samples_per_traj.py` | Trains from scratch |
| **Appendix:** computational cost breakdown | `python sensitivity_timing.py` | Trains from scratch |

### Step 1: Run the ablation (Fig. 3)

Open `run_ablation.py`. In the `ABLATION_CONFIG` block at the top, set `"case"` to `"cstr"` or `"bioreactor"`, then run:

```bash
python run_ablation.py
```

Repeat for the other case. The defaults match the paper:

| Setting | Value |
|---|---|
| `n_values` | `[1, 3, 5, 10, 20]` |
| `epochs` | `1000` |
| `n_repeats` | `3` |
| `k_sigma` | `3.0` |
| `grid_n` | `60` |

Results are written to `results/ablation_<case>_full/`:

```
ablation_avg_nll_vs_N_<s>.png                  average grid NLL vs N
ablation_constraint_satisfaction_vs_N_<s>.png  % coverage area vs N
ablation_lines_<s>.png, ablation_heatmap_<s>.png, ablation_uq_metrics_<s>.png
checkpoint/ablation_results.pkl                raw per-N, per-repeat metrics
checkpoint/models/N<N>_rep<r>/                 trained weights and metadata
```

To redraw the figures from a saved `ablation_results.pkl` without retraining, set `"plot_only": True`.

### Step 2: State-space figures (Figs. 4 and 5)

These scripts reload the models saved in Step 1, which requires `save_models: True`. They use the same evaluation grid and the same `k_sigma` as Fig. 3. In each script's `PLOT_CONFIG` block, set:

- `case`
- `n_train`
- `rep`: a single repeat index, or `None` to average predictions over all repeats
- `n_mc`: number of Monte Carlo weight samples for the BNN (ignored by the GP-based models, whose moments are closed-form)

Figures are saved as PNG and PDF in `results/ablation_<case>_full/figures/`.

### Step 3: Sensitivity studies (Appendix)

Each script has a `STUDY_CONFIG` block and writes a figure, a CSV and a `.pkl` of the raw per-repeat values to `results/sensitivity_<case>/`.

1. **MC samples × feature dimension.** Sweeps the number of Monte Carlo rollouts per gradient step, $M_S$, against the DKL latent dimension, $n_z$. The baseline cell ($M_S = 3$, $n_z = 6$) exactly reproduces the corresponding DKL fit from the main ablation, which is a useful cross-check.
2. **Measurements per trajectory.** Holds $N$ fixed and varies the number of measurement times per trajectory, $K$. This separates temporal sparsity from state-space coverage.
3. **Cost breakdown.** Splits DKL training time into four parts:
   - GP posterior, including the Cholesky factorization
   - feature extractor
   - ODE integration
   - backward pass

   A small `epochs` value (default 100) is enough here, because the split stabilizes long before convergence.

---

## Reproducibility notes

- **Fixed evaluation grid.** The evaluation domain is set explicitly in `config.py::GRID_BOUNDS` and matches Eq. (14) of the paper. The scripts do not infer it from the data and do not pad it (`GRID_PAD = 0.0`), so all of them score models on the same domain.
- **Hyperparameters.** `HIDDEN_DIM = 16` and `BNN_HIDDEN_DIM = 16`, as reported in Appendix A.
- **Seeding.** The code re-seeds before every model fit, so a model's initialization depends only on its repeat index.
- **Training length matters.** Grid NLL depends strongly on the number of epochs. For a single DKL fit on the CSTR with $N = 5$:

  | Epochs | 25 | 75 | 1000 |
  |---|---|---|---|
  | Grid NLL | +1.084 | +0.009 | −0.507 |

  Keep `epochs = 1000` for any reported result. Shorter runs are only for checking that a script runs.
- **Loss curves are not comparable across models.** The DNN minimizes a measurement NLL, while the probabilistic closures minimize a negative ELBO. Compare curves across $N$ and repeats for one model, not between models.

---

## Repository structure

```
├── config.py                       hyperparameters, case definitions, GRID_BOUNDS
├── run_ablation.py                 main experiment (Fig. 3)
├── plot_ablation_4panel.py         four-panel Fig. 3
├── plot_spatial_coverage.py        Fig. 4
├── plot_kinetic_surfaces.py        Fig. 5
├── plot_loss_curves.py             convergence diagnostics
├── sensitivity_mc_featdim.py       appendix study 1
├── sensitivity_samples_per_traj.py appendix study 2
├── sensitivity_timing.py           appendix study 3
├── sensitivity_utils.py            shared helpers for the sensitivity studies
├── backfill_grid_metrics.py        recompute grid metrics for saved results
├── evaluate.py                     trajectory-level metrics
├── plotting.py                     figure routines used by the ablation
├── figure_utils.py                 checkpoint loaders and grid reconstruction
├── surrogates/                     DNN, BNN, VGP, DKL closures and integrator-in-the-loop training
├── systems/                        CSTR, bioreactor and Lotka–Volterra ODEs with true kinetics
└── figures/                        figures used in this README
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{Kudva2026,
  title = {Probabilistic hybrid modeling in data-scarce regime via integration of first-principles ODEs with variational deep kernel learning},
  ISSN = {0098-1354},
  url = {http://dx.doi.org/10.1016/j.compchemeng.2026.109936},
  DOI = {10.1016/j.compchemeng.2026.109936},
  journal = {Computers \& Chemical Engineering},
  publisher = {Elsevier BV},
  author = {Kudva,  Akshay and Shah,  Parth and Kim,  Taehyun and Cho,  Hangjun and Kwon,  Joseph Sang-Il},
  year = {2026},
  month = Oct,
  pages = {109936}
}
```

## License

Released under the [MIT License](LICENSE). Replace this with your chosen license.

## Contact

For questions, open an [issue](https://github.com/kudvaakshayosu/DeepKernels4HybridODE/issues) or email **[your email]**.
