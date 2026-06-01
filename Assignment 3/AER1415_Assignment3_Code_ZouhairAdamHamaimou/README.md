# AER1415 – Assignment 3

This repository contains all scripts, data, and results for **Assignment 3** of the AER1415 course. The assignment covers stochastic gradient-based optimization, hybrid metaheuristics, and tolerance analysis in constrained engineering design.

## Contents Overview

### 📂 Data Files
- `airfoil_training_data.dat` — Training dataset for airfoil noise regression.
- `airfoil_testing_data.dat` — Test dataset for airfoil noise regression.

### 📂 Python Scripts
- `Q2.py` — Solution code for Question 2 (Constrained Optimization).
- `Q5.py` — Main script for Question 5: SGD vs CG comparison.
- `Q5_bonus_momentum.py` — Bonus: Implements SGD with momentum and compares against standard SGD and CG.
- `Q6.py` — Hybrid PSO + SLSQP algorithm applied to the bump test function.
- `particle_swarm_optimizer.py` — Particle Swarm Optimization (PSO) implementation with static penalty.
- `bump_test_function.py` — Defines the bump test function and constraints for Q6.
- `minimize.py` — Conjugate Gradient optimization utility.
- `loss.py` — Loss function and gradients for neural network regression.
- `model_predict.py` — Neural network forward pass for predictions.
- `airfoil_train_script.py` — Provided baseline script for CG training.

### 📂 Results Files
- `results_q5.csv` — Performance summary of SGD vs CG (Q5).
- `results_q5_bonus.csv` — Performance summary of SGD with momentum vs others.
- `runs_bsXX_lr1e-XX.npy` — Cached per-configuration results for SGD runs.
- `cg_runs.npy` — Cached CG results.
- `loss_histories.npy`, `grad_histories.npy`, `bonus_loss_histories.npy` — Global loss/gradient history caches.

### 📂 Figures
- `loss_overlay_grid.png` — Loss convergence for different SGD hyperparameters.
- `gradnorm_overlay_grid.png` — Gradient norm convergence for different SGD hyperparameters.
- `momentum_vs_sgd.png` — Loss comparison: SGD w/ momentum vs standard SGD vs CG.
- `gradnorm_vs_sgd_mom_cg.png` — Gradient norm comparison of SGD variants and CG.
- `pso_hybrid_comparison.png` — Convergence comparison: hybrid PSO–SLSQP vs standard PSO.

---

## Instructions

### Q5 — SGD vs CG
1. Run `Q5.py` to train neural networks using various SGD hyperparameters and CG baseline.
2. Outputs: loss plots, gradient plots, and `results_q5.csv`.

### Bonus — SGD with Momentum
1. Run `Q5_bonus_momentum.py` to compare SGD with momentum (default $\beta=0.9$).
2. Modify momentum via CLI: `python Q5_bonus_momentum.py --beta 0.5`
3. Outputs: `results_q5_bonus.csv`, momentum plots.

### Q6 — Hybrid PSO
1. Run `Q6.py` to apply hybrid PSO–SLSQP to the bump test function.
2. Adjust `k` and `n_refine` inside the script for refinement control.
3. Outputs: performance plots and comparison to standard PSO.

---

## Notes
- Cached `.npy` files ensure that previously computed runs are not repeated.
- All figures are generated at high DPI (300) for inclusion in the final report.

---

## Report Reference
Please refer to `Assignment3.pdf` for detailed problem statements and LaTeX-formatted solutions, figures, and tables derived from this codebase.

---
