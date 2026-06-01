# Assignment 3 — Stochastic & Hybrid Optimization

## Overview

Three problems covering stochastic gradient methods, hybrid metaheuristics, and constrained optimization applied to real aerospace data.

### Q2 — Constrained Optimization
KKT conditions and penalty methods on an engineering design problem.

### Q5 — SGD vs. Conjugate Gradient for Neural Network Training
Train a neural network to predict **airfoil self-noise** (UCI airfoil dataset) using:
- **SGD** — mini-batch gradient descent with grid search over batch size ∈ {16, 32, 64} and learning rate ∈ {1e-1, 1e-2, 1e-3, 1e-4}
- **SGD + Momentum** (bonus) — β = 0.9, compared against standard SGD and CG
- **Conjugate Gradient** — from-scratch implementation as baseline

### Q6 — Hybrid PSO–SLSQP
Combine PSO global search with SLSQP local refinement on the constrained Bump test function. PSO identifies promising basins; SLSQP polishes the best candidates.

## Files

| File | Description |
|------|-------------|
| `AER1415_Assignment3_Code_ZouhairAdamHamaimou/Q2.py` | Constrained optimization |
| `AER1415_Assignment3_Code_ZouhairAdamHamaimou/Q5.py` | SGD vs. CG training |
| `AER1415_Assignment3_Code_ZouhairAdamHamaimou/Q5_bonus_momentum.py` | SGD + momentum variant |
| `AER1415_Assignment3_Code_ZouhairAdamHamaimou/Q6.py` | Hybrid PSO–SLSQP |
| `AER1415_Assignment3_Code_ZouhairAdamHamaimou/minimize.py` | Conjugate Gradient implementation |
| `AER1415_Assignment3_Code_ZouhairAdamHamaimou/loss.py` | Neural network loss + gradients |
| `AER1415_Assignment3_Code_ZouhairAdamHamaimou/model_predict.py` | Neural network forward pass |
| `AER1415_Assignment3_Code_ZouhairAdamHamaimou/airfoil_train_script.py` | Training driver |
| `AER1415_Assignment3_Code_ZouhairAdamHamaimou/airfoil_training_data.dat` | UCI airfoil dataset (train) |
| `AER1415_Assignment3_Code_ZouhairAdamHamaimou/airfoil_testing_data.dat` | UCI airfoil dataset (test) |
| `AER1415_Assignment3_Code_ZouhairAdamHamaimou/plots/` | Loss/gradient convergence figures |
| `AER1415_Assignment3_Zouhair_Hamaimou_1004891986.pdf` | Submitted report |

## Running

```bash
python AER1415_Assignment3_Code_ZouhairAdamHamaimou/Q5.py            # SGD vs CG
python AER1415_Assignment3_Code_ZouhairAdamHamaimou/Q5_bonus_momentum.py  # Momentum
python AER1415_Assignment3_Code_ZouhairAdamHamaimou/Q6.py            # Hybrid PSO
```
