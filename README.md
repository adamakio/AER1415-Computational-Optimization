# AER1415 — Computational Optimization

**University of Toronto | MEng Aerospace Science & Engineering | Winter 2025**

A graduate course covering the theory and implementation of optimization algorithms — from classical gradient methods to metaheuristics and stochastic methods — applied to engineering problems including aerospace design and neural network training.

---

## Assignments

| Folder | Topic | Key Methods |
|--------|-------|-------------|
| [Assignment 1](Assignment%201/) | Metaheuristic Optimization | Particle Swarm Optimization (static & adaptive hyperparameters), applied to Rosenbrock, Bump, Brachistochrone, and inverse problems |
| [Assignment 2](Assignment%202/) | Gradient-Based Optimization | Steepest descent, Barzilai-Borwein, BFGS — benchmarked on convex quadratic (341×341) |
| [Assignment 3](Assignment%203/) | Stochastic & Hybrid Methods | SGD, SGD + momentum, Conjugate Gradient, PSO–SLSQP hybrid — neural network trained on airfoil aerodynamic data |

---

## Skills Demonstrated

- **Metaheuristics:** PSO with static and adaptive inertia/acceleration hyperparameters, bonus 50-D solution
- **Classical gradient methods:** Exact line search, Barzilai-Borwein non-monotone step, BFGS quasi-Newton
- **Stochastic optimization:** SGD with mini-batches, momentum; hyperparameter grid search (batch size × learning rate)
- **Hybrid methods:** PSO + SLSQP local refinement for constrained problems
- **ML application:** Neural network regression on real airfoil acoustic dataset (UCI)
- **Conjugate Gradient:** Implementation from scratch, compared against SGD variants

## Tools

Python · NumPy · SciPy · Matplotlib
