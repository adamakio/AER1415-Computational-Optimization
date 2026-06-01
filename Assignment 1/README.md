# Assignment 1 — Particle Swarm Optimization

## Overview

Implement and tune Particle Swarm Optimization (PSO) — a population-based metaheuristic — and apply it to five engineering problems spanning 2D through 50D search spaces.

**Hyperparameter tuning:** grid search over inertia weight `w`, cognitive coefficient `c1`, and social coefficient `c2`, comparing static vs. adaptive strategies. Best configuration: `w=0.406, c1=1.527, c2=2.135` (static), with adaptive variant using a cooling schedule.

## Problems

| Problem | Description | Dimensions |
|---------|-------------|------------|
| P1 | Rosenbrock function | 2D, 5D |
| P2 | Case study (MeasuredResponse system identification) | — |
| P3 | Bump test function (constrained maximization) | 2D, 10D, **50D (bonus)** |
| P4 | Brachistochrone problem (discrete path) | n=15, n=30 |
| P5 | Inverse problem (parameter identification) | — |

## Files

| File | Description |
|------|-------------|
| `particle_swarm_optimizer.py` | Core PSO implementation (static & adaptive) |
| `bump_test_function_optimize.py` | Hyperparameter search on 2D bump function |
| `rosenbrock_test_function.py` | P1 — Rosenbrock |
| `p2_case_study.py` | P2 — Case study |
| `bump_test_function.py` | P3 — Bump function |
| `brachistochrone_problem.py` | P4 — Brachistochrone |
| `inverse_problem.py` | P5 — Inverse problem |
| `verify_P3_n50.py` | Verifies bonus 50D solution |
| `best_solution_P3_n50.txt` | Best found solution for 50D bonus |
| `create_latex_tables.py` | Generates LaTeX result tables |
| `MeasuredResponse.dat` | Dataset for P2 |
| `requirements.txt` | Python dependencies |
| `best_plots/` | Representative convergence plots (13 figures) |
| `best_solutions/` | Best solution files |
| `assignment/` | Assignment instructions |
| `report/` | Submitted report (PDF + LaTeX source) |

## Running

```bash
pip install -r requirements.txt

# Tune hyperparameters
python bump_test_function_optimize.py

# Run problems (set dimensions inside script as needed)
python rosenbrock_test_function.py   # P1
python p2_case_study.py              # P2
python bump_test_function.py         # P3
python brachistochrone_problem.py    # P4
python inverse_problem.py            # P5

# Verify bonus solution
python verify_P3_n50.py
```
