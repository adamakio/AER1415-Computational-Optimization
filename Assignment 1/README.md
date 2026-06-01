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

## Structure

```
Assignment 1/
├── particle_swarm_optimizer.py    # Core PSO (static & adaptive)
├── bump_test_function_optimize.py # Hyperparameter grid search
├── rosenbrock_test_function.py    # P1
├── p2_case_study.py               # P2
├── bump_test_function.py          # P3
├── brachistochrone_problem.py     # P4
├── inverse_problem.py             # P5
├── verify_P3_n50.py               # Verify bonus 50D solution
├── MeasuredResponse.dat           # Data for P2/P5
├── requirements.txt
├── results/
│   ├── best_plots/                # Best convergence plots per problem
│   ├── best_solutions/            # Best solution files
│   ├── best_solution_P3_n50.txt   # Bonus 50D result
│   ├── latex_adaptive_table.txt   # Result tables (adaptive PSO)
│   └── latex_static_table.txt     # Result tables (static PSO)
├── utils/
│   ├── create_latex_tables.py     # Generate LaTeX tables from results
│   └── get_top_runs_png_filenames.py
├── instructions/
│   ├── Assignment1-1.pdf
│   └── CaseStudies.pdf
└── report/
    ├── AER1415_Assignment_1_ZouhairAdamHamaimou.pdf
    └── AER1415_Assignment_1_ZouhairAdamHamaimou.tex
```

## Running

```bash
pip install -r requirements.txt

# Tune hyperparameters first
python bump_test_function_optimize.py

# Run problems (set dimensions inside script as needed)
python rosenbrock_test_function.py   # P1
python p2_case_study.py              # P2
python bump_test_function.py         # P3
python brachistochrone_problem.py    # P4
python inverse_problem.py            # P5

# Verify bonus 50D solution
python verify_P3_n50.py
```
