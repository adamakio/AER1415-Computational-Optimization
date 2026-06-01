# Assignment 2 — Gradient-Based Optimization

## Overview

Implement and compare three gradient-based optimization methods on a large-scale convex quadratic problem (`A` is 341×341 symmetric positive definite):

| Method | Script | Description |
|--------|--------|-------------|
| Steepest Descent + exact line search | `Q2.py` | Baseline first-order method |
| Barzilai-Borwein (BB) | `Q4.py` | Non-monotone gradient method with step recycling |
| BFGS (quasi-Newton) | `Q3.py` | Second-order approximation via secant updates |

Contour plot visualizations compare convergence paths on 2D quadratic variants (`Q1a`, `Q1c`). 10 random trials per method quantify statistical convergence behavior.

## Key Results

- BB consistently outperforms steepest descent in iteration count, with non-monotone steps enabling faster escape from slow-convergence regions
- BFGS achieves superlinear convergence on the Rosenbrock function, dramatically outperforming first-order methods at higher dimensions

## Files

| File | Description |
|------|-------------|
| `AER1415_Assignment2_Code_ZouhairHamaimou/Q2.py` | Steepest descent implementation |
| `AER1415_Assignment2_Code_ZouhairHamaimou/Q3.py` | BFGS implementation |
| `AER1415_Assignment2_Code_ZouhairHamaimou/Q4.py` | Barzilai-Borwein implementation |
| `AER1415_Assignment2_Code_ZouhairHamaimou/Q1a_contour_plots.py` | Contour visualization (Q1a) |
| `AER1415_Assignment2_Code_ZouhairHamaimou/Q1c_contour_plots.py` | Contour visualization (Q1c) |
| `AER1415_Assignment2_Code_ZouhairHamaimou/ConvexQuadratic.mat` | Problem data (A, b) |
| `AER1415_Assignment2_Code_ZouhairHamaimou/plots/` | All generated figures |
| `AER1415_Assignment2_Report_ZouhairHamaimou.pdf` | Submitted report |
| `Assignment2.pdf` | Assignment instructions |

## Running

```bash
python AER1415_Assignment2_Code_ZouhairHamaimou/Q2.py   # Steepest descent
python AER1415_Assignment2_Code_ZouhairHamaimou/Q4.py   # Barzilai-Borwein (run after Q2)
python AER1415_Assignment2_Code_ZouhairHamaimou/Q3.py   # BFGS
```
