# Quadratic Optimization Methods

This repository contains implementations of various optimization methods for solving convex quadratic problems. The methods included are:

- **Steepest Descent with Exact Line Search** (Q2)
- **Barzilai-Borwein (BB) Method** (Q4)
- **Quasi-Newton Methods (BFGS)** (Q3)
- **Contour Plot Visualization for Quadratic Functions** (Q1a, Q1c)

## Requirements

The code has been tested with **Python 3.9** and requires the following dependencies:

- `numpy`
- `scipy`
- `matplotlib`
- `pickle`

You can install the required dependencies using:

```bash
pip install numpy scipy matplotlib
```

## File Structure

```bash
├── Q1a_contour_plots.py        # Contour plots for Q1a
├── Q1c_contour_plots.py        # Contour plots for Q1c
├── Q2.py                       # Steepest Descent with Exact Line Search
├── Q3.py                       # BFGS Quasi-Newton Method
├── Q4.py                       # Barzilai-Borwein (BB) Method
├── ConvexQuadratic.mat         # Data file containing A (341x341 SPD) and b (341x1)
├── steepest_descent_results.pkl # Results of steepest descent (for Q4 comparison)
├── barzilai_borwein_results.pkl # Results of BB method
├── plots/                      # Folder containing generated plots
```

## Running the Code

### 1. Steepest Descent (Q2)

To run the steepest descent method with exact line search:

```bash
python Q2.py
```

This script:

- Loads the quadratic problem data from `ConvexQuadratic.mat`
- Runs the steepest descent method for 10 trials with different initial points
- Saves results in `steepest_descent_results.pkl`
- Generates and saves plots in `plots/`

### 2. Barzilai-Borwein (BB) Method (Q4)

To run the BB method:

```bash
python Q4.py
```

This script:

- Loads the same initial points from `steepest_descent_results.pkl`
- Runs the BB method for 10 trials
- Saves results in `barzilai_borwein_results.pkl`
- Generates comparative plots of BB vs. steepest descent

### 3. Quasi-Newton BFGS (Q3)

To run the BFGS method:

```bash
python Q3.py
```

This script:

- Tests BFGS on the Rosenbrock function for different dimensions
- Compares it against the steepest descent method
- Generates and saves plots in `plots/`

### 4. Contour Plots for Visualization (Q1)

To generate contour plots:

```bash
python Q1a_contour_plots.py
python Q1c_contour_plots.py
```

These scripts generate contour plots for specific quadratic functions and highlight critical points.

## Output and Results

All figures generated are stored in the `plots/` directory. The numerical results, including convergence statistics, are printed to the console and saved in `steepest_descent_results.pkl` and `barzilai_borwein_results.pkl` for analysis.

For further comparison of methods, analyze the output plots in `plots/` and compare the numerical performance of steepest descent vs. Barzilai-Borwein and BFGS.

## Notes

- Ensure that `ConvexQuadratic.mat` is present in the working directory before running the scripts.
- The scripts will automatically create the `plots/` directory if it does not exist.
- The `pkl` files store intermediate results to facilitate comparisons across different methods.