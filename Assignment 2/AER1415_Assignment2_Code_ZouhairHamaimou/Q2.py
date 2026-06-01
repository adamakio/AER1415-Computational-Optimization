# Question 2: Steepest descent method for convex quadratics
# Import necessary libraries
import os
import numpy as np
import scipy.io
import pickle
import matplotlib.pyplot as plt


def steepest_descent(A, b, x0, num_iters=1000):
    """
    Implements the steepest descent algorithm with exact line search for
    minimizing the quadratic function f(x) = 0.5 * x^T * A * x - b^T * x.

    Parameters:
    - A: SPD matrix (n x n)
    - b: Vector (n,)
    - x0: Initial guess (n,)
    - num_iters: Number of iterations (default 1000)

    Returns:
    - fs: List of function values f(x_k)
    - gs: List of gradient norms ||g(x_k)||
    """
    x = x0
    fs, gs = [], []

    for k in range(num_iters):
        g = A @ x - b  # Compute gradient
        p = -g  # Descent direction
        
        # Exact line search step size
        alpha = -(p.T @ g) / (p.T @ A @ p)
        
        # Update x
        x = x + alpha * p

        # Record function value and gradient norm
        fs.append(0.5 * x.T @ A @ x - b.T @ x)
        gs.append(np.linalg.norm(g))

    return np.array(fs), np.array(gs)

if __name__ == '__main__':
    # Load the .mat file
    data = scipy.io.loadmat('ConvexQuadratic.mat')
    A = data['A']  # 341x341 SPD matrix
    b = data['b'].flatten()  # Convert (341,1) to (341,)

    # Generate 10 random initial guesses
    num_trials = 10
    num_iters = 1000
    n = A.shape[0]
    initial_guesses = [np.random.uniform(-10, 10, n) for _ in range(num_trials)]

    # Run the algorithm for 10 trials
    all_fs, all_gs = [], []
    for x0 in initial_guesses:
        fs, gs = steepest_descent(A, b, x0, num_iters)
        all_fs.append(fs)
        all_gs.append(gs)

    # Convert to numpy arrays for analysis
    all_fs = np.array(all_fs)  # Shape: (10, 1000)
    all_gs = np.array(all_gs)  # Shape: (10, 1000)

    # Find the lowest gradient norm achieved across all trials
    lowest_gradient_norm = np.min(all_gs)

    # find the lowest objective value achieved across all trials
    lowest_obj_value = np.min(all_fs)

    # Save initial points and results for comparison with Barzilai-Borwein method
    results = {
        "initial_guesses": initial_guesses,
        "steepest_descent_fs": all_fs,
        "steepest_descent_gs": all_gs,
        "num_iters": num_iters
    }

    with open("steepest_descent_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Plot the convergence of the objective function and gradient norm
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))


    # Plot and save all trials of objective function values
    if os.path.exists("plots") == False:
        os.mkdir("plots")

    plt.figure(figsize=(8, 6), dpi=300)
    for i in range(num_trials):
        plt.plot(all_fs[i], label=f'Trial {i+1}')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/objective_function_all_trials.png", dpi=300, bbox_inches='tight')

    # Plot and save zoomed-in objective function values
    plt.figure(figsize=(8, 6), dpi=300)
    for i in range(num_trials):
        plt.plot(all_fs[i], label=f'Trial {i+1}')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.ylim([-11, 50])
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/objective_function_zoomed.png", dpi=300, bbox_inches='tight')

    # Plot and save all trials of gradient norms using semilogy (log scale on y-axis)
    plt.figure(figsize=(8, 6), dpi=300)
    for i in range(num_trials):
        plt.semilogy(all_gs[i], label=f'Trial {i+1}')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm (log scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/gradient_norm_all_trials.png", dpi=300, bbox_inches='tight')


    # Print the lowest gradient norm achieved
    print(f"Lowest gradient norm achieved within 1000 iterations: {lowest_gradient_norm:.6f}")

    # Print the lowest objective value achieved
    print(f"Lowest objective value achieved within 1000 iterations: {lowest_obj_value:.6f}")
