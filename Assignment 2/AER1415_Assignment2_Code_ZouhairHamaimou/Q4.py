import os
import json
import numpy as np
import scipy.io
import pickle
import matplotlib.pyplot as plt

def barzilai_borwein(A, b, x0, num_iters=1000):
    """
    Implements the Barzilai-Borwein (BB) method for minimizing
    the quadratic function f(x) = 0.5 * x^T * A * x - b^T * x.

    Parameters:
    - A: SPD matrix (n x n)
    - b: Vector (n,)
    - x0: Initial guess (n,)
    - num_iters: Number of iterations (default 1000)

    Returns:
    - fs: List of function values f(x_k)
    - gs: List of gradient norms ||g(x_k)||
    - total_convergence_iters: Number of iterations before function stops improving
    - num_increases: Percentage of non-monotonic steps (where f(x_{k+1}) > f(x_k))
    - max_increase: Maximum observed increase in objective function value
    """
    x_prev = x0
    x = x0.copy()
    fs, gs = [], []
    num_increases = 0
    max_increase = 0
    total_convergence_iters = num_iters  # Track when convergence actually happens

    # First iteration: Steepest descent with exact line search
    g_prev = A @ x - b
    p = -g_prev
    alpha = -(p.T @ g_prev) / (p.T @ A @ p)  # Exact line search
    x = x + alpha * p

    f_prev = 0.5 * x.T @ A @ x - b.T @ x
    fs.append(f_prev)
    gs.append(np.linalg.norm(g_prev))

    # Parameters for convergence check
    tol = 1e-6

    # BB iterations
    for k in range(1, num_iters):
        g = A @ x - b
        s = x - x_prev  # Corrected s update
        y = g - g_prev

        if np.dot(s, y) > 0:  # Ensure denominator is positive
            alpha = np.dot(s, s) / np.dot(s, y)
        else:
            alpha = 1e-2  # Small step size to avoid division by zero

        x_prev = x  # Store previous x for next iteration
        x = x - alpha * g
        f_new = 0.5 * x.T @ A @ x - b.T @ x

        # Track non-monotonic behavior
        if f_new > f_prev:
            num_increases += 1
            max_increase = max(max_increase, f_new - f_prev)

        f_prev = f_new
        g_prev = g

        # Determine when function stops improving over a window of iterations
        if np.linalg.norm(g) < tol and total_convergence_iters == num_iters:
            total_convergence_iters = k
            
        fs.append(f_new)
        gs.append(np.linalg.norm(g))

    return (
        np.array(fs), np.array(gs), total_convergence_iters,
        num_increases / num_iters * 100, max_increase
    )

if __name__ == '__main__':
    if os.path.exists("plots") == False:
        os.mkdir("plots")
        
    # Load saved results from steepest descent
    with open("steepest_descent_results.pkl", "rb") as f:
        sd_results = pickle.load(f)

    initial_guesses = sd_results["initial_guesses"]
    num_iters = sd_results["num_iters"]

    # Load the .mat file
    data = scipy.io.loadmat('ConvexQuadratic.mat')
    A = data['A']  # 341x341 SPD matrix
    b = data['b'].flatten()  # Convert (341,1) to (341,)

    num_trials = len(initial_guesses)
    bb_fs_all, bb_gs_all = [], []
    bb_total_convergence_iters, bb_non_monotonic_rates, bb_max_increases = [], [], []

    for x0 in initial_guesses:
        fs, gs, total_iters, non_monotonic_rate, max_increase = barzilai_borwein(A, b, x0, num_iters)
        bb_fs_all.append(fs)
        bb_gs_all.append(gs)
        bb_total_convergence_iters.append(total_iters)
        bb_non_monotonic_rates.append(non_monotonic_rate)
        bb_max_increases.append(max_increase)

    # Convert to numpy arrays for analysis
    bb_fs_all = np.array(bb_fs_all)
    bb_gs_all = np.array(bb_gs_all)

    # Save results for further analysis
    bb_results = {
        "bb_fs": bb_fs_all,
        "bb_gs": bb_gs_all,
        "bb_total_convergence_iters": bb_total_convergence_iters,
        "bb_non_monotonic_rates": bb_non_monotonic_rates,
        "bb_max_increases": bb_max_increases,
        "num_iters": num_iters
    }

    with open("barzilai_borwein_results.pkl", "wb") as f:
        pickle.dump(bb_results, f)

    # Choose representative trials (median, min, max of total iterations)
    sorted_trials = np.argsort(bb_total_convergence_iters)
    print(f"Sorted Trials: {sorted_trials+1}")
    min_trial, med_trial, max_trial = sorted_trials[0], sorted_trials[len(sorted_trials) // 2], sorted_trials[-1]
    print(f"Representative Trials: Min={min_trial+1}, Median={med_trial+1}, Max={max_trial+1}")
    representative_trials = [min_trial, med_trial, max_trial]

    # Plot convergence for the selected trials
    for i in representative_trials:
        plt.figure(figsize=(8, 6))
        plt.plot(bb_fs_all[i], label="BB Objective Function")
        plt.plot(sd_results["steepest_descent_fs"][i], label="SD Objective Function", linestyle='dashed')
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value (log scale)")
        plt.legend()
        plt.title(f"BB vs SD Convergence for Trial {i+1}")
        plt.grid(True)
        plt.savefig(f"plots/bb_vs_sd_obj_trial{i+1}.png", dpi=300)
        # plt.show()

        # Zoomed-in plot with ylim = [-11, 50]
        plt.figure(figsize=(8, 6))
        plt.plot(bb_fs_all[i], label="BB Objective Function")
        plt.plot(sd_results["steepest_descent_fs"][i], label="SD Objective Function", linestyle='dashed')
        plt.ylim([-11, 50])
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value")
        plt.legend()
        plt.title(f"BB vs SD Convergence (Zoomed) for Trial {i+1}")
        plt.grid(True)
        plt.savefig(f"plots/bb_vs_sd_obj_zoom_trial{i+1}.png", dpi=300)
        # plt.show()

        # Gradient norm plot
        plt.figure(figsize=(8, 6))
        plt.semilogy(bb_gs_all[i], label="BB Gradient Norm")
        plt.semilogy(sd_results["steepest_descent_gs"][i], label="SD Gradient Norm", linestyle='dashed')
        plt.xlabel("Iteration")
        plt.ylabel("Gradient Norm (log scale)")
        plt.legend()
        plt.title(f"BB vs SD Gradient Norm for Trial {i+1}")
        plt.grid(True)
        plt.savefig(f"plots/bb_vs_sd_grad_trial{i+1}.png", dpi=300)
        # plt.show()

    # Print the best trial objective value and gradient norm
    best_trial = np.argmin(bb_fs_all[:, -1])
    print(f"Best BB Trial: Objective Value={bb_fs_all[best_trial, -1]:.6f}, Gradient Norm={bb_gs_all[best_trial, -1]:.6f}")

    # Compute statistics over all trials
    stats = lambda data: (np.mean(data), np.std(data), np.min(data), np.max(data))

    avg_iters_bb, std_iters_bb, min_iters_bb, max_iters_bb = stats(bb_total_convergence_iters)
    avg_non_monotonic_rate, std_non_monotonic_rate, min_non_monotonic_rate, max_non_monotonic_rate = stats(bb_non_monotonic_rates)
    avg_max_increase, std_max_increase, min_max_increase, max_max_increase = stats(bb_max_increases)

    print("Analysis of Barzilai-Borwein (BB) Method:")

    full_results = {
        f"Trial {i+1}": {
            "Total Iterations to Convergence": bb_total_convergence_iters[i],
            "Non-Monotonic Rate (%)": bb_non_monotonic_rates[i],
            "Maximum Increase in Objective Value": bb_max_increases[i]
        } for i in sorted_trials
    }
    print(f"Full results {json.dumps(full_results, indent=4)}")
    print(f"Total Iterations to Convergence: Mean={avg_iters_bb:.2f}, Std={std_iters_bb:.2f}, Min={min_iters_bb}, Max={max_iters_bb}")
    print(f"Non-Monotonic Rate (%): Mean={avg_non_monotonic_rate:.2f}, Std={std_non_monotonic_rate:.2f}, Min={min_non_monotonic_rate:.2f}, Max={max_non_monotonic_rate:.2f}")
    print(f"Maximum Increase in Objective Value: Mean={avg_max_increase:.6f}, Std={std_max_increase:.6f}, Min={min_max_increase:.6f}, Max={max_max_increase:.6f}")

    # Boxplots for non-monotonic behavior
    plt.figure(figsize=(8, 6))
    plt.boxplot(bb_non_monotonic_rates, vert=True, patch_artist=True)
    plt.ylabel("Percentage of Non-Monotonic Steps")
    plt.title("Distribution of Non-Monotonic Behavior Across Trials")
    plt.grid(True)
    plt.savefig("plots/bb_non_monotonic_distribution.png", dpi=300)
    # plt.show()

    # Boxplots for maximum increase in objective value
    plt.figure(figsize=(8, 6))
    plt.boxplot(bb_max_increases, vert=True, patch_artist=True)
    plt.ylabel("Maximum Increase in Objective Value")
    plt.title("Distribution of Maximum Objective Value Increases Across Trials")
    plt.grid(True)
    plt.savefig("plots/bb_max_increase_distribution.png", dpi=300)
    # plt.show()

    # Boxplots for total iterations to convergence
    plt.figure(figsize=(8, 6))
    plt.boxplot(bb_total_convergence_iters, vert=True, patch_artist=True)
    plt.ylabel("Total Iterations to Convergence")
    plt.title("Distribution of Total Iterations to Convergence Across Trials")
    plt.grid(True)
    plt.savefig("plots/bb_total_iters_distribution.png", dpi=300)
    # plt.show()
