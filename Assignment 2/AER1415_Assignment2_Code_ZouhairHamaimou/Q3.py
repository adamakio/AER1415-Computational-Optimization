import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen, rosen_der

def backtracking_line_search(f, grad_f, xk, fk, pk, mu1=1e-4, rho=0.5, alpha_init=0.9):
    alpha = alpha_init
    evals = 1
    fk_new = f(xk + alpha * pk)
    while fk_new > fk + mu1 * alpha * np.dot(grad_f(xk), pk):
        alpha *= rho
        fk_new = f(xk + alpha * pk)
        evals += 1
    return alpha, fk_new, evals

def bfgs(f, grad_f, x0, tol_g=1e-6, tol_r=1e-6, max_iters=10000, mu1=1e-4, rho=0.5):
    n = len(x0)
    B_inv = np.eye(n)
    xk = x0.copy()
    fk = f(xk)
    grad_k = grad_f(xk)
    function_evals = 1
    history_f = []
    history_g = []
    
    for k in range(max_iters):
        if np.linalg.norm(grad_k) < tol_g:
            break
        
        pk = -B_inv @ grad_k
        alpha_k, fk_new, evals = backtracking_line_search(f, grad_f, xk, fk, pk, mu1, rho)
        function_evals += evals
        sk = alpha_k * pk
        xk_new = xk + sk
        grad_k_new = grad_f(xk_new)
        function_evals += 1
        
        if np.abs(fk_new - fk) < tol_r * np.abs(fk):
            break
        
        yk = grad_k_new - grad_k
        rho_k = 1.0 / (yk.T @ sk) if np.abs(yk.T @ sk) > 1e-10 else 0.0
        
        if rho_k > 0:
            B_inv = (np.eye(n) - rho_k * np.outer(sk, yk)) @ B_inv @ (np.eye(n) - rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)
        
        xk, fk, grad_k = xk_new, fk_new, grad_k_new
        history_f.append(fk)
        history_g.append(np.linalg.norm(grad_k))
        
    return xk, fk, function_evals, len(history_f), history_f, history_g

def steepest_descent(f, grad_f, x0, tol_g=1e-6, tol_r=1e-6, max_iters=10000, mu1=1e-4, rho=0.5):
    xk = x0.copy()
    fk = f(xk)
    grad_k = grad_f(xk)
    function_evals = 1
    history_f = []
    history_g = []
    
    for k in range(max_iters):
        if np.linalg.norm(grad_k) < tol_g:
            break
        
        pk = -grad_k
        alpha_k, fk_new, evals = backtracking_line_search(f, grad_f, xk, fk, pk, mu1, rho)
        function_evals += evals
        xk_new = xk + alpha_k * pk
        grad_k_new = grad_f(xk_new)
        function_evals += 1
        
        if np.abs(fk_new - fk) < tol_r * np.abs(fk):
            break
        
        xk, fk, grad_k = xk_new, fk_new, grad_k_new
        history_f.append(fk)
        history_g.append(np.linalg.norm(grad_k))
        
    return xk, fk, function_evals, len(history_f), history_f, history_g

def best_trial(metric, method_results):
    return np.argmin([data[metric] for data in method_results])

if __name__ == "__main__":
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    np.random.seed(42)
    n_values = [2, 5]
    n_trials = 10
    results = {}

    success_threshold = 1e-3  # Gradient norm threshold for success
    
    for n in n_values:
        initial_guesses = [np.random.uniform(-5, 5, n) for _ in range(n_trials)]
        bfgs_data = []
        sd_data = []
        
        for i, x0 in enumerate(initial_guesses):
            x_opt_bfgs, f_opt_bfgs, fevals_bfgs, iters_bfgs, history_f_bfgs, history_g_bfgs = bfgs(rosen, rosen_der, x0)
            success_bfgs = 1 if min(history_g_bfgs) < success_threshold else 0
            bfgs_data.append((f_opt_bfgs, fevals_bfgs, iters_bfgs, success_bfgs, history_f_bfgs, history_g_bfgs, x_opt_bfgs))
            
            x_opt_sd, f_opt_sd, fevals_sd, iters_sd, history_f_sd, history_g_sd = steepest_descent(rosen, rosen_der, x0)
            success_sd = 1 if min(history_g_sd) < success_threshold else 0
            sd_data.append((f_opt_sd, fevals_sd, iters_sd, success_sd, history_f_sd, history_g_sd, x_opt_sd))
            
            print(f"Trial {i+1} for n={n} completed.")
            
        results[n] = {"bfgs": bfgs_data, "sd": sd_data}
 
        # Plotting section
        # Plot objective function value for all BFGS trials
        plt.figure(figsize=(8, 6))
        for trial in range(n_trials):
            plt.semilogy(results[n]["bfgs"][trial][4], alpha=0.5)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Objective Function Value")
        plt.title(f"BFGS Convergence (n={n})")
        plt.grid(True)
        plt.savefig(f"plots/bfgs_obj_all_trials_n{n}.png", dpi=300)
        # plt.show()

        # Plot objective function value for all Steepest Descent trials
        plt.figure(figsize=(8, 6))
        for trial in range(n_trials):
            plt.semilogy(results[n]["sd"][trial][4], alpha=0.5)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Objective Function Value")
        plt.title(f"Steepest Descent Convergence (n={n})")
        plt.grid(True)
        plt.savefig(f"plots/sd_obj_all_trials_n{n}.png", dpi=300)
        # plt.show()

        # Log-plot of gradient norm for all BFGS trials
        plt.figure(figsize=(8, 6))
        for trial in range(n_trials):
            plt.semilogy(results[n]["bfgs"][trial][5], alpha=0.5)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Gradient Norm (log scale)")
        plt.title(f"BFGS Gradient Norm (n={n})")
        plt.grid(True)
        plt.savefig(f"plots/bfgs_grad_all_trials_n{n}.png", dpi=300)
        # plt.show()

        # Log-plot of gradient norm for all Steepest Descent trials
        plt.figure(figsize=(8, 6))
        for trial in range(n_trials):
            plt.semilogy(results[n]["sd"][trial][5], alpha=0.5)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Gradient Norm (log scale)")
        plt.title(f"Steepest Descent Gradient Norm (n={n})")
        plt.grid(True)
        plt.savefig(f"plots/sd_grad_all_trials_n{n}.png", dpi=300)
        # plt.show()

        # Identify the best trial based on the lowest objective value achieved
        best_bfgs_trial = best_trial(0, results[n]["bfgs"])
        best_sd_trial = best_trial(0, results[n]["sd"])

        # Compare best trials of BFGS and Steepest Descent
        plt.figure(figsize=(8, 6))
        plt.semilogy(results[n]["bfgs"][best_bfgs_trial][4], label="Best BFGS Trial")
        plt.semilogy(results[n]["sd"][best_sd_trial][4], label="Best SD Trial")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Objective Function Value")
        plt.title(f"Best BFGS vs Best SD (n={n})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/best_bfgs_vs_best_sd_obj_n{n}.png", dpi=300)
        # plt.show()

        # Log-plot of gradient norm for best trials of both methods
        plt.figure(figsize=(8, 6))
        plt.semilogy(results[n]["bfgs"][best_bfgs_trial][5], label="Best BFGS Trial")
        plt.semilogy(results[n]["sd"][best_sd_trial][5], label="Best SD Trial")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Gradient Norm (log scale)")
        plt.title(f"Best BFGS vs Best SD (n={n})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/best_bfgs_vs_best_sd_grad_n{n}.png", dpi=300)
        # plt.show()

        # Results section
        avg_iters_bfgs = np.mean([data[2] for data in results[n]['bfgs']])
        avg_iters_sd = np.mean([data[2] for data in results[n]['sd']])
        avg_fevals_bfgs = np.mean([data[1] for data in results[n]['bfgs']])
        avg_fevals_sd = np.mean([data[1] for data in results[n]['sd']])
        std_obj_bfgs = np.std([data[0] for data in results[n]['bfgs']])
        std_obj_sd = np.std([data[0] for data in results[n]['sd']])
        success_rate_bfgs = np.mean([data[3] for data in results[n]['bfgs']]) * 100
        success_rate_sd = np.mean([data[3] for data in results[n]['sd']]) * 100

        print(f"Results for n={n}")
        print(f"Lowest Objective Value (BFGS): {min([data[0] for data in results[n]['bfgs']]):.6f}")
        print(f"Lowest Objective Value (Steepest Descent): {min([data[0] for data in results[n]['sd']]):.6f}")
        print(f"Lowest Gradient Norm (BFGS): {min([min(data[5]) for data in results[n]['bfgs']]):.6f}")
        print(f"Lowest Gradient Norm (Steepest Descent): {min([min(data[5]) for data in results[n]['sd']]):.6f}")
        print(f"Standard Deviation of Objective Value (BFGS): {std_obj_bfgs:.6f}")
        print(f"Standard Deviation of Objective Value (Steepest Descent): {std_obj_sd:.6f}")
        print(f"Optimal x for lowest BFGS objective value: {results[n]['bfgs'][best_bfgs_trial][6]}")
        print(f"Optimal x for lowest Steepest Descent objective value: {results[n]['sd'][best_sd_trial][6]}")
        print(f"Average Iterations to Convergence (BFGS): {avg_iters_bfgs:.2f}")
        print(f"Average Iterations to Convergence (Steepest Descent): {avg_iters_sd:.2f}")
        print(f"Average Function Evaluations (BFGS): {avg_fevals_bfgs:.2f}")
        print(f"Average Function Evaluations (Steepest Descent): {avg_fevals_sd:.2f}")
        print(f"Success Rate (BFGS): {success_rate_bfgs:.2f}%")
        print(f"Success Rate (Steepest Descent): {success_rate_sd:.2f}%")
        print("-")

