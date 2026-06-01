import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from particle_swarm_optimizer import ParticleSwarmOptimizer, PenaltyMethod, LogLevel
from bump_test_function import bump_test_function, inequality_constraint1, inequality_constraint2

# Problem setup
n_dimensions = 20
bounds = [(0, 10)] * n_dimensions
inequality_constraints = [inequality_constraint1, inequality_constraint2]
equality_constraints = []
independent_runs = 10
max_iters = 1000

# PSO hyperparameters
n_particles = 59
w = 0.406
c1 = 1.527
c2 = 2.135
penalty_method = PenaltyMethod.STATIC
penalty_method.set_static_penalty(7535)

# Hybrid parameters
k = 100  # refinement every k iterations
n_refine = 3  # top particles to refine with SLSQP

# Tracking stats
standard_pso_results = []
hybrid_pso_results = []

def run_standard_pso(seeds):
    best_vals = []
    histories = []

    for run, seed in enumerate(seeds):
        print(f"[Standard PSO] Run {run + 1}/{independent_runs} (seed={seed})")
        np.random.seed(seed)  # ensure reproducibility of particle positions/velocities
        pso = ParticleSwarmOptimizer(
            objective_func=bump_test_function,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            n_dimensions=n_dimensions,
            bounds=bounds,
            n_particles=n_particles,
            w=w,
            c1=c1,
            c2=c2,
            penalty_method=penalty_method,
            log_level=LogLevel.INFO,
        )

        x_best, f_best, history = pso.optimize(n_iterations=max_iters)
        best_vals.append(f_best)
        histories.append(history)

    return best_vals, histories

def run_hybrid_pso(seeds):
    best_vals = []
    histories = []

    for run, seed in enumerate(seeds):
        print(f"[Hybrid PSO–SLSQP] Run {run + 1}/{independent_runs} (seed={seed})")
        np.random.seed(seed)  # ensure reproducibility of particle positions/velocities
        pso = ParticleSwarmOptimizer(
            objective_func=bump_test_function,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            n_dimensions=n_dimensions,
            bounds=bounds,
            n_particles=n_particles,
            w=w,
            c1=c1,
            c2=c2,
            penalty_method=penalty_method,
            log_level=LogLevel.INFO,
        )

        history = []
        for t in range(max_iters):
            for i in range(n_particles):
                fitness = pso.evaluate(pso.positions[i], t)
                pso.update_personal_best(i, fitness)
                if fitness < pso.global_best_value:
                    pso.global_best_position = pso.positions[i]
                    pso.global_best_value = fitness
            pso.update_positions_and_velocities()
            history.append(pso.global_best_value)

            if (t + 1) % k == 0:
                top_indices = np.argsort(pso.personal_best_values)[:n_refine]
                for idx in top_indices:
                    result = minimize(
                        lambda x: pso.evaluate(x, t),
                        pso.personal_best_positions[idx],
                        bounds=bounds,
                        method='SLSQP',
                        options={'disp': False, 'maxiter': 1000}
                    )
                    if result.success and result.fun < pso.personal_best_values[idx]:
                        pso.personal_best_positions[idx] = result.x
                        pso.personal_best_values[idx] = result.fun
                        if result.fun < pso.global_best_value:
                            pso.global_best_position = result.x
                            pso.global_best_value = result.fun

        best_vals.append(pso.global_best_value)
        histories.append(history)

    return best_vals, histories


# === Run both algorithms ===
# Generate shared seeds for reproducibility
np.random.seed(42)
seeds = np.random.randint(0, 10_000, size=independent_runs)

# Run both algorithms using the same seeds
standard_vals, standard_histories = run_standard_pso(seeds)
hybrid_vals, hybrid_histories = run_hybrid_pso(seeds)

# === Print results ===
def summarize(title, values):
    print(f"\n{title}")
    print(f"Mean:  {np.mean(values):.6f}")
    print(f"Std:   {np.std(values):.6f}")
    print(f"Min:   {np.min(values):.6f}")
    print(f"Max:   {np.max(values):.6f}")

summarize("Standard PSO Results", standard_vals)
summarize("Hybrid PSO–SLSQP Results", hybrid_vals)

# === Plot convergence ===


def plot_convergence_comparison(hist_std, hist_hybrid, output):
    def pad_histories(histories):
        max_len = max(len(h) for h in histories)
        return np.array([h + [h[-1]] * (max_len - len(h)) for h in histories])

    hist_std = pad_histories(hist_std)
    hist_hybrid = pad_histories(hist_hybrid)

    mean_std = np.mean(hist_std, axis=0)
    std_std = np.std(hist_std, axis=0)
    mean_hybrid = np.mean(hist_hybrid, axis=0)
    std_hybrid = np.std(hist_hybrid, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300, sharey=True)

    # --- Standard PSO ---
    for run in hist_std:
        ax1.plot(run, alpha=0.3, linestyle='--', linewidth=1)
    ax1.plot(mean_std, color='black', label='Mean', linewidth=2)
    ax1.fill_between(
        np.arange(len(mean_std)), mean_std - 3 * std_std, mean_std + 3 * std_std,
        color='gray', alpha=0.2, label=r'$\pm 3\sigma$'
    )
    ax1.set_title("Standard PSO")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Objective Value")
    ax1.grid(True)
    ax1.legend()

    # --- Hybrid PSO–SLSQP ---
    for run in hist_hybrid:
        ax2.plot(run, alpha=0.3, linestyle='--', linewidth=1)
    ax2.plot(mean_hybrid, color='black', label='Mean', linewidth=2)
    ax2.fill_between(
        np.arange(len(mean_hybrid)), mean_hybrid - 3 * std_hybrid, mean_hybrid + 3 * std_hybrid,
        color='gray', alpha=0.2, label=r'$\pm 3\sigma$'
    )
    ax2.set_title("Hybrid PSO–SLSQP")
    ax2.set_xlabel("Iteration")
    ax2.grid(True)
    ax2.legend()

    fig.suptitle("Convergence Comparison: Standard PSO vs Hybrid PSO–SLSQP", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output)
    plt.close()
    print(f"Saved: {output}")

plot_convergence_comparison(standard_histories, hybrid_histories, "pso_hybrid_comparison.png")


