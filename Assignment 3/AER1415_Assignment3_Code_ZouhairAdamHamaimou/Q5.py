#!/usr/bin/env python3
"""
AER1415 – Assignment 3, Question 5 (b,c)
========================================
Mini‑batch SGD versus CG on the NASA air‑foil noise data set.

• Three batch‑sizes  {16, 32, 64}
• Three learning‑rates {1e‑3, 1e‑2, 1e‑1}
• 10 random seeds per configuration
• Trains a 5‑64‑1 tanh network (same init as baseline script)
• Caches every run; re‑runs ONLY missing hyper‑parameter pairs
• Builds 3×3 semilog grids (loss / ‖∇L‖₂) + CSV table
"""

from __future__ import annotations
import csv, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from minimize import minimize
from loss import loss
from model_predict import model_predict


# --------------------------------------------------------------------- #
# Helper utilities                                                       #
# --------------------------------------------------------------------- #
def init_theta(d: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Xavier‑like initialisation (identical to baseline script)."""
    W1 = np.sqrt((5 / 3) / d) * rng.standard_normal((d, m))
    W2 = np.sqrt(1 / m) * rng.standard_normal(m)
    b1 = np.zeros(m)
    b2 = 0.0
    return np.concatenate([W1.ravel(), b1, W2, [b2]])


def runfile(bs: int, lr: float) -> str:
    """Unique file name for a (batch_size, learning_rate) pair."""
    return f"runs_bs{bs}_lr{lr:.0e}.npy"


def evaluate(
    theta: np.ndarray,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    d: int,
    m: int,
) -> Tuple[float, float]:
    """Return train/test MSE for given parameters."""
    yhat_tr = model_predict(theta, Xtr, d, m)
    yhat_te = model_predict(theta, Xte, d, m)
    return float(np.mean((ytr - yhat_tr) ** 2)), float(np.mean((yte - yhat_te) ** 2))


# --------------------------------------------------------------------- #
# Mini‑batch SGD                                                        #
# --------------------------------------------------------------------- #
def sgd(
    theta0: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    d: int,
    m: int,
    *,
    batch_size: int,
    lr: float,
    epochs: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant‑step‑size SGD; returns θ*, per‑epoch loss and grad‑norm."""
    X = X.copy()
    y = y.copy()
    N = len(y)
    theta = theta0.copy()
    losses = np.empty(epochs)
    grads = np.empty(epochs)

    for ep in range(epochs):
        # fresh permutation each epoch
        perm = rng.permutation(N)
        X, y = X[perm], y[perm]

        # mini‑batch updates
        for start in range(0, N, batch_size):
            stop = min(start + batch_size, N)
            Xb, yb = X[start:stop], y[start:stop]
            _, grad = loss(theta, Xb, yb, d, m)
            theta -= lr * grad

        lval, gfull = loss(theta, X, y, d, m)
        losses[ep] = lval
        grads[ep] = np.linalg.norm(gfull)

    return theta, losses, grads


# --------------------------------------------------------------------- #
# Main experiment                                                       #
# --------------------------------------------------------------------- #
def main() -> None:
    tic = time.time()

    # ----------------------------- data --------------------------------
    print("Loading NASA air‑foil noise data …")
    train = np.loadtxt("airfoil_training_data.dat")
    test = np.loadtxt("airfoil_testing_data.dat")
    Xtr, ytr = train[:, :5], train[:, 5]
    Xte, yte = test[:, :5], test[:, 5]
    mu, sig = Xtr.mean(0), Xtr.std(0)
    Xtr = (Xtr - mu) / sig
    Xte = (Xte - mu) / sig
    print(f"  • training samples: {len(ytr)}   • test samples: {len(yte)}\n")

    # ------------------------- experiment setup ------------------------
    d, m = 5, 64
    epochs, seeds = 1000, range(10)
    batch_sizes = [16, 32, 64]
    lrs = [1e-4, 1e-3, 1e-2]
    pairs = [(bs, lr) for lr in lrs for bs in batch_sizes]

    # ---------- global history dictionaries (load or start fresh) ------
    hist_loss_file = Path("loss_histories.npy")
    hist_grad_file = Path("grad_histories.npy")
    if hist_loss_file.exists() and hist_grad_file.exists():
        loss_histories: Dict[Tuple[int, float], List[np.ndarray]] = np.load(
            hist_loss_file, allow_pickle=True
        ).item()
        grad_histories: Dict[Tuple[int, float], List[np.ndarray]] = np.load(
            hist_grad_file, allow_pickle=True
        ).item()
        print("↪ Loaded global loss/grad history caches.\n")
    else:
        loss_histories, grad_histories = {}, {}
        print("↪ No global caches found – starting from scratch.\n")

    results = []  # for CSV + pretty table later

    # ---------------------- loop over SGD pairs ------------------------
    for idx, (bs, lr) in enumerate(pairs, 1):
        tag = f"bs{bs}_lr{lr:.0e}"
        fname = Path(runfile(bs, lr))

        if fname.exists():
            print(f"[{idx}/{len(pairs)}] ↪ loading cached file {fname.name}")
            cache = np.load(fname, allow_pickle=True).item()
            loss_histories[(bs, lr)] = cache["losses"]
            grad_histories[(bs, lr)] = cache["grads"]
            tr_mse, te_mse = cache["train_mse"], cache["test_mse"]

        else:
            print(f"[{idx}/{len(pairs)}] training   {tag}")
            loss_runs, grad_runs, tr_mse, te_mse = [], [], [], []

            for seed in seeds:
                rng = np.random.default_rng(seed)
                theta0 = init_theta(d, m, rng)
                theta, losses, grads = sgd(
                    theta0,
                    Xtr,
                    ytr,
                    d,
                    m,
                    batch_size=bs,
                    lr=lr,
                    epochs=epochs,
                    rng=rng,
                )
                loss_runs.append(losses)
                grad_runs.append(grads)
                mse_tr, mse_te = evaluate(theta, Xtr, ytr, Xte, yte, d, m)
                tr_mse.append(mse_tr)
                te_mse.append(mse_te)
                print(
                    f"   seed {seed:2d} → train MSE={mse_tr:7.4f}, "
                    f"test MSE={mse_te:7.4f}"
                )

            # save pair‑wise cache
            np.save(
                fname,
                dict(
                    losses=loss_runs,
                    grads=grad_runs,
                    train_mse=tr_mse,
                    test_mse=te_mse,
                ),
            )
            loss_histories[(bs, lr)] = loss_runs
            grad_histories[(bs, lr)] = grad_runs

            # immediate plots (not saved)
            fig, ax = plt.subplots()
            for run in loss_runs:
                ax.semilogy(run, alpha=0.6)
            ax.set_title(f"Loss – {tag}")
            ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
            plt.show()

            fig, ax = plt.subplots()
            for run in grad_runs:
                ax.semilogy(run, alpha=0.6)
            ax.set_title(r"$\|\nabla\mathcal{L}\|_2$ – " + tag)
            ax.set_xlabel("epoch"); ax.set_ylabel(r"$\|\nabla\mathcal{L}\|_2$")
            plt.show()

        # ----- statistics for this pair ---------------------------------
        results.append(
            (
                f"SGD {tag}",
                float(np.mean(tr_mse)),
                float(np.std(tr_mse)),
                float(np.mean(te_mse)),
                float(np.std(te_mse)),
            )
        )

    # --------------------------- CG baseline ---------------------------
    cg_file = Path("cg_runs.npy")
    if cg_file.exists():
        print("\nLoading cached CG runs …")
        cg_cache = np.load(cg_file, allow_pickle=True).item()
    else:
        print("\nRunning CG baseline …")
        cg_cache = dict(losses=[], grads=[], train_mse=[], test_mse=[])
        for seed in seeds:
            rng = np.random.default_rng(seed)
            theta0 = init_theta(d, m, rng)
            theta, obj_hist, grad_hist, _ = minimize(
                theta0,
                loss,
                args=(Xtr, ytr, d, m),
                maxnumlinesearch=None,
                maxnumfuneval=1000,
                verbose=False,
            )
            cg_cache["losses"].append(obj_hist)
            cg_cache["grads"].append(grad_hist)
            mse_tr, mse_te = evaluate(theta, Xtr, ytr, Xte, yte, d, m)
            cg_cache["train_mse"].append(mse_tr)
            cg_cache["test_mse"].append(mse_te)
            print(
                f"   seed {seed:2d} → train MSE={mse_tr:7.4f}, "
                f"test MSE={mse_te:7.4f}"
            )
        np.save(cg_file, cg_cache)

    loss_histories[("CG", 0)] = cg_cache["losses"]
    grad_histories[("CG", 0)] = cg_cache["grads"]

    results.append(
        (
            "CG (baseline)",
            float(np.mean(cg_cache["train_mse"])),
            float(np.std(cg_cache["train_mse"])),
            float(np.mean(cg_cache["test_mse"])),
            float(np.std(cg_cache["test_mse"])),
        )
    )

    # ------------------- save / update global caches -------------------
    np.save(hist_loss_file, loss_histories, allow_pickle=True)
    np.save(hist_grad_file, grad_histories, allow_pickle=True)
    print("\nGlobal history files updated.\n")

    # ----------------------- 3×1 overlay grids -------------------------

    colors = {
        16: "tab:blue",
        32: "tab:green",
        64: "tab:red",
    }
    labels = {
        16: r"$B=16$",
        32: r"$B=32$",
        64: r"$B=64$",
    }
    cg_color = "black"
    cg_label = "CG (baseline)"

    # --- Loss Plot (3 rows = 3 learning rates)
    figL, axsL = plt.subplots(1, len(lrs), figsize=(15, 6), sharey=True)
    figG, axsG = plt.subplots(1, len(lrs), figsize=(15, 6), sharey=True)

    for c, lr in enumerate(lrs):
        axL = axsL[c]
        axG = axsG[c]

        # ---- Overlay CG run (same axis for all lrs) ----

        cg_losses = loss_histories[("CG", 0)]
        cg_grads = grad_histories[("CG", 0)]

        ep_cg  = [np.arange(len(run)) for run in cg_losses]  # variable length


        # CG overlay (average over available steps)
        max_len = max(len(run) for run in cg_losses)
        L_padded = np.full((len(cg_losses), max_len), np.nan)
        G_padded = np.full((len(cg_grads), max_len), np.nan)

        for i, (loss_run, grad_run) in enumerate(zip(cg_losses, cg_grads)):
            L_padded[i, :len(loss_run)] = loss_run
            G_padded[i, :len(grad_run)] = grad_run

        L_mu_cg = np.nanmean(L_padded, axis=0)
        G_mu_cg = np.nanmean(G_padded, axis=0)
        ep_cg = np.arange(max_len)

        axL.semilogy(ep_cg, L_mu_cg, color=cg_color, lw=2.5, linestyle="--", label=cg_label, alpha=0.7)
        axG.semilogy(ep_cg, G_mu_cg, color=cg_color, lw=2.5, linestyle="--", label=cg_label, alpha=0.7)

        # ---- Overlay SGD runs (3 batch sizes) ----

        for bs in batch_sizes:
            # Stack runs for current (bs, lr)
            L_runs = np.vstack(loss_histories[(bs, lr)])
            G_runs = np.vstack(grad_histories[(bs, lr)])

            # Compute mean curves
            L_mu, G_mu = L_runs.mean(0), G_runs.mean(0)
            ep = np.arange(len(L_mu))

            # Plot
            axL.semilogy(ep, L_mu, lw=2, label=labels[bs], color=colors[bs])
            axG.semilogy(ep, G_mu, lw=2, label=labels[bs], color=colors[bs])

        # ---- Cosmetic details ----

        axL.set_title(rf"Loss Convergence ($\eta={lr}$)")
        axG.set_title(rf"Gradient Norm Convergence ($\eta={lr}$)")
        axL.set_xlabel("Number of Function Evaluations (CG) / Epoch (SGD)")
        axG.set_xlabel("Number of Function Evaluations (CG) / Epoch (SGD)")
        axL.legend()
        axG.legend()

    axsL[0].set_ylabel("MSE")
    axsG[0].set_ylabel(r"$\|\nabla \mathcal{L}\|_2$")
    figL.tight_layout()
    figG.tight_layout()

    figL.savefig("loss_overlay_grid.png", dpi=300)
    figG.savefig("gradnorm_overlay_grid.png", dpi=300)
    print("Saved 3x1 overlay plots: loss_overlay_grid.png & gradnorm_overlay_grid.png")



    # ----------------------- CSV & pretty table ------------------------
    with open("results_q5.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["method", "train_mse_mean", "train_mse_std", "test_mse_mean", "test_mse_std"]
        )
        for row in results:
            w.writerow(row)
    print("Saved results table to results_q5.csv\n")

    print("Summary over 10 seeds")
    print(f"{'Method':<20} {'Train MSE':>14} {'±σ':>7} {'Test MSE':>14} {'±σ':>7}")
    for name, mu_tr, sd_tr, mu_te, sd_te in results:
        print(f"{name:<20} {mu_tr:14.6f} {sd_tr:7.6f} {mu_te:14.6f} {sd_te:7.6f}")

    print(f"\nTotal wall‑time: {time.time() - tic:.1f} s")


if __name__ == "__main__":
    main()
