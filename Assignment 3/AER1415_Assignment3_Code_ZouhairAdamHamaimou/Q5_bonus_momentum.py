#!/usr/bin/env python3
"""
Q5 – BONUS (momentum)
=====================
Mini‑batch SGD **with momentum** versus
plain SGD and the CG baseline on the NASA air‑foil noise data set.

• Network: 5‑64‑1 (tanh), identical initialisation to baseline
• 3×3 hyper‑parameter grid (batch‑size × learning‑rate)
• Momentum coef β = 0.9  (can be changed via CLI)
• 10 independent seeds
• Caches every configuration; re‑runs only missing ones
"""

from __future__ import annotations
import argparse, csv, time
from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import matplotlib.pyplot as plt

from minimize import minimize           # Conjugate‑gradient (baseline)
from loss import loss                    # objective + grad
from model_predict import model_predict  # inference


# --------------------------------------------------------------------- #
# Utility functions                                                     #
# --------------------------------------------------------------------- #
def init_theta(d: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Xavier‑like init (same as baseline)."""
    W1 = np.sqrt((5 / 3) / d) * rng.standard_normal((d, m))
    W2 = np.sqrt(1 / m) * rng.standard_normal(m)
    b1 = np.zeros(m)
    b2 = 0.0
    return np.concatenate([W1.ravel(), b1, W2, [b2]])


def mse(theta: np.ndarray,
        Xtr: np.ndarray, ytr: np.ndarray,
        Xte: np.ndarray, yte: np.ndarray,
        d: int, m: int) -> Tuple[float, float]:
    """train/test MSE of parameters θ."""
    yhat_tr = model_predict(theta, Xtr, d, m)
    yhat_te = model_predict(theta, Xte, d, m)
    return float(np.mean((ytr - yhat_tr) ** 2)), float(np.mean((yte - yhat_te) ** 2))


# --------------------------------------------------------------------- #
# SGDs                                                                  #
# --------------------------------------------------------------------- #
def plain_sgd(theta0: np.ndarray,
              X: np.ndarray, y: np.ndarray,
              d: int, m: int, *,
              batch_size: int, lr: float,
              epochs: int,
              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant‑step mini‑batch SGD."""
    X, y = X.copy(), y.copy()
    N = len(y)
    theta = theta0.copy()
    L_hist, G_hist = np.empty(epochs), np.empty(epochs)

    for ep in range(epochs):
        perm = rng.permutation(N)
        X, y = X[perm], y[perm]
        for s in range(0, N, batch_size):
            Xb, yb = X[s:s+batch_size], y[s:s+batch_size]
            _, g = loss(theta, Xb, yb, d, m)
            theta -= lr * g
        L, g_full = loss(theta, X, y, d, m)
        L_hist[ep] = L
        G_hist[ep] = np.linalg.norm(g_full)
    return theta, L_hist, G_hist


def sgd_mom(theta0: np.ndarray,
            X: np.ndarray, y: np.ndarray,
            d: int, m: int, *,
            batch_size: int, lr: float, beta: float,
            epochs: int,
            rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mini‑batch SGD with classical momentum (Polyak, 1964)."""
    X, y = X.copy(), y.copy()
    N = len(y)
    theta = theta0.copy()
    v = np.zeros_like(theta)          # momentum buffer
    L_hist, G_hist = np.empty(epochs), np.empty(epochs)

    for ep in range(epochs):
        perm = rng.permutation(N)
        X, y = X[perm], y[perm]
        for s in range(0, N, batch_size):
            Xb, yb = X[s:s+batch_size], y[s:s+batch_size]
            _, g = loss(theta, Xb, yb, d, m)
            v = beta * v + g          # momentum update
            theta -= lr * v
        L, g_full = loss(theta, X, y, d, m)
        L_hist[ep] = L
        G_hist[ep] = np.linalg.norm(g_full)
    return theta, L_hist, G_hist


def pad_histories(histories):
    max_len = max(len(h) for h in histories)
    return np.array([np.pad(h, (0, max_len - len(h)), mode='edge') for h in histories])


# --------------------------------------------------------------------- #
# Experiment                                                            #
# --------------------------------------------------------------------- #
def experiment(beta: float = .9) -> None:
    tic = time.time()

    # data ----------------------------------------------------------------
    print("Loading NASA air‑foil noise data …")
    train = np.loadtxt("airfoil_training_data.dat")
    test = np.loadtxt("airfoil_testing_data.dat")
    Xtr, ytr = train[:, :5], train[:, 5]
    Xte, yte = test[:, :5], test[:, 5]
    mu, sig = Xtr.mean(0), Xtr.std(0)
    Xtr, Xte = (Xtr - mu) / sig, (Xte - mu) / sig
    print(f"  • training {len(ytr)}  • test {len(yte)}\n")

    d, m = 5, 64
    epochs, seeds = 1000, range(10)
    batch_sizes = [16]
    lrs = [1e-2]

    pairs = [(bs, lr) for lr in lrs for bs in batch_sizes]

    # global caches -------------------------------------------------------
    cache_loss = Path("bonus_loss_histories.npy")
    cache_grad = Path("bonus_grad_histories.npy")
    losses: Dict[Tuple[str, int, float], List[np.ndarray]] = (
        np.load(cache_loss, allow_pickle=True).item()
        if cache_loss.exists() else {}
    )
    grads: Dict[Tuple[str, int, float], List[np.ndarray]] = (
        np.load(cache_grad, allow_pickle=True).item()
        if cache_grad.exists() else {}
    )
    results = []

    # -------------------------- SGD w/ momentum -------------------------
    for k, (bs, lr) in enumerate(pairs, 1):
        key = ("MOM", bs, lr, beta)
        if key in losses:
            print(f"[{k}/{len(pairs)}] cached MOM bs={bs} lr={lr:.0e}")
            # Recover cached statistics
            cache_data = losses[key]
            loss_runs = cache_data["losses"]
            grad_runs = cache_data["grads"]
            tr_mse = cache_data["train_mse"]
            te_mse = cache_data["test_mse"]
            results.append((key[0], key[1], key[2],
                np.mean(tr_mse), np.std(tr_mse),
                np.mean(te_mse), np.std(te_mse)))
            continue

        print(f"[{k}/{len(pairs)}] training MOM bs={bs} lr={lr:.0e}")
        loss_runs, grad_runs, tr_mse, te_mse = [], [], [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            theta0 = init_theta(d, m, rng)
            theta, L, G = sgd_mom(theta0, Xtr, ytr, d, m,
                                  batch_size=bs, lr=lr, beta=beta,
                                  epochs=epochs, rng=rng)
            loss_runs.append(L); grad_runs.append(G)
            mse_tr, mse_te = mse(theta, Xtr, ytr, Xte, yte, d, m)
            tr_mse.append(mse_tr); te_mse.append(mse_te)
            print(f"   seed {seed:2d} → train={mse_tr:7.4f}  test={mse_te:7.4f}")

        # Save everything as a dictionary per key
        cache_data = {
            "losses": loss_runs,
            "grads": grad_runs,
            "train_mse": tr_mse,
            "test_mse": te_mse,
        }

        losses[key] = cache_data  # store full dictionarylosses[key], grads[key] = loss_runs, grad_runs
        results.append(("MOM", bs, lr,
                        np.mean(tr_mse), np.std(tr_mse),
                        np.mean(te_mse), np.std(te_mse)))

    # --------------------- plain SGD (reuse or compute) -----------------
    for k, (bs, lr) in enumerate(pairs, 1):
        key = ("SGD", bs, lr)
        if key in losses:
            print(f"[plain SGD] cached bs={bs} lr={lr:.0e}")
            # Recover cached statistics
            cache_data = losses[key]
            loss_runs = cache_data["losses"]
            grad_runs = cache_data["grads"]
            tr_mse = cache_data["train_mse"]
            te_mse = cache_data["test_mse"]
            results.append((key[0], key[1], key[2],
                np.mean(tr_mse), np.std(tr_mse),
                np.mean(te_mse), np.std(te_mse)))
            continue


        print(f"[plain SGD] training bs={bs} lr={lr:.0e}")
        loss_runs, grad_runs, tr_mse, te_mse = [], [], [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            theta0 = init_theta(d, m, rng)
            theta, L, G = plain_sgd(theta0, Xtr, ytr, d, m,
                                    batch_size=bs, lr=lr,
                                    epochs=epochs, rng=rng)
            loss_runs.append(L); grad_runs.append(G)
            mse_tr, mse_te = mse(theta, Xtr, ytr, Xte, yte, d, m)
            tr_mse.append(mse_tr); te_mse.append(mse_te)
        cache_data = {
            "losses": loss_runs,
            "grads": grad_runs,
            "train_mse": tr_mse,
            "test_mse": te_mse,
        }

        losses[key] = cache_data  # store full dictionarylosses[key], grads[key] = loss_runs, grad_runs
        results.append(("SGD", bs, lr,
                        np.mean(tr_mse), np.std(tr_mse),
                        np.mean(te_mse), np.std(te_mse)))

    # -------------------------- CG baseline -----------------------------
    if ("CG", 0, 0.0) in losses:
        print("Cached CG baseline …")
        # Recover cached statistics
        cache_data = losses[("CG", 0, 0.0)]
        loss_runs = cache_data["losses"]
        grad_runs = cache_data["grads"]
        tr_mse = cache_data["train_mse"]
        te_mse = cache_data["test_mse"]
        results.append(("CG", 0, 0.0,
            np.mean(tr_mse), np.std(tr_mse),
            np.mean(te_mse), np.std(te_mse)))
    else:
        print("Running CG baseline …")
        loss_runs, grad_runs, tr_mse, te_mse = [], [], [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            theta0 = init_theta(d, m, rng)
            theta, L, G, _ = minimize(theta0, loss,
                                      args=(Xtr, ytr, d, m),
                                      maxnumlinesearch=None,
                                      maxnumfuneval=1000,
                                      verbose=False)
            loss_runs.append(L); grad_runs.append(G)
            mse_tr, mse_te = mse(theta, Xtr, ytr, Xte, yte, d, m)
            tr_mse.append(mse_tr); te_mse.append(mse_te)
        cache_data = {
            "losses": loss_runs,
            "grads": grad_runs,
            "train_mse": tr_mse,
            "test_mse": te_mse,
        }

        losses[("CG", 0, 0.0)] = cache_data  # store full dictionarylosses[key], grads[key] = loss_runs, grad_runs
        results.append(("CG", 0, 0.0,
                        np.mean(tr_mse), np.std(tr_mse),
                        np.mean(te_mse), np.std(te_mse)))

    # save global caches --------------------------------------------------
    np.save(cache_loss, losses, allow_pickle=True)
    print("\nGlobal cache updated.\n")

    # ---------------------- write CSV summary ---------------------------
    with open("results_q5_bonus.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "batch", "lr",
                    "train_mse_mu", "train_mse_sd",
                    "test_mse_mu", "test_mse_sd"])
        for row in results:
            w.writerow(row)
    print("Saved table   → results_q5_bonus.csv")

    # ---------------------- print summary table -------------------------
    print("Summary table →")
    print("method   batch      lr     train_mse_mu  train_mse_sd  test_mse_mu  test_mse_sd")
    print("-------  -----  --------  ------------  ------------  -----------  -----------")
    for row in results:
        print(f"{row[0]:<7}  {row[1]:<5}  {row[2]:>8.1e}  "
              f"{row[3]:>12.4f}  {row[4]:>12.4f}  "
              f"{row[5]:>11.4f}  {row[6]:>11.4f}")
    print("---------------------------------------------------------------")

    # ---------------------- plot loss histories --------------------------
    lr_choice = 1e-2
    fig, ax = plt.subplots()
    for bs in batch_sizes:
        mu_mom = np.mean(losses[("MOM", bs, lr_choice, beta)]["losses"], 0)
        mu_sgd = np.mean(losses[("SGD", bs, lr_choice)]["losses"], 0)
        ax.semilogy(mu_mom, label=f"SGD w/ momentum", lw=2)
        ax.semilogy(mu_sgd, "--", label=f"SGD", lw=1.5)

    # Add CG baseline
    cg_losses_padded = pad_histories(losses[("CG", 0, 0.0)]["losses"])
    mu_cg = np.mean(cg_losses_padded, axis=0)
    ax.semilogy(mu_cg, "-.", color='black', label="CG (baseline)", lw=2)

    ax.set_ylim(1, 1e2)
    ax.set_xlabel("Iteration / Function Evaluation")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("momentum_vs_sgd.png", dpi=300)
    plt.show()

    # ---------------------- plot gradient histories --------------------------
    fig, ax = plt.subplots()
    for bs in batch_sizes:
        mu_mom = np.mean(losses[("MOM", bs, lr_choice, beta)]["grads"], 0)
        mu_sgd = np.mean(losses[("SGD", bs, lr_choice)]["grads"], 0)
        ax.semilogy(mu_mom, label=f"SGD w/ momentum", lw=2)
        ax.semilogy(mu_sgd, "--", label=f"SGD", lw=1.5)

    # Add CG baseline
    cg_grads_padded = pad_histories(losses[("CG", 0, 0.0)]["grads"])
    mu_cg_grads = np.mean(cg_grads_padded, axis=0)
    ax.semilogy(mu_cg, "-.", color='black', label="CG (baseline)", lw=2)

    ax.set_ylim(1, 1e2)
    ax.set_xlabel("Iteration / Function Evaluation")
    ax.set_ylabel(r"$\|\nabla \mathcal{L}\|_2$")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("gradnorm_vs_sgd_mom_cg.png", dpi=300)
    plt.show()

    print("Saved gradient norms plot → gradnorm_vs_sgd_mom_cg.png")


    print(f"\nWall‑time: {time.time() - tic:.1f} s")


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Q5‑bonus – SGD with momentum")
    ap.add_argument("--beta", type=float, default=0.9,
                    help="momentum coefficient (default 0.9)")
    args = ap.parse_args()
    experiment(beta=args.beta)
