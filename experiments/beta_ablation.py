"""
beta_ablation.py

Runs SRM-VAE shared-space encoding across multiple beta values and lags.
Saves partial results after each lag (per beta) for safe resume and
later plotting of LAG vs CORRELATION per beta.

Outputs (per beta):
- results/beta_ablation/beta_{beta}.json: incremental JSON with all lag results
- results/beta_ablation/beta_{beta}.csv: final CSV summary when beta completes

Notes:
- Uses the core model and data utilities from srm_vs_vae_shared_encoding.py
"""

import os
import json
import math
import pickle
from typing import Dict, List, Tuple

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import LinearRegression

# Import core utilities and model from the existing script
import sys
# Ensure project root is importable when running from experiments/
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_CURR_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
import srm_vs_vae_shared_encoding as core


# ========== CONFIG (edit here) ==========
DATA_PATH   = "./all_data.pkl"
SEED        = 1234
TRAIN_RATIO = 0.7

# Lags to evaluate (ms)
LAG_LIST = [-2000, -1000, -500, 0, 100, 200, 300, 500, 1000, 1500, 1800]

# Model dims
VAE_K   = 5

# Training hyperparams (can be different from the core defaults if desired)
VAE_EPOCHS = 1000
VAE_LR     = 1e-3

# Embedding preprocessing
PCA_DIM = 50

# Beta sweep
BETAS = [0.1,0.5,1,2,3]

# Output directory
OUTPUT_DIR = "./results/beta_ablation2"
# Number of repeated runs per beta
NUM_RUNS = 3
# ========================================


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def atomic_write_json(path: str, obj: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prep_embeddings(X_train: np.ndarray, X_test: np.ndarray, pca_dim: int, seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit PCA on train only, then L2 row-normalize."""
    pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=seed)
    X_train_p = pca.fit_transform(X_train)
    X_test_p  = pca.transform(X_test)
    X_train_p = normalize(X_train_p, axis=1)
    X_test_p  = normalize(X_test_p, axis=1)
    return X_train_p, X_test_p, float(pca.explained_variance_ratio_.sum())


def colwise_pearsonr(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0)) + eps
    return num / den


def shared_encoding_for_beta(batch: core.LagBatch, k: int, pca_dim: int, seed: int,
                             beta: float, epochs: int, lr: float, ortho_weight: float = 1e-3) -> Tuple[float, List[float]]:
    """
    Train SRM-VAE on the provided lag batch with a specific beta and
    compute test correlation per shared dimension.
    Returns (mean_r, list_r_dims)
    """
    # Train
    vae = core.train_srmvae_on_batch(batch, epochs=epochs, lr=lr, beta=beta, verbose=True, ortho_weight=ortho_weight)
    vae.eval()

    # Infer z (μ) on train/test
    z_tr, _, _ = vae.infer_z(batch.subject_views, split="train", use_mu=True)
    z_te, _, _ = vae.infer_z(batch.subject_views, split="test",  use_mu=True)
    z_train = z_tr.cpu().numpy()
    z_test  = z_te.cpu().numpy()

    # Standardize per dim
    z_scaler = StandardScaler(with_mean=True, with_std=True)
    z_train_std = z_scaler.fit_transform(z_train)
    z_test_std  = z_scaler.transform(z_test)

    # Embeddings: center by train mean → PCA → L2
    X_train_raw = batch.X_train.cpu().numpy()
    X_test_raw  = batch.X_test.cpu().numpy()
    X_train_p, X_test_p, _ = prep_embeddings(X_train_raw, X_test_raw, pca_dim=pca_dim, seed=seed)

    # Linear regression: X -> z
    reg = LinearRegression()
    reg.fit(X_train_p, z_train_std)
    z_hat_test_std = reg.predict(X_test_p)

    r_dims = colwise_pearsonr(z_test_std, z_hat_test_std)
    r_mean = float(np.nanmean(r_dims))
    return r_mean, [float(x) for x in r_dims]


def save_csv_summary(csv_path: str, x_requested: List[int], lag_results: Dict[str, dict]) -> None:
    import csv
    rows = [(int(lag), lag_results[str(lag)]["actual_lag"], lag_results[str(lag)]["r_mean"]) for lag in x_requested if str(lag) in lag_results]
    rows.sort(key=lambda t: x_requested.index(t[0]))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["requested_lag_ms", "chosen_lag_ms", "r_mean"])
        for r in rows:
            w.writerow(r)


def save_csv_summary_mean(csv_path: str, x_requested: List[int], runs: Dict[str, dict]) -> None:
    """Save mean and SEM across runs for each requested lag.
    CSV columns: requested_lag_ms, r_mean_mean, r_mean_sem
    """
    import csv
    def compute_sem(values: List[float]) -> float:
        if not values:
            return float('nan')
        arr = np.asarray(values, dtype=float)
        if arr.size <= 1:
            return float('nan')
        return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))

    rows = []
    for lag in x_requested:
        key = str(lag)
        vals = []
        for rk, rstate in runs.items():
            lag_results = rstate.get("lag_results", {})
            if key in lag_results:
                vals.append(float(lag_results[key]["r_mean"]))
        if vals:
            mean_val = float(np.nanmean(vals))
            sem_val = compute_sem(vals)
            rows.append((int(lag), mean_val, sem_val))

    rows.sort(key=lambda t: x_requested.index(t[0]))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["requested_lag_ms", "r_mean_mean", "r_mean_sem"])
        for r in rows:
            w.writerow(r)


def run_beta_ablation(Y_data: np.ndarray, elec_num: np.ndarray, X: np.ndarray, lags: np.ndarray,
                      betas: List[float], lag_list: List[int], train_ratio: float, k: int,
                      pca_dim: int, seed: int, epochs: int, lr: float, out_dir: str) -> None:
    ensure_dir(out_dir)

    for beta in betas:
        beta_str = ("%g" % beta).replace(" ", "")
        json_path = os.path.join(out_dir, f"beta_{beta_str}.json")

        # Resume if exists
        if os.path.exists(json_path):
            state = load_json(json_path)
            # Migrate old single-run format to multi-run
            if "runs" not in state:
                lag_results = state.get("lag_results", {})
                state = {
                    "beta": float(beta),
                    "k": int(k),
                    "pca_dim": int(pca_dim),
                    "train_ratio": float(train_ratio),
                    "seed": int(seed),
                    "epochs": int(epochs),
                    "lr": float(lr),
                    "x_requested": list(map(int, lag_list)),
                    "num_runs": int(NUM_RUNS),
                    "runs": {"1": {"lag_results": lag_results}},
                }
                atomic_write_json(json_path, state)
                print(f"[MIGRATE] beta={beta} migrated old format to multi-run at {json_path}")
            print(f"[RESUME] beta={beta} loaded state with runs={len(state.get('runs', {}))} from {json_path}")
        else:
            state = {
                "beta": float(beta),
                "k": int(k),
                "pca_dim": int(pca_dim),
                "train_ratio": float(train_ratio),
                "seed": int(seed),
                "epochs": int(epochs),
                "lr": float(lr),
                "x_requested": list(map(int, lag_list)),
                "num_runs": int(NUM_RUNS),
                "runs": {},
            }
        # Ensure number of runs in state
        state["num_runs"] = int(state.get("num_runs", NUM_RUNS))

        # Execute runs
        for run_idx in range(1, state["num_runs"] + 1):
            run_key = str(run_idx)
            run_state = state["runs"].setdefault(run_key, {"lag_results": {}})
            lag_results = run_state["lag_results"]

            # Seed per-run for stochastic variation
            core.set_global_seed(seed + run_idx, deterministic=True)

            for req in lag_list:
                if str(req) in lag_results:
                    print(f"[SKIP] beta={beta} run={run_idx} lag={req} already computed.")
                    continue

                # Build batch for this lag
                batch = core.build_lag_batch_from_loaded(
                    Y_data=Y_data,
                    elec_num=elec_num,
                    X=X,
                    lags=lags,
                    lag_ms=req,
                    latent_dim=k,
                    train_ratio=train_ratio,
                )

                print(f"[RUN] beta={beta} run={run_idx} lag req={req} → chosen={batch.lag_ms}")
                r_mean, r_dims = shared_encoding_for_beta(
                    batch=batch,
                    k=k,
                    pca_dim=pca_dim,
                    seed=seed + run_idx,
                    beta=beta,
                    epochs=epochs,
                    lr=lr,
                    ortho_weight=1e-3,
                )

                lag_results[str(req)] = {
                    "requested_lag": int(req),
                    "actual_lag": int(batch.lag_ms),
                    "r_mean": float(r_mean),
                    "r_dims": r_dims,
                }

                # Save after each lag for safety
                atomic_write_json(json_path, state)
                print(f"[SAVE] beta={beta} run={run_idx} lag={req} saved → {json_path}")

            # Per-run CSV
            run_csv = os.path.join(out_dir, f"beta_{beta_str}_run{run_idx}.csv")
            save_csv_summary(run_csv, x_requested=state["x_requested"], lag_results=lag_results)
            print(f"[DONE] beta={beta} run={run_idx} CSV → {run_csv}")

        # Mean/SEM CSV across runs
        mean_csv = os.path.join(out_dir, f"beta_{beta_str}.csv")
        save_csv_summary_mean(mean_csv, x_requested=state["x_requested"], runs=state["runs"])
        print(f"[DONE] beta={beta} mean/SEM CSV → {mean_csv}")


def main():
    core.set_global_seed(SEED, deterministic=True)

    try:
        Y_data, elec_num, X, lags = core.load_all_data(DATA_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    print(f"[DATA] Y={Y_data.shape}, X={X.shape}, lags_count={len(lags)}")
    print(f"[CONF] TRAIN_RATIO={TRAIN_RATIO}, VAE_K={VAE_K}, PCA_DIM={PCA_DIM}, EPOCHS={VAE_EPOCHS}, LR={VAE_LR}")
    print(f"[SWEEP] betas={BETAS}")

    run_beta_ablation(
        Y_data=Y_data,
        elec_num=elec_num,
        X=X,
        lags=lags,
        betas=BETAS,
        lag_list=LAG_LIST,
        train_ratio=TRAIN_RATIO,
        k=VAE_K,
        pca_dim=PCA_DIM,
        seed=SEED,
        epochs=VAE_EPOCHS,
        lr=VAE_LR,
        out_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()


