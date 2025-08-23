"""
ortho_ablation.py

Sweep orthogonality penalty weights for SRM-VAE shared-space encoding.
Runs multiple repeats per weight with safe resume and saves per-run CSVs
and aggregated mean±SEM CSVs. Results can be plotted with
experiments/plot_experiment_curves.py (labels will show ortho_weight if present).
"""

import os
import json
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import LinearRegression

import sys
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_CURR_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
import srm_vs_vae_shared_encoding as core


# ========== CONFIG (edit here) ==========
DATA_PATH   = "./all_data.pkl"
SEED        = 1234
TRAIN_RATIO = 0.7

LAG_LIST = [-2000, -1000, -500, 0, 100, 200, 300, 500, 1000, 1500, 1800]

VAE_K      = 5
VAE_EPOCHS = 1000
VAE_LR     = 1e-3
PCA_DIM    = 50

# Orthogonality weight sweep
ORTHO_WEIGHTS = [1,1e-1]

OUTPUT_DIR = "./results/ortho_ablation"
NUM_RUNS   = 3
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


def shared_encoding_for_ortho(batch: core.LagBatch, k: int, pca_dim: int, seed: int,
                              beta: float, epochs: int, lr: float, ortho_weight: float) -> Tuple[float, List[float]]:
    vae = core.train_srmvae_on_batch(batch, epochs=epochs, lr=lr, beta=beta, verbose=True, ortho_weight=ortho_weight)
    vae.eval()
    z_tr, _, _ = vae.infer_z(batch.subject_views, split="train", use_mu=True)
    z_te, _, _ = vae.infer_z(batch.subject_views, split="test",  use_mu=True)
    z_train = z_tr.cpu().numpy()
    z_test  = z_te.cpu().numpy()
    z_scaler = StandardScaler(with_mean=True, with_std=True)
    z_train_std = z_scaler.fit_transform(z_train)
    z_test_std  = z_scaler.transform(z_test)
    X_train_raw = batch.X_train.cpu().numpy()
    X_test_raw  = batch.X_test.cpu().numpy()
    X_train_p, X_test_p, _ = prep_embeddings(X_train_raw, X_test_raw, pca_dim=pca_dim, seed=seed)
    reg = LinearRegression()
    reg.fit(X_train_p, z_train_std)
    z_hat_test_std = reg.predict(X_test_p)
    r_dims = colwise_pearsonr(z_test_std, z_hat_test_std)
    r_mean = float(np.nanmean(r_dims))
    return r_mean, [float(x) for x in r_dims]


def save_csv_single(csv_path: str, x_requested: List[int], lag_results: Dict[str, dict]) -> None:
    import csv
    rows = [(int(lag), lag_results[str(lag)]["actual_lag"], lag_results[str(lag)]["r_mean"]) for lag in x_requested if str(lag) in lag_results]
    rows.sort(key=lambda t: x_requested.index(t[0]))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["requested_lag_ms", "chosen_lag_ms", "r_mean"])
        for r in rows:
            w.writerow(r)


def save_csv_mean(csv_path: str, x_requested: List[int], runs: Dict[str, dict]) -> None:
    import csv
    def sem(vals: List[float]) -> float:
        if not vals:
            return float('nan')
        arr = np.asarray(vals, dtype=float)
        if arr.size <= 1:
            return float('nan')
        return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))
    rows = []
    for lag in x_requested:
        key = str(lag)
        vals = []
        for rk, rstate in runs.items():
            lr = rstate.get("lag_results", {})
            if key in lr:
                vals.append(float(lr[key]["r_mean"]))
        if vals:
            rows.append((int(lag), float(np.nanmean(vals)), sem(vals)))
    rows.sort(key=lambda t: x_requested.index(t[0]))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["requested_lag_ms", "r_mean_mean", "r_mean_sem"])
        for r in rows:
            w.writerow(r)


def run_ortho_ablation(Y_data: np.ndarray, elec_num: np.ndarray, X: np.ndarray, lags: np.ndarray,
                       ortho_weights: List[float], lag_list: List[int], train_ratio: float, k: int,
                       pca_dim: int, seed: int, epochs: int, lr: float, out_dir: str, num_runs: int) -> None:
    ensure_dir(out_dir)
    beta = 0.5  # use default unless you want to expose as sweep as well
    for ow in ortho_weights:
        ow_str = ("%g" % ow).replace(" ", "")
        json_path = os.path.join(out_dir, f"ortho_{ow_str}.json")

        if os.path.exists(json_path):
            state = load_json(json_path)
            print(f"[RESUME] ortho={ow} runs={len(state.get('runs', {}))} from {json_path}")
        else:
            state = {
                "ortho_weight": float(ow),
                "beta": float(beta),
                "k": int(k),
                "pca_dim": int(pca_dim),
                "train_ratio": float(train_ratio),
                "seed": int(seed),
                "epochs": int(epochs),
                "lr": float(lr),
                "x_requested": list(map(int, lag_list)),
                "num_runs": int(num_runs),
                "runs": {},
            }

        state["num_runs"] = int(state.get("num_runs", num_runs))

        for run_idx in range(1, state["num_runs"] + 1):
            run_key = str(run_idx)
            run_state = state["runs"].setdefault(run_key, {"lag_results": {}})
            lag_results = run_state["lag_results"]

            core.set_global_seed(seed + run_idx, deterministic=True)

            for req in lag_list:
                if str(req) in lag_results:
                    print(f"[SKIP] ortho={ow} run={run_idx} lag={req} already computed")
                    continue
                batch = core.build_lag_batch_from_loaded(
                    Y_data=Y_data,
                    elec_num=elec_num,
                    X=X,
                    lags=lags,
                    lag_ms=req,
                    latent_dim=k,
                    train_ratio=train_ratio,
                )
                print(f"[RUN] ortho={ow} run={run_idx} lag req={req} → chosen={batch.lag_ms}")
                r_mean, r_dims = shared_encoding_for_ortho(
                    batch=batch,
                    k=k,
                    pca_dim=pca_dim,
                    seed=seed + run_idx,
                    beta=beta,
                    epochs=epochs,
                    lr=lr,
                    ortho_weight=ow,
                )
                lag_results[str(req)] = {
                    "requested_lag": int(req),
                    "actual_lag": int(batch.lag_ms),
                    "r_mean": float(r_mean),
                    "r_dims": r_dims,
                }
                atomic_write_json(json_path, state)
                print(f"[SAVE] ortho={ow} run={run_idx} lag={req} saved → {json_path}")

            # per-run CSV
            run_csv = os.path.join(out_dir, f"ortho_{ow_str}_run{run_idx}.csv")
            save_csv_single(run_csv, x_requested=state["x_requested"], lag_results=lag_results)
            print(f"[DONE] ortho={ow} run={run_idx} CSV → {run_csv}")

        # mean CSV
        mean_csv = os.path.join(out_dir, f"ortho_{ow_str}.csv")
        save_csv_mean(mean_csv, x_requested=state["x_requested"], runs=state["runs"])
        print(f"[DONE] ortho={ow} mean/SEM CSV → {mean_csv}")


def main():
    core.set_global_seed(SEED, deterministic=True)
    try:
        Y_data, elec_num, X, lags = core.load_all_data(DATA_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return
    print(f"[DATA] Y={Y_data.shape}, X={X.shape}, lags_count={len(lags)}")
    print(f"[CONF] TRAIN_RATIO={TRAIN_RATIO}, VAE_K={VAE_K}, PCA_DIM={PCA_DIM}, EPOCHS={VAE_EPOCHS}, LR={VAE_LR}")
    print(f"[SWEEP] ortho_weights={ORTHO_WEIGHTS}")
    ensure_dir(OUTPUT_DIR)
    run_ortho_ablation(
        Y_data=Y_data,
        elec_num=elec_num,
        X=X,
        lags=lags,
        ortho_weights=ORTHO_WEIGHTS,
        lag_list=LAG_LIST,
        train_ratio=TRAIN_RATIO,
        k=VAE_K,
        pca_dim=PCA_DIM,
        seed=SEED,
        epochs=VAE_EPOCHS,
        lr=VAE_LR,
        out_dir=OUTPUT_DIR,
        num_runs=NUM_RUNS,
    )


if __name__ == "__main__":
    main()


