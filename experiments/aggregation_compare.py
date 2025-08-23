"""
aggregation_compare.py

Compare three aggregation methods to form shared z from per-subject posteriors:
1) mean: simple average of subject means
2) precision: precision-weighted average using each subject's posterior variance
3) minvar: pick the subject with the lowest posterior variance per timepoint

Runs across lags, trains one SRM-VAE model per lag, evaluates encoding
performance (lag vs mean correlation) for each aggregation and saves results.
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

VAE_K       = 5
VAE_EPOCHS  = 1000
VAE_LR      = 1e-3
VAE_BETA    = 0.5
ORTHO_WEIGHT = 1e-3
PCA_DIM     = 50

OUTPUT_DIR = "./results/aggregation_compare"
# Number of repeated runs
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


def encode_all_subjects(model: core.SRMVAE, subject_views: Dict[int, core.SubjectView], split: str):
    mu_list, logvar_list = [], []
    for sid, view in subject_views.items():
        x = getattr(view, split)
        mu_i, logvar_i = model.encoders[str(sid)](x)
        mu_list.append(mu_i)
        logvar_list.append(logvar_i)
    # T x k tensors per subject stacked → list length S
    return mu_list, logvar_list


def aggregate_mean(mu_list: List, logvar_list: List):
    mu = sum(mu_list) / float(len(mu_list))
    return mu


def aggregate_precision(mu_list: List, logvar_list: List):
    # precision = 1/var = exp(-logvar)
    precisions = [(-lv).exp() for lv in logvar_list]
    # elementwise: mu = (sum p_i * mu_i) / (sum p_i)
    num = None
    denom = None
    for p, m in zip(precisions, mu_list):
        num = m * p if num is None else num + m * p
        denom = p if denom is None else denom + p
    mu = num / (denom + 1e-8)
    return mu


def aggregate_minvar(mu_list: List, logvar_list: List):
    # Choose per-timepoint subject with lowest mean variance across dims (all in torch)
    import torch
    device = mu_list[0].device
    variances = [lv.exp() for lv in logvar_list]  # list of [T, k]
    var_stack = torch.stack(variances, dim=0)     # [S, T, k]
    mu_stack = torch.stack(mu_list, dim=0)        # [S, T, k]
    mean_var = var_stack.mean(dim=2)              # [S, T]
    best_s = torch.argmin(mean_var, dim=0)        # [T]
    # Gather per timepoint
    T_len = mu_stack.shape[1]
    z = mu_stack.permute(1, 0, 2)[torch.arange(T_len, device=device), best_s, :]  # [T, k]
    return z


def evaluate_aggregations(batch: core.LagBatch, beta: float, epochs: int, lr: float, ortho_weight: float,
                          pca_dim: int, seed: int) -> Dict[str, float]:
    # Train one model
    vae = core.train_srmvae_on_batch(batch, epochs=epochs, lr=lr, beta=beta, verbose=True, ortho_weight=ortho_weight)
    vae.eval()

    # Encode per-subject posteriors
    mu_tr_list, lv_tr_list = encode_all_subjects(vae, batch.subject_views, split="train")
    mu_te_list, lv_te_list = encode_all_subjects(vae, batch.subject_views, split="test")

    # Build z trains/tests
    z_builders = {
        "mean": (aggregate_mean(mu_tr_list, lv_tr_list), aggregate_mean(mu_te_list, lv_te_list)),
        "precision": (aggregate_precision(mu_tr_list, lv_tr_list), aggregate_precision(mu_te_list, lv_te_list)),
        "minvar": (aggregate_minvar(mu_tr_list, lv_tr_list), aggregate_minvar(mu_te_list, lv_te_list)),
    }

    results: Dict[str, float] = {}
    # Embeddings pipeline
    X_train_raw = batch.X_train.cpu().numpy()
    X_test_raw  = batch.X_test.cpu().numpy()
    X_train_p, X_test_p, _ = prep_embeddings(X_train_raw, X_test_raw, pca_dim=pca_dim, seed=seed)

    for name, (z_tr_t, z_te_t) in z_builders.items():
        z_train = z_tr_t.detach().cpu().numpy()
        z_test  = z_te_t.detach().cpu().numpy()
        # Standardize dims
        z_scaler = StandardScaler(with_mean=True, with_std=True)
        z_train_std = z_scaler.fit_transform(z_train)
        z_test_std  = z_scaler.transform(z_test)
        # Linear regression X -> z
        reg = LinearRegression()
        reg.fit(X_train_p, z_train_std)
        z_hat_test_std = reg.predict(X_test_p)
        r_dims = colwise_pearsonr(z_test_std, z_hat_test_std)
        results[name] = float(np.nanmean(r_dims))
    return results


def save_csv_single(csv_path: str, rows: List[Tuple[int, Dict[str, float]]]) -> None:
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        headers = ["requested_lag_ms", "mean", "precision", "minvar"]
        w.writerow(headers)
        for lag, res in rows:
            w.writerow([lag, res.get("mean", np.nan), res.get("precision", np.nan), res.get("minvar", np.nan)])


def save_csv_mean(csv_path: str, x_requested: List[int], runs: Dict[str, dict]) -> None:
    import csv
    def sem(vals: List[float]) -> float:
        if not vals:
            return float('nan')
        arr = np.asarray(vals, dtype=float)
        if arr.size <= 1:
            return float('nan')
        return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))

    methods = ["mean", "precision", "minvar"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        headers = ["requested_lag_ms"]
        for m in methods:
            headers += [f"{m}_mean", f"{m}_sem"]
        w.writerow(headers)

        for lag in x_requested:
            row = [int(lag)]
            for m in methods:
                vals: List[float] = []
                for rk, rstate in runs.items():
                    results = rstate.get("results", {})
                    if str(lag) in results and m in results[str(lag)]:
                        vals.append(float(results[str(lag)][m]))
                if vals:
                    row += [float(np.nanmean(vals)), sem(vals)]
                else:
                    row += [float('nan'), float('nan')]
            w.writerow(row)


def main():
    core.set_global_seed(SEED, deterministic=True)
    try:
        Y_data, elec_num, X, lags = core.load_all_data(DATA_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return
    ensure_dir(OUTPUT_DIR)
    state_path = os.path.join(OUTPUT_DIR, "aggregation_compare.json")
    if os.path.exists(state_path):
        state = load_json(state_path)
        # Migrate old format without multi-run support
        if "runs" not in state:
            old_results = state.get("results", {})
            state["runs"] = {"1": {"results": old_results}}
            state.pop("results", None)
            state.setdefault("num_runs", NUM_RUNS)
            atomic_write_json(state_path, state)
            print("[MIGRATE] existing state to multi-run format")
    else:
        state = {
            "aggregation_methods": ["mean", "precision", "minvar"],
            "beta": float(VAE_BETA),
            "ortho_weight": float(ORTHO_WEIGHT),
            "k": int(VAE_K),
            "pca_dim": int(PCA_DIM),
            "train_ratio": float(TRAIN_RATIO),
            "seed": int(SEED),
            "x_requested": list(map(int, LAG_LIST)),
            "num_runs": int(NUM_RUNS),
            "runs": {},
        }
    state["num_runs"] = int(state.get("num_runs", NUM_RUNS))

    # Execute runs with resume
    for run_idx in range(1, state["num_runs"] + 1):
        run_key = str(run_idx)
        run_state = state["runs"].setdefault(run_key, {"results": {}})
        results_map: Dict[str, Dict[str, float]] = run_state["results"]

        core.set_global_seed(SEED + run_idx, deterministic=True)

        rows_single: List[Tuple[int, Dict[str, float]]] = []
        for req in LAG_LIST:
            if str(req) in results_map:
                print(f"[SKIP] run={run_idx} lag={req} already computed")
                rows_single.append((int(req), results_map[str(req)]))
                continue
            batch = core.build_lag_batch_from_loaded(
                Y_data=Y_data,
                elec_num=elec_num,
                X=X,
                lags=lags,
                lag_ms=req,
                latent_dim=VAE_K,
                train_ratio=TRAIN_RATIO,
            )
            print(f"[RUN] run={run_idx} lag req={req} → chosen={batch.lag_ms}")
            res = evaluate_aggregations(
                batch=batch,
                beta=VAE_BETA,
                epochs=VAE_EPOCHS,
                lr=VAE_LR,
                ortho_weight=ORTHO_WEIGHT,
                pca_dim=PCA_DIM,
                seed=SEED + run_idx,
            )
            results_map[str(req)] = res
            atomic_write_json(state_path, state)
            print("[SAVE] run={} lag={} saved -> {}".format(run_idx, req, state_path))
            rows_single.append((int(req), res))

        # per-run CSV
        run_csv = os.path.join(OUTPUT_DIR, f"aggregation_compare_run{run_idx}.csv")
        save_csv_single(run_csv, rows_single)
        print(f"[DONE] run={run_idx} CSV → {run_csv}")

    # mean/SEM CSV across runs
    mean_csv = os.path.join(OUTPUT_DIR, "aggregation_compare.csv")
    save_csv_mean(mean_csv, x_requested=state["x_requested"], runs=state["runs"])
    print(f"[DONE] mean/SEM CSV → {mean_csv}")


if __name__ == "__main__":
    main()


