# inspect_latent_predictions.py
# - Loads a previously trained cross-reconstruction SRM-VAE (single lag).
# - Trains a regression from word embeddings to the shared latent space using **test** data only.
# - For selected test samples, prints:
#       * per-subject encoded latents
#       * the averaged latent (regression target)
#       * regression prediction
#       * decoder outputs from the regression-predicted latent.

# ========= CONFIG (edit here) =========
DATA_PATH        = "./all_data.pkl"
MODEL_PATH       = "models/cross_recon_model_lag_200.pt"
TARGET_LAG_MS    = 200
TRAIN_RATIO      = 0.8
VAE_K            = 5      # must match latent dim used when training/saving the model
NUM_SAMPLES      = 5      # number of test timepoints to inspect
SEED             = 1234
SHUFFLE_SAMPLES  = True   # random sample test points (True) or take first NUM_SAMPLES (False)
PRINT_DIM_LIMIT  = 8      # number of dims/electrodes to print per vector for readability
# =====================================

import os
import sys
import math
import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    print("ERROR: PyTorch not found. Install with: pip install torch")
    sys.exit(1)

try:
    from sklearn.linear_model import LinearRegression
except Exception:
    print("ERROR: scikit-learn not found. Install with: pip install scikit-learn")
    sys.exit(1)


def set_global_seed(seed: int = 1234, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SubjectView:
    train: torch.Tensor
    test: torch.Tensor


@dataclass
class LagBatch:
    lag_ms: int
    latent_dim: int
    subjects: List[int]
    subject_views: Dict[int, SubjectView]
    X_train: torch.Tensor
    X_test: torch.Tensor
    train_index: np.ndarray
    test_index: np.ndarray
    elec_num: Dict[int, int]


def load_all_data(pkl_path: str):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    Y_data = np.asarray(obj["electrode_data"])
    elec_num = np.asarray(obj["electrode_number"], int)
    X = np.asarray(obj["word_embeddings"])
    lags = np.asarray(obj["lags"]).reshape(-1)
    return Y_data, elec_num, X, lags


def _zscore_train_apply(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mu) / sd, (test - mu) / sd


def choose_lag_index(lags_ms: np.ndarray, target_ms: int) -> int:
    diffs = np.abs(lags_ms - target_ms)
    return int(np.argmin(diffs))


def time_split_indices(T: int, train_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    n_train = max(1, int(round(T * train_ratio)))
    n_train = min(n_train, T - 1)
    return np.arange(0, n_train, dtype=int), np.arange(n_train, T, dtype=int)


def build_lag_batch_from_loaded(
    Y_data: np.ndarray,
    elec_num: np.ndarray,
    X: np.ndarray,
    lags: np.ndarray,
    lag_ms: int,
    latent_dim: int,
    train_ratio: float,
) -> LagBatch:
    S, L, T, Emax = Y_data.shape
    lag_idx = choose_lag_index(lags, lag_ms)
    chosen_ms = int(lags[lag_idx])
    train_index, test_index = time_split_indices(T, train_ratio)

    X_train = X[train_index, :]
    X_test = X[test_index, :]
    X_train = X_train - X_train.mean(axis=0, keepdims=True)
    X_test = X_test - X_train.mean(axis=0, keepdims=True)

    subjects = list(range(1, S + 1))
    subject_views: Dict[int, SubjectView] = {}
    per_sub_elec: Dict[int, int] = {}

    for s in range(S):
        e_i = int(elec_num[s])
        mat = Y_data[s, lag_idx, :, :e_i]
        tr = mat[train_index, :]
        te = mat[test_index, :]
        tr_z, te_z = _zscore_train_apply(tr, te)

        subject_views[subjects[s]] = SubjectView(
            train=torch.from_numpy(tr_z).float(),
            test=torch.from_numpy(te_z).float(),
        )
        per_sub_elec[subjects[s]] = e_i

    return LagBatch(
        lag_ms=chosen_ms,
        latent_dim=latent_dim,
        subjects=subjects,
        subject_views=subject_views,
        X_train=torch.from_numpy(X_train).float(),
        X_test=torch.from_numpy(X_test).float(),
        train_index=train_index,
        test_index=test_index,
        elec_num=per_sub_elec,
    )


class PerSubjectEncoder(nn.Module):
    def __init__(self, e_i: int, k: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(e_i, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * k)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar.clamp(min=-8.0, max=8.0)


class PerSubjectDecoder(nn.Module):
    def __init__(self, k: int, e_i: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, e_i)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, zf: torch.Tensor) -> torch.Tensor:
        return self.net(zf)


class SRMVAE(nn.Module):
    def __init__(self, elec_num: Dict[int, int], k: int, hidden_dim: int):
        super().__init__()
        self.k = k
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for sid, e_i in elec_num.items():
            self.encoders[str(sid)] = PerSubjectEncoder(e_i, k, hidden_dim)
            self.decoders[str(sid)] = PerSubjectDecoder(k, e_i, hidden_dim)
        self.core = nn.Identity()


def summarize_vector(vec: np.ndarray, limit: int) -> str:
    if vec.ndim == 1:
        arr = vec[:limit]
    else:
        arr = vec.reshape(-1)[:limit]
    return np.array2string(arr, precision=4, suppress_small=True)


if __name__ == '__main__':
    set_global_seed(SEED, deterministic=True)
    dev = torch_device()
    print(f"[INFO] Using device: {dev}")

    print("[INFO] Loading data...")
    try:
        Y_data, elec_num, X, lags = load_all_data(DATA_PATH)
    except Exception as e:
        print(f"[ERROR] Unable to load data: {e}")
        sys.exit(1)
    print("[INFO] Data loaded.")

    print(f"[INFO] Building lag batch for {TARGET_LAG_MS}ms")
    batch = build_lag_batch_from_loaded(
        Y_data=Y_data,
        elec_num=elec_num,
        X=X,
        lags=lags,
        lag_ms=TARGET_LAG_MS,
        latent_dim=VAE_K,
        train_ratio=TRAIN_RATIO,
    )

    print(f"[INFO] Loading trained model from {MODEL_PATH}")
    try:
        saved = torch.load(MODEL_PATH, map_location='cpu')
    except Exception as e:
        print(f"[ERROR] Unable to load model: {e}")
        sys.exit(1)

    model = SRMVAE(batch.elec_num, k=saved['config']['k'], hidden_dim=saved['config']['hidden_dim']).to(dev)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()
    print("[INFO] Model loaded.")

    for sid in batch.subjects:
        batch.subject_views[sid].train = batch.subject_views[sid].train.to(dev)
        batch.subject_views[sid].test = batch.subject_views[sid].test.to(dev)

    print("[INFO] Encoding test data for each subject...")
    mu_by_subject = {}
    with torch.no_grad():
        for sid in batch.subjects:
            sv = batch.subject_views[sid]
            mu, logvar = model.encoders[str(sid)](sv.test)
            mu_by_subject[sid] = mu.cpu().numpy()

    z_avg = np.mean(np.stack(list(mu_by_subject.values()), axis=0), axis=0)
    print(f"[INFO] Averaged latent shape: {z_avg.shape}")

    X_test_np = batch.X_test.cpu().numpy()

    print("[INFO] Training regression on test data (embeddings -> averaged latent)...")
    reg = LinearRegression()
    reg.fit(X_test_np, z_avg)
    print("[INFO] Regression training complete.")

    total_test = X_test_np.shape[0]
    num_samples = min(NUM_SAMPLES, total_test)
    if num_samples <= 0:
        print("[WARN] No test samples available for inspection.")
        sys.exit(0)

    if SHUFFLE_SAMPLES:
        rng = np.random.default_rng(SEED)
        inspect_indices = np.sort(rng.choice(total_test, size=num_samples, replace=False))
    else:
        inspect_indices = np.arange(num_samples)

    print(f"\n[INFO] Inspecting {len(inspect_indices)} test samples...")

    for local_idx in inspect_indices:
        global_time = int(batch.test_index[local_idx])
        word_vec = X_test_np[local_idx]

        print(f"\n{'-'*60}")
        print(f"Test sample (global time idx {global_time}, local test idx {local_idx})")
        print(f"Word embedding (first {PRINT_DIM_LIMIT} dims): {summarize_vector(word_vec, PRINT_DIM_LIMIT)}")

        for sid in batch.subjects:
            subj_latent = mu_by_subject[sid][local_idx]
            subj_original = batch.subject_views[sid].test[local_idx].cpu().numpy()
            print(f"Subject {sid} original (first {PRINT_DIM_LIMIT} electrodes): "
                  f"{summarize_vector(subj_original, PRINT_DIM_LIMIT)}")
            print(f"Subject {sid} encoded mu: {summarize_vector(subj_latent, PRINT_DIM_LIMIT)}")

        avg_latent = z_avg[local_idx]
        print(f"Average latent (regression target): {summarize_vector(avg_latent, PRINT_DIM_LIMIT)}")

        pred_latent = reg.predict(word_vec[None, :])[0]
        print(f"Regression prediction: {summarize_vector(pred_latent, PRINT_DIM_LIMIT)}")

        pred_tensor = torch.from_numpy(pred_latent).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            for sid in batch.subjects:
                decoded = model.decoders[str(sid)](pred_tensor).cpu().numpy().squeeze(0)
                print(f"Decoded (subject {sid}) first {PRINT_DIM_LIMIT} electrodes: {summarize_vector(decoded, PRINT_DIM_LIMIT)}")

    print("\n[INFO] Inspection complete.")

