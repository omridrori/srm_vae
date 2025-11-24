# train_and_save_model.py
# - Trains the deep, non-linear, group-reconstruction VAE for a *single* lag.
# - Saves the trained model weights to a .pt file for later inspection.

# ========= CONFIG (edit here) =========
DATA_PATH       = "./all_data.pkl"  # Path relative to the project root
MODEL_SAVE_PATH = "./debugging/inspection_values/trained_vae_model.pt"
TARGET_LAG_MS   = 200

# --- Model & Training Hyperparams
SEED            = 1234
TRAIN_RATIO     = 0.8
VAE_K           = 5
VAE_EPOCHS      = 1000  # Adjust as needed for convergence
VAE_LR          = 1e-3
VAE_BETA        = 0.1
VAE_HIDDEN_DIMS = [128, 64]
RECON_WEIGHT    = 2
# =====================================

import os
import sys
import math
import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- Torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    print("ERROR: PyTorch not found. Install with: pip install torch")
    sys.exit(1)

# -----------------------------
# Reproducibility utilities
# -----------------------------
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

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class SubjectView:
    train: torch.Tensor
    test:  torch.Tensor

@dataclass
class LagBatch:
    lag_ms: int
    latent_dim: int
    subjects: List[int]
    subject_views: Dict[int, SubjectView]
    elec_num: Dict[int, int]

# -----------------------------
# I/O and preprocessing
# -----------------------------
def load_all_data(pkl_path: str):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    Y_data = np.asarray(obj["electrode_data"])
    elec_num = np.asarray(obj["electrode_number"], int)
    lags = np.asarray(obj["lags"]).reshape(-1)
    return Y_data, elec_num, lags

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
    Y_data: np.ndarray, elec_num: np.ndarray, lags: np.ndarray,
    lag_ms: int, latent_dim: int, train_ratio: float,
) -> LagBatch:
    S, L, T, Emax = Y_data.shape
    lag_idx = choose_lag_index(lags, lag_ms)
    chosen_ms = int(lags[lag_idx])
    train_index, test_index = time_split_indices(T, train_ratio)
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
        lag_ms=chosen_ms, latent_dim=latent_dim, subjects=subjects,
        subject_views=subject_views, elec_num=per_sub_elec,
    )

# -----------------------------
# SRM-VAE model
# -----------------------------
class PerSubjectEncoder(nn.Module):
    def __init__(self, e_i: int, k: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        input_dim = e_i
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 2 * k))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar.clamp(min=-8.0, max=8.0)

class PerSubjectDecoder(nn.Module):
    def __init__(self, k: int, e_i: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        input_dim = k
        for hidden_dim in reversed(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, e_i))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, zf: torch.Tensor) -> torch.Tensor:
        return self.net(zf)

class SRMVAE(nn.Module):
    def __init__(self, elec_num: Dict[int, int], k: int, hidden_dims: List[int]):
        super().__init__()
        self.k = k
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for sid, e_i in elec_num.items():
            self.encoders[str(sid)] = PerSubjectEncoder(e_i, k, hidden_dims)
            self.decoders[str(sid)] = PerSubjectDecoder(k, e_i, hidden_dims)
        self.core = nn.Identity()

    def forward_group_recon(self, subject_views: Dict[int, SubjectView], 
                            source_sid: int, recon_weight: float, beta: float):
        source_view = subject_views[source_sid]
        mu, logvar = self.encoders[str(source_sid)](source_view.train)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        zf = self.core(z)
        
        total_recon_loss = torch.tensor(0.0, device=z.device)
        num_subjects = len(subject_views)
        for target_sid, target_view in subject_views.items():
            target_recon = self.decoders[str(target_sid)](zf)
            loss_i = F.mse_loss(target_recon, target_view.train, reduction='mean')
            total_recon_loss += loss_i
        
        avg_recon_loss = total_recon_loss / num_subjects
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_weight * avg_recon_loss + beta * kl
        return total_loss, avg_recon_loss.detach(), kl.detach()

# -----------------------------
# Training
# -----------------------------
def train_srmvae_on_batch(batch: LagBatch, epochs: int, lr: float, beta: float, 
                          recon_weight: float, hidden_dims: List[int]) -> SRMVAE:
    dev = torch_device()
    for sid in batch.subjects:
        sv = batch.subject_views[sid]
        sv.train = sv.train.to(dev)
        sv.test  = sv.test.to(dev)

    model = SRMVAE(batch.elec_num, k=batch.latent_dim, hidden_dims=hidden_dims).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = math.inf
    best_state = None
    subject_ids = list(batch.subjects)
    num_subjects = len(subject_ids)

    print(f"\n--- Starting training for lag={batch.lag_ms}ms ---")
    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_recon, epoch_kl = 0, 0, 0
        source_sids = list(subject_ids)
        random.shuffle(source_sids)

        for source_sid in source_sids:
            beta_ep = beta * min(1.0, ep / 500)
            loss, recon, kl = model.forward_group_recon(
                batch.subject_views, source_sid, recon_weight, beta_ep
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()

        avg_loss = epoch_loss / num_subjects
        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 25 == 0 or ep == 1 or ep == epochs:
            print(f"[Ep {ep:04d}/{epochs}] loss={avg_loss:.5f}  "
                  f"recon={epoch_recon/num_subjects:.5f}  "
                  f"kl={epoch_kl/num_subjects:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print("--- Training finished ---")
    return model

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    set_global_seed(SEED, deterministic=True)
    dev = torch_device()
    print(f"[INFO] Using device: {dev}")

    try:
        print("[INFO] Loading data... this may take a while.")
        Y_data, elec_num, lags = load_all_data(DATA_PATH)
        print("[INFO] Data loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load data from {DATA_PATH}: {e}")
        sys.exit(1)

    print(f"[INFO] Building data batch for lag={TARGET_LAG_MS}ms")
    batch = build_lag_batch_from_loaded(
        Y_data=Y_data, elec_num=elec_num, lags=lags,
        lag_ms=TARGET_LAG_MS, latent_dim=VAE_K, train_ratio=TRAIN_RATIO
    )

    trained_model = train_srmvae_on_batch(
        batch=batch, epochs=VAE_EPOCHS, lr=VAE_LR, beta=VAE_BETA,
        recon_weight=RECON_WEIGHT, hidden_dims=VAE_HIDDEN_DIMS
    )
    
    # --- Save the trained model ---
    try:
        save_obj = {
            'config': {
                'lag_ms': batch.lag_ms,
                'k': VAE_K,
                'hidden_dims': VAE_HIDDEN_DIMS,
                'elec_num': batch.elec_num,
            },
            'model_state_dict': trained_model.state_dict(),
        }
        torch.save(save_obj, MODEL_SAVE_PATH)
        print(f"[SUCCESS] Model saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")
        sys.exit(1)
