# vae_single_lag.py
# Train & evaluate the SRM-VAE on a SINGLE lag (default 20 ms).
# - No CLI, no folds. Simple time split by TRAIN_RATIO.
# - Prints reconstruction MSE per subject and shared-space encoding r.

# ========= CONFIG (edit here) =========
DATA_PATH   = "./all_data.pkl"   # path to your all_data.pkl
SEED        = 1234               # random seed
TRAIN_RATIO = 0.8                # first 80% train, last 20% test
LAG_MS      = 20                 # target lag (ms); uses closest available lag in the file

LATENT_K    = 5                # VAE shared latent dims
EPOCHS      = 5000
LR          =5e-4
BETA        = 2               # β-VAE weight

# For shared-space encoding evaluation (X -> z)
PCA_DIM     = 50                 # reduce embeddings to 50D then L2 row-normalize
VERBOSE     = True
SAVE_NPZ    = None               # e.g., "vae_single_lag_results.npz" or None
# =====================================

import os
import sys
import math
import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    print("ERROR: PyTorch not found. Install with: pip install torch")
    raise

# SciKit
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import LinearRegression


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
    """Time × Electrodes for one subject, already train/test split & normalized."""
    train: torch.Tensor  # [T_train, E_i]
    test:  torch.Tensor  # [T_test , E_i]
    mask_train: Optional[torch.Tensor] = None  # [E_i] (optional)
    mask_test:  Optional[torch.Tensor] = None  # [E_i] (optional)

@dataclass
class LagBatch:
    """Container for a single lag across all subjects and a simple time split."""
    lag_ms: int
    latent_dim: int
    subjects: List[int]
    subject_views: Dict[int, SubjectView]
    X_train: torch.Tensor               # [T_train, D_emb]
    X_test:  torch.Tensor               # [T_test , D_emb]
    train_index: np.ndarray
    test_index:  np.ndarray
    elec_num: Dict[int, int]


# -----------------------------
# I/O and preprocessing
# -----------------------------
def load_all_data(pkl_path: str):
    """
    Expected keys:
      - 'electrode_data': [S, L, T, Emax]
      - 'electrode_number': [S]
      - 'word_embeddings': [T, D_emb]
      - 'lags': [L] (ms)
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    Y_data = np.asarray(obj["electrode_data"])            # [S, L, T, Emax]
    elec_num = np.asarray(obj["electrode_number"], int)   # [S]
    X = np.asarray(obj["word_embeddings"])                # [T, D_emb]
    lags = np.asarray(obj["lags"]).reshape(-1)            # [L]
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
    n_train = min(n_train, T - 1)  # ensure at least 1 test
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
    """Build a single-lag batch with z-scoring per subject and centered embeddings."""
    S, L, T, Emax = Y_data.shape
    lag_idx = choose_lag_index(lags, lag_ms)
    chosen_ms = int(lags[lag_idx])

    train_index, test_index = time_split_indices(T, train_ratio)

    # Embeddings: center by train mean only
    X_train = X[train_index, :]
    X_test  = X[test_index, :]
    X_train = X_train - X_train.mean(axis=0, keepdims=True)
    X_test  = X_test  - X_train.mean(axis=0, keepdims=True)

    subjects = list(range(1, S + 1))
    subject_views: Dict[int, SubjectView] = {}
    per_sub_elec: Dict[int, int] = {}

    for s in range(S):
        e_i = int(elec_num[s])
        mat = Y_data[s, lag_idx, :, :e_i]   # [T, E_i]
        tr = mat[train_index, :]
        te = mat[test_index, :]
        tr_z, te_z = _zscore_train_apply(tr, te)
        mask = np.ones((e_i,), dtype=bool)

        subject_views[subjects[s]] = SubjectView(
            train=torch.from_numpy(tr_z).float(),
            test=torch.from_numpy(te_z).float(),
            mask_train=torch.from_numpy(mask),
            mask_test=torch.from_numpy(mask),
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


# -----------------------------
# SRM-VAE model
# -----------------------------
class PerSubjectEncoder(nn.Module):
    """Linear encoder per subject: R^{E_i} -> (mu, logvar) in R^k"""
    def __init__(self, e_i: int, k: int):
        super().__init__()
        self.lin = nn.Linear(e_i, 2 * k, bias=True)
        nn.init.xavier_uniform_(self.lin.weight)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.lin(x)  # [T, 2k]
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar.clamp(min=-8.0, max=8.0)

class PerSubjectDecoder(nn.Module):
    """Linear decoder per subject: R^k -> R^{E_i}"""
    def __init__(self, k: int, e_i: int):
        super().__init__()
        self.lin = nn.Linear(k, e_i, bias=False)
        nn.init.xavier_uniform_(self.lin.weight)
    def forward(self, zf: torch.Tensor) -> torch.Tensor:
        return self.lin(zf)

class SRMVAE(nn.Module):
    """
    Encoders per subject -> precision-weighted group posterior -> shared core f(z)
    -> decoders per subject (SRM-like).
    """
    def __init__(self, elec_num: Dict[int, int], k: int):
        super().__init__()
        self.k = k
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for sid, e_i in elec_num.items():
            self.encoders[str(sid)] = PerSubjectEncoder(e_i, k)
            self.decoders[str(sid)] = PerSubjectDecoder(k, e_i)
        self.core = nn.Sequential(nn.Linear(k, k), nn.ReLU(), nn.Linear(k, k))
        for m in self.core:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def _agg_posteriors(mu_list, logvar_list):
        precisions = [torch.exp(-lv) for lv in logvar_list]
        prec_sum = torch.stack(precisions, dim=0).sum(dim=0)
        weighted_mu = torch.stack([m * p for m, p in zip(mu_list, precisions)], dim=0).sum(dim=0)
        var = 1.0 / (prec_sum + 1e-8)
        mu = var * weighted_mu
        logvar = torch.log(var + 1e-8)
        return mu, logvar

    def encode_group(self, subject_views: Dict[int, SubjectView], split: str):
        mu_list, logvar_list = [], []
        for sid, view in subject_views.items():
            x = getattr(view, split)  # [T,E_i]
            mu_i, logvar_i = self.encoders[str(sid)](x)
            mu_list.append(mu_i)
            logvar_list.append(logvar_i)
        return self._agg_posteriors(mu_list, logvar_list)

    def forward(self, subject_views: Dict[int, SubjectView], split: str, beta: float = 1.0):
        mu, logvar = self.encode_group(subject_views, split)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        zf = self.core(z)
        # tensor (not python float) for stability
        recon_loss = torch.tensor(0.0, device=z.device)
        for sid, view in subject_views.items():
            x = getattr(view, split)           # [T,E_i]
            x_hat = self.decoders[str(sid)](zf)
            recon_loss = recon_loss + F.mse_loss(x_hat, x, reduction='mean')
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl
        return loss, recon_loss.detach(), kl.detach()

    @torch.no_grad()
    def infer_z(self, subject_views: Dict[int, SubjectView], split: str, use_mu: bool = True):
        mu, logvar = self.encode_group(subject_views, split)
        if use_mu:
            z = mu
        else:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return z, mu, logvar

    @torch.no_grad()
    def reconstruct_subjects(self, z: torch.Tensor) -> Dict[int, torch.Tensor]:
        zf = self.core(z)
        return {int(sid): dec(zf) for sid, dec in self.decoders.items()}


# -----------------------------
# Training
# -----------------------------
def train_srmvae_on_batch(batch: LagBatch, epochs: int, lr: float, beta: float, verbose: bool = True) -> SRMVAE:
    dev = torch_device()
    for sid in batch.subjects:
        sv = batch.subject_views[sid]
        sv.train = sv.train.to(dev)
        sv.test  = sv.test.to(dev)
        if sv.mask_train is not None: sv.mask_train = sv.mask_train.to(dev)
        if sv.mask_test  is not None: sv.mask_test  = sv.mask_test.to(dev)

    model = SRMVAE(batch.elec_num, k=batch.latent_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = math.inf
    best_state = None
    patience = 10
    no_imp = 0

    for ep in range(1, epochs + 1):
        model.train()
        loss, rec, kl = model(batch.subject_views, split="train", beta=beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        curr = float(loss.item())
        if curr < best_loss - 1e-5:
            best_loss = curr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if verbose and (ep % 25 == 0 or ep == 1):
            print(f"[ep {ep:03d}] loss={curr:.5f}  rec={float(rec):.5f}  kl={float(kl):.5f}  no_imp={no_imp}")

        if no_imp >= patience:
            if verbose:
                print(f"[early stop] best_loss={best_loss:.5f} at ep≈{ep-no_imp}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -----------------------------
# Evaluation helpers
# -----------------------------
def _prep_embeddings(X_train: np.ndarray, X_test: np.ndarray, pca_dim: int, seed: int):
    """PCA -> L2 normalize rows. Fit PCA on train only."""
    pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=seed)
    X_train_p = pca.fit_transform(X_train)
    X_test_p  = pca.transform(X_test)
    X_train_p = normalize(X_train_p, axis=1)
    X_test_p  = normalize(X_test_p, axis=1)
    return X_train_p, X_test_p, float(pca.explained_variance_ratio_.sum())

def _colwise_pearsonr(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8):
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0)) + eps
    return num / den

def eval_encoding_PCA_Ridge(batch: LagBatch, vae: SRMVAE, pca_dim: int = 50, seed: int = SEED):
    """Shared-space encoding: regress PCA+L2(X) -> standardized μ(z) on train; eval on test."""
    dev = torch_device()
    vae.eval()
    with torch.no_grad():
        z_tr, _, _ = vae.infer_z(batch.subject_views, split="train", use_mu=True)
        z_te, _, _ = vae.infer_z(batch.subject_views, split="test",  use_mu=True)
    z_train = z_tr.cpu().numpy()  # [T_train, k]
    z_test  = z_te.cpu().numpy()  # [T_test,  k]

    # Standardize z (train stats only)
    z_scaler = StandardScaler(with_mean=True, with_std=True)
    z_train_std = z_scaler.fit_transform(z_train)
    z_test_std  = z_scaler.transform(z_test)

    # Embeddings: PCA(pca_dim) + L2 per row
    X_train_raw = batch.X_train.cpu().numpy()
    X_test_raw  = batch.X_test.cpu().numpy()
    X_train_p, X_test_p, pca_var = _prep_embeddings(X_train_raw, X_test_raw, pca_dim=pca_dim, seed=seed)

    # Linear regression X -> z
    reg = LinearRegression()
    reg.fit(X_train_p, z_train_std)
    z_hat_test_std = reg.predict(X_test_p)

    r_dims = _colwise_pearsonr(z_test_std, z_hat_test_std)
    r_mean = float(np.nanmean(r_dims))
    return {
        "r_z_mean": r_mean,
        "r_z_dims": r_dims,
        "pca_explained": pca_var,
    }


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    set_global_seed(SEED, deterministic=True)
    dev = torch_device()
    print(f"[INFO] torch: {torch.__version__}, cuda: {torch.cuda.is_available()}, device: {dev}")
    print(f"[INFO] DATA_PATH={DATA_PATH} | LATENT_K={LATENT_K} | TRAIN_RATIO={TRAIN_RATIO} | LAG_MS(target)={LAG_MS}")

    try:
        Y_data, elec_num, X, lags = load_all_data(DATA_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

    # Build single-lag batch (closest to LAG_MS)
    batch = build_lag_batch_from_loaded(
        Y_data=Y_data,
        elec_num=elec_num,
        X=X,
        lags=lags,
        lag_ms=LAG_MS,
        latent_dim=LATENT_K,
        train_ratio=TRAIN_RATIO,
    )
    print(f"[OK] using lag_ms={batch.lag_ms} (closest to {LAG_MS})")
    s1 = batch.subjects[0]
    print(f"     S{s1} train shape: {tuple(batch.subject_views[s1].train.shape)}")
    print(f"     X_train: {tuple(batch.X_train.shape)}  X_test: {tuple(batch.X_test.shape)}")

    # Train VAE
    print(f"[TRAIN] SRM-VAE on lag={batch.lag_ms} ms, k={batch.latent_dim}")
    vae = train_srmvae_on_batch(batch, epochs=EPOCHS, lr=LR, beta=BETA, verbose=VERBOSE)

    # Inference & quick reconstruction MSEs on test
    vae.eval()
    with torch.no_grad():
        z_te, _, _ = vae.infer_z(batch.subject_views, split="test", use_mu=True)
        recons_test = vae.reconstruct_subjects(z_te.to(dev))
        for sid in batch.subjects:
            x_true = batch.subject_views[sid].test.to(dev)
            x_hat  = recons_test[sid]
            mse = F.mse_loss(x_hat, x_true).item()
            print(f"[TEST] Subject {sid}: recon MSE = {mse:.6f}")

    # Shared-space encoding eval (X -> z)
    print(f"[EVAL] PCA({PCA_DIM})+Linear encoding to z")
    enc = eval_encoding_PCA_Ridge(batch, vae, pca_dim=PCA_DIM, seed=SEED)
    print(f"[ENC] Shared-space r (mean over k={LATENT_K}): {enc['r_z_mean']:.4f}  | PCA var={enc['pca_explained']:.2f}")

    if SAVE_NPZ:
        np.savez(
            SAVE_NPZ,
            lag_ms=batch.lag_ms,
            r_z_mean=enc["r_z_mean"],
            r_z_dims=enc["r_z_dims"],
            latent_k=LATENT_K,
            train_ratio=TRAIN_RATIO,
            pca_dim=PCA_DIM,
        )
        print(f"[SAVE] Wrote metrics to {SAVE_NPZ}")
