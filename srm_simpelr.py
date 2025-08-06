# srm_vae_simple.py
# Single-file, no folds, no CLI. Modular and easy to follow.

# ====== CONFIG (edit here) ======
DATA_PATH   = "./all_data.pkl"  # path to your all_data.pkl
SEED        = 1234              # random seed
LATENT_K    = 5                 # shared latent dimensionality
TRAIN_RATIO = 0.8               # time-based split ratio (e.g., first 80% train, last 20% test)
LAG_MS      = 0                 # choose lag in milliseconds; if not present, closest lag is used
EPOCHS      = 10000
LR          = 1e-3
BETA        = 0.5             # β-VAE weight
PCA_DIM     = 50                # PCA dim for encoding evaluation
VERBOSE     = True
# =================================

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
    raise

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import RidgeCV


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
    subjects: List[int]                 # e.g., [1..S]
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
    Reads the single pickle created in your codebase.
    Expected keys:
      - 'electrode_data': shape [S, L, T, Emax]
      - 'electrode_number': length S
      - 'word_embeddings': shape [T, D_emb]
      - 'lags': length L (ms)
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    Y_data = obj["electrode_data"]         # [S, L, T, Emax]
    elec_num = obj["electrode_number"]     # [S]
    word_embeddings = obj["word_embeddings"]  # [T, D_emb]
    lags = obj["lags"]                     # [L]
    return Y_data, elec_num, word_embeddings, lags

def _zscore_train_apply(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score by train stats only; epsilon to avoid div-by-zero."""
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mu) / sd, (test - mu) / sd

def choose_lag_index(lags_ms: np.ndarray, target_ms: int) -> int:
    """Return index of the lag closest to target_ms."""
    diffs = np.abs(lags_ms - target_ms)
    return int(np.argmin(diffs))

def time_split_indices(T: int, train_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simple contiguous time split: first part train, last part test."""
    n_train = max(1, int(round(T * train_ratio)))
    n_train = min(n_train, T - 1)  # ensure at least 1 test sample
    train_index = np.arange(0, n_train, dtype=int)
    test_index = np.arange(n_train, T, dtype=int)
    return train_index, test_index


# -----------------------------
# Build a single-lag batch
# -----------------------------
def make_lag_batch(
    pkl_path: str,
    lag_ms: int,
    latent_dim: int,
    train_ratio: float,
) -> LagBatch:
    """
    Loads data, selects one lag (closest to lag_ms), time-splits embeddings,
    z-scores per subject using train stats, and returns a LagBatch.
    """
    Y_data, elec_num, X, lags = load_all_data(pkl_path)
    Y_data = np.asarray(Y_data)
    elec_num = np.asarray(elec_num, dtype=int)
    X = np.asarray(X)
    lags = np.asarray(lags).reshape(-1)

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

    batch = LagBatch(
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
    return batch


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
        # x: [T, E_i]
        h = self.lin(x)  # [T, 2k]
        mu, logvar = torch.chunk(h, 2, dim=-1)  # [T,k], [T,k]
        return mu, logvar.clamp(min=-8.0, max=8.0)

class PerSubjectDecoder(nn.Module):
    """Linear decoder per subject: R^k -> R^{E_i}"""
    def __init__(self, k: int, e_i: int):
        super().__init__()
        self.lin = nn.Linear(k, e_i, bias=False)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, zf: torch.Tensor) -> torch.Tensor:
        return self.lin(zf)  # [T, E_i]

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
        self.core = nn.Sequential(
            nn.Linear(k, k), nn.ReLU(),
            nn.Linear(k, k),
        )
        for m in self.core:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def _agg_posteriors(mu_list, logvar_list):
        """Precision-weighted aggregation across subjects per timepoint."""
        precisions = [torch.exp(-lv) for lv in logvar_list]   # [T,k]
        prec_sum = torch.stack(precisions, dim=0).sum(dim=0)  # [T,k]
        weighted_mu = torch.stack([m * p for m, p in zip(mu_list, precisions)], dim=0).sum(dim=0)
        var = 1.0 / (prec_sum + 1e-8)
        mu = var * weighted_mu
        logvar = torch.log(var + 1e-8)
        return mu, logvar

    def encode_group(self, subject_views: Dict[int, SubjectView], split: str):
        mu_list, logvar_list = [], []
        for sid, view in subject_views.items():
            x = getattr(view, split)  # [T, E_i]
            mu_i, logvar_i = self.encoders[str(sid)](x)
            mu_list.append(mu_i)
            logvar_list.append(logvar_i)
        mu, logvar = self._agg_posteriors(mu_list, logvar_list)
        return mu, logvar

    def forward(self, subject_views: Dict[int, SubjectView], split: str, beta: float = 1.0):
        mu, logvar = self.encode_group(subject_views, split)  # [T,k]
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps               # [T,k]
        zf = self.core(z)                                     # [T,k]

        recon_loss = 0.0
        for sid, view in subject_views.items():
            x = getattr(view, split)                          # [T,E_i]
            x_hat = self.decoders[str(sid)](zf)               # [T,E_i]
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
        out = {}
        for sid, dec in self.decoders.items():
            out[int(sid)] = dec(zf)
        return out


# -----------------------------
# Training
# -----------------------------
def train_srmvae_on_batch(
    batch: LagBatch,
    epochs: int = 300,
    lr: float = 1e-3,
    beta: float = 1.0,
    verbose: bool = True,
) -> SRMVAE:
    dev = torch_device()
    # move tensors to device in-place
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
    patience = 1000
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
# Evaluation: shared z and electrode space
# -----------------------------
def _colwise_pearsonr(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8):
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0)) + eps
    return num / den

@torch.no_grad()
def eval_encoding_PCA_Ridge(batch: LagBatch, vae: SRMVAE, pca_dim: int = 50):
    dev = torch_device()

    # --- latent z (μ) ---
    z_tr, _, _ = vae.infer_z(batch.subject_views, split="train", use_mu=True)
    z_te, _, _ = vae.infer_z(batch.subject_views, split="test",  use_mu=True)
    z_train = z_tr.cpu().numpy()  # [T_train, k]
    z_test  = z_te.cpu().numpy()  # [T_test,  k]

    # --- standardize z per-dim (train stats only) ---
    z_scaler = StandardScaler(with_mean=True, with_std=True)
    z_train_std = z_scaler.fit_transform(z_train)
    z_test_std  = z_scaler.transform(z_test)

    # --- embeddings: PCA(pca_dim) fit on train only + L2 normalize rows ---
    X_train_raw = batch.X_train.cpu().numpy()
    X_test_raw  = batch.X_test.cpu().numpy()
    pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=SEED)
    X_train_p = pca.fit_transform(X_train_raw)
    X_test_p  = pca.transform(X_test_raw)
    X_train_p = normalize(X_train_p, axis=1)
    X_test_p  = normalize(X_test_p, axis=1)

    # --- RidgeCV (alphas in log-space) ---
    alphas = np.logspace(-3, 3, 13)
    ridge = RidgeCV(alphas=alphas, fit_intercept=True, store_cv_values=False)
    ridge.fit(X_train_p, z_train_std)
    z_hat_test_std = ridge.predict(X_test_p)

    # --- Shared-space r ---
    r_z_dims = _colwise_pearsonr(z_test_std, z_hat_test_std)
    r_z_mean = float(np.nanmean(r_z_dims))

    # --- Electrode-space: decode z_hat_test ---
    z_hat_test = torch.from_numpy(z_scaler.inverse_transform(z_hat_test_std)).float().to(dev)
    zf_hat = vae.core(z_hat_test)
    r_elec_by_subject = []
    for sid in batch.subjects:
        x_true = batch.subject_views[sid].test.to(dev)     # [T_test, E_i]
        x_pred = vae.decoders[str(sid)](zf_hat)            # [T_test, E_i]
        xt = x_true.detach().cpu().numpy()
        xp = x_pred.detach().cpu().numpy()
        r_cols = _colwise_pearsonr(xt, xp)                 # [E_i]
        r_elec_by_subject.append(float(np.nanmean(r_cols)))
    r_elec_mean = float(np.mean(r_elec_by_subject))

    return {
        "r_z_mean": r_z_mean,
        "r_z_dims": r_z_dims,
        "r_elec_mean": r_elec_mean,
        "r_elec_by_subject": r_elec_by_subject,
        "pca_explained": float(pca.explained_variance_ratio_.sum())
    }


# -----------------------------
# Main flow
# -----------------------------
if __name__ == "__main__":
    set_global_seed(SEED, deterministic=True)
    dev = torch_device()
    print(f"[INFO] torch: {torch.__version__}, cuda: {torch.cuda.is_available()}, device: {dev}")
    print(f"[INFO] DATA_PATH={DATA_PATH}  LATENT_K={LATENT_K}  SEED={SEED}  TRAIN_RATIO={TRAIN_RATIO}  LAG_MS(target)={LAG_MS}")

    # Build the batch for a single lag (closest to LAG_MS)
    try:
        batch = make_lag_batch(
            pkl_path=DATA_PATH,
            lag_ms=LAG_MS,
            latent_dim=LATENT_K,
            train_ratio=TRAIN_RATIO,
        )
    except Exception as e:
        print(f"[ERROR] Failed to prepare batch: {e}")
        sys.exit(1)

    # Smoke about shapes
    s1 = batch.subjects[0]
    print(f"[OK] Lag(ms)={batch.lag_ms}  k={batch.latent_dim}  Subjects={batch.subjects}")
    print(f"     S{s1} train shape: {tuple(batch.subject_views[s1].train.shape)}")
    print(f"     X_train: {tuple(batch.X_train.shape)}  X_test: {tuple(batch.X_test.shape)}")

    # Train SRM-VAE
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

    # Encoding evaluation: PCA+Ridge (shared z & electrode space)
    print("[EVAL] PCA({})+RidgeCV encoding".format(PCA_DIM))
    enc = eval_encoding_PCA_Ridge(batch, vae, pca_dim=PCA_DIM)
    print(f"[ENC] Shared-space r (mean over k={LATENT_K}): {enc['r_z_mean']:.4f}  | PCA var={enc['pca_explained']:.2f}")
    print(f"[ENC] Electrode-space r (mean over subjects): {enc['r_elec_mean']:.4f}")
    for i, rsub in enumerate(enc["r_elec_by_subject"], start=1):
        print(f"       S{i}: r_elec_mean={rsub:.4f}")
