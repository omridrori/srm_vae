# srm_vs_vae_shared_encoding.py
# Single-file comparison: classic SRM vs SRM-VAE
# - No CLI, no folds. Train/test split by time.
# - Evaluates a list of lags (ms) and plots shared-space encoding curves.

# ========= CONFIG (edit here) =========
DATA_PATH   = "./all_data.pkl"   # path to your all_data.pkl
SEED        = 1234               # random seed
TRAIN_RATIO = 0.8                # first 80% train, last 20% test
LAG_LIST    = [-500, 100, 1500]

SRM_K       = 5
VAE_K       = 5

# Training / eval hyperparams
VAE_EPOCHS  = 1000
VAE_LR      = 5e-4
VAE_BETA    = 0
SELF_RECON_WEIGHT = 0  # Alpha
CROSS_RECON_WEIGHT =2 # Gamma
PCA_DIM     = 100

# Plotting
YLIM        = (0.0, 0.40)
FIGSIZE     = (4.8, 3.5)
SAVE_PNG    = None              
# =====================================

import os
import sys
import math
import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# --- Torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    print("ERROR: PyTorch not found. Install with: pip install torch")
    raise

# --- SciKit
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import LinearRegression

# --- BrainIAK SRM
try:
    from brainiak.funcalign import srm as brainiak_srm
except Exception:
    print("ERROR: BrainIAK not found. Install with: pip install brainiak")
    raise


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
    Expected keys in the pkl:
      - 'electrode_data': shape [S, L, T, Emax]
      - 'electrode_number': length S
      - 'word_embeddings': shape [T, D_emb]
      - 'lags': length L (ms)
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


# -----------------------------
# Build a lag batch (from loaded arrays)
# -----------------------------

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

    # Embeddings: PCA->L2 will be applied later, but we center here with train mean only.
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
        self.core = nn.Identity()

    @staticmethod
    def _agg_posteriors(mu_list, logvar_list):
        # Simple averaging of means (like SRM) instead of precision-weighted aggregation
        mu = torch.stack(mu_list, dim=0).mean(dim=0)
        # Average the logvars as well for consistency
        logvar = torch.stack(logvar_list, dim=0).mean(dim=0)
        return mu, logvar

    def encode_group(self, subject_views: Dict[int, SubjectView], split: str):
        mu_list, logvar_list = [], []
        for sid, view in subject_views.items():
            x = getattr(view, split)
            mu_i, logvar_i = self.encoders[str(sid)](x)
            mu_list.append(mu_i)
            logvar_list.append(logvar_i)
        return self._agg_posteriors(mu_list, logvar_list)

    def forward(self, subject_views: Dict[int, SubjectView], split: str, beta: float = 1.0, ortho_weight: float = 1e-3):
        mu, logvar = self.encode_group(subject_views, split)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        zf = self.core(z)
        recon_loss = torch.tensor(0.0, device=z.device)
        for sid, view in subject_views.items():
            x = getattr(view, split)
            x_hat = self.decoders[str(sid)](zf)
            recon_loss = recon_loss + F.mse_loss(x_hat, x, reduction='mean')
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
     
        ortho_pen = torch.tensor(0.0, device=z.device)
        for dec in self.decoders.values():
            W = dec.lin.weight  # [E_i, k]
            G = W.T @ W
            I = torch.eye(G.shape[0], device=G.device)
            ortho_pen = ortho_pen + torch.norm(G - I, p='fro')**2
        
        loss = recon_loss + beta * kl + ortho_weight * ortho_pen  # 
        return loss, recon_loss.detach(), kl.detach()

    def forward_cross(self, subject_views: Dict[int, SubjectView], 
                      source_sid: int, target_sid: int, 
                      alpha: float, gamma: float, beta: float):
        # 1. Encode source subject
        source_view = subject_views[source_sid]
        mu, logvar = self.encoders[str(source_sid)](source_view.train)
        
        # 2. Sample from latent space
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        zf = self.core(z)
        
        # 3. Self-reconstruction loss
        source_recon = self.decoders[str(source_sid)](zf)
        loss_self = F.mse_loss(source_recon, source_view.train, reduction='mean')
        
        # 4. Cross-reconstruction loss
        target_view = subject_views[target_sid]
        target_recon = self.decoders[str(target_sid)](zf)
        loss_cross = F.mse_loss(target_recon, target_view.train, reduction='mean')
        
        # 5. KL divergence
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 6. Total weighted loss
        total_loss = alpha * loss_self + gamma * loss_cross + beta * kl
        
        return total_loss, loss_self.detach(), loss_cross.detach(), kl.detach()

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
# Training (VAE)
# -----------------------------

def train_srmvae_on_batch(batch: LagBatch, epochs: int, lr: float, beta: float, 
                          alpha: float, gamma: float, verbose: bool = True) -> SRMVAE:
    dev = torch_device()
    for sid in batch.subjects:
        sv = batch.subject_views[sid]
        sv.train = sv.train.to(dev)
        sv.test  = sv.test.to(dev)

    model = SRMVAE(batch.elec_num, k=batch.latent_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = math.inf
    best_state = None
    
    num_subjects = len(batch.subjects)
    subject_ids = list(batch.subjects)

    for ep in range(1, epochs + 1):
        model.train()
        
        epoch_loss, epoch_self, epoch_cross, epoch_kl = 0, 0, 0, 0
        
        for i in range(num_subjects):
            source_sid = random.choice(subject_ids)
            target_sid = subject_ids[i]
            
            beta_ep = beta * min(1.0, ep / 500)
            
            loss, rec_self, rec_cross, kl = model.forward_cross(
                batch.subject_views, source_sid, target_sid, alpha, gamma, beta_ep
            )
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            epoch_self += rec_self.item()
            epoch_cross += rec_cross.item()
            epoch_kl += kl.item()

        avg_loss = epoch_loss / num_subjects
        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (ep % 25 == 0 or ep == 1):
            print(f"[ep {ep:03d}] loss={avg_loss:.5f}  "
                  f"self={epoch_self/num_subjects:.5f}  "
                  f"cross={epoch_cross/num_subjects:.5f}  "
                  f"kl={epoch_kl/num_subjects:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -----------------------------
# Shared-space encoding metrics (both methods)
# -----------------------------

def _prep_embeddings(X_train: np.ndarray, X_test: np.ndarray, pca_dim: int, seed: int):
    """PCA->L2 normalize rows. Fit PCA on train only."""
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

def shared_encoding_vae(batch: LagBatch, k: int, pca_dim: int, seed: int) -> Tuple[float, np.ndarray]:
    """Return (mean r over dims, r per dim) for SRM-VAE at this lag."""
    dev = torch_device()
    vae = train_srmvae_on_batch(
        batch, 
        epochs=VAE_EPOCHS, 
        lr=VAE_LR, 
        beta=VAE_BETA,
        alpha=SELF_RECON_WEIGHT,
        gamma=CROSS_RECON_WEIGHT,
        verbose=True
    )
    vae.eval()

    # Infer shared z (μ)
    z_tr, _, _ = vae.infer_z(batch.subject_views, split="train", use_mu=True)
    z_te, _, _ = vae.infer_z(batch.subject_views, split="test",  use_mu=True)
    z_train = z_tr.cpu().numpy()  # [T_train, k]
    z_test  = z_te.cpu().numpy()  # [T_test,  k]

    # Standardize z per dim
    z_scaler = StandardScaler(with_mean=True, with_std=True)
    z_train_std = z_scaler.fit_transform(z_train)
    z_test_std  = z_scaler.transform(z_test)

    # Embeddings -> PCA+L2
    X_train_raw = batch.X_train.cpu().numpy()
    X_test_raw  = batch.X_test.cpu().numpy()
    X_train_p, X_test_p, _ = _prep_embeddings(X_train_raw, X_test_raw, pca_dim=pca_dim, seed=seed)

    # Linear map X -> z (multi-output regression)
    reg = LinearRegression()
    reg.fit(X_train_p, z_train_std)
    z_hat_test_std = reg.predict(X_test_p)

    r_dims = _colwise_pearsonr(z_test_std, z_hat_test_std)
    r_mean = float(np.nanmean(r_dims))
    return r_mean, r_dims


def shared_encoding_srm(Y_data: np.ndarray, elec_num: np.ndarray, X: np.ndarray,
                        lags: np.ndarray, lag_ms: int, train_ratio: float,
                        k: int, pca_dim: int, seed: int) -> Tuple[int, float, np.ndarray]:
    """
    Classic SRM baseline (brainiak):
    - Builds time-based train/test splits
    - Fits SRM on train data per subject
    - Averages shared time series across subjects
    - Regresses embeddings → shared dims and returns correlation per dim.
    Returns (chosen_lag_ms, mean r, r per dim)
    """
    S, L, T, Emax = Y_data.shape
    lag_idx = choose_lag_index(lags, lag_ms)
    chosen_ms = int(lags[lag_idx])

    train_idx, test_idx = time_split_indices(T, train_ratio)

    # Prepare per-subject matrices
    train_data, test_data = [], []
    for s in range(S):
        e_i = int(elec_num[s])
        mat = Y_data[s, lag_idx, :, :e_i]  # [T, E_i]
        tr = mat[train_idx, :]
        te = mat[test_idx, :]
        tr_z, te_z = _zscore_train_apply(tr, te)
        train_data.append(tr_z.T)  # [E_i, T_train]
        test_data.append(te_z.T)   # [E_i, T_test]

    # Fit SRM
    srm = brainiak_srm.SRM(n_iter=100, features=k)
    srm.fit(train_data)

    shared_train_list = srm.transform(train_data)
    shared_test_list  = srm.transform(test_data)

    s_train = np.mean(np.stack(shared_train_list, axis=0), axis=0).T  # [T_train, k]
    s_test  = np.mean(np.stack(shared_test_list,  axis=0), axis=0).T  # [T_test , k]

    # Standardize dims
    z_scaler = StandardScaler(with_mean=True, with_std=True)
    s_train_std = z_scaler.fit_transform(s_train)
    s_test_std  = z_scaler.transform(s_test)

    # Embeddings -> PCA+L2
    X_train = X[train_idx, :]
    X_test  = X[test_idx, :]
    X_train = X_train - X_train.mean(axis=0, keepdims=True)
    X_test  = X_test  - X_train.mean(axis=0, keepdims=True)
    X_train_p, X_test_p, _ = _prep_embeddings(X_train, X_test, pca_dim=pca_dim, seed=seed)

    reg = LinearRegression()
    reg.fit(X_train_p, s_train_std)
    s_hat_test_std = reg.predict(X_test_p)

    r_dims = _colwise_pearsonr(s_test_std, s_hat_test_std)
    r_mean = float(np.nanmean(r_dims))
    return chosen_ms, r_mean, r_dims


# -----------------------------
# Multi-lag sweep + plot
# -----------------------------

def sweep_and_plot(Y_data, elec_num, X, lags, lag_list, srm_k, vae_k,
                   train_ratio, pca_dim, seed, ylim, figsize, save_png):
    dev = torch_device()
    print(f"[INFO] torch: {torch.__version__}, cuda: {torch.cuda.is_available()}, device: {dev}")
    print(f"[DATA] Y={Y_data.shape}, X={X.shape}, lags_count={len(lags)}")
    print(f"[CONF] TRAIN_RATIO={train_ratio}, SRM_K={srm_k}, VAE_K={vae_k}, PCA_DIM={pca_dim}")

    srm_means, srm_sems, lag_actual_srm = [], [], []
    vae_means, vae_sems, lag_actual_vae = [], [], []

    for i, req in enumerate(lag_list):
        print(f"\n--- Processing lag {i+1}/{len(lag_list)} (requested: {req}ms) ---")
        # SRM baseline
        chosen_ms, srm_mean, srm_r_dims = shared_encoding_srm(
            Y_data, elec_num, X, lags,
            lag_ms=req, train_ratio=train_ratio,
            k=srm_k, pca_dim=pca_dim, seed=seed
        )
        srm_means.append(srm_mean)
        srm_sems.append(np.nanstd(srm_r_dims, ddof=1) / np.sqrt(len(srm_r_dims)))
        lag_actual_srm.append(chosen_ms)
        print(f"[SRM] lag req={req} -> chosen={chosen_ms} | r_mean={srm_mean:.3f}")

        # VAE baseline
        batch = build_lag_batch_from_loaded(
            Y_data, elec_num, X, lags,
            lag_ms=req, latent_dim=vae_k, train_ratio=train_ratio
        )
        vae_mean, vae_r_dims = shared_encoding_vae(batch, k=vae_k, pca_dim=pca_dim, seed=seed)
        vae_means.append(vae_mean)
        vae_sems.append(np.nanstd(vae_r_dims, ddof=1) / np.sqrt(len(vae_r_dims)))
        lag_actual_vae.append(batch.lag_ms)
        print(f"[VAE] lag req={req} -> chosen={batch.lag_ms} | r_mean={vae_mean:.3f}")

    x = np.asarray(lag_list, dtype=int)
    srm_means = np.asarray(srm_means)
    srm_sems  = np.asarray(srm_sems)
    vae_means = np.asarray(vae_means)
    vae_sems  = np.asarray(vae_sems)

    # Plot
    plt.figure(figsize=figsize)
    plt.fill_between(x, srm_means - srm_sems, srm_means + srm_sems, alpha=0.2, color='darkorange')
    plt.plot(x, srm_means, linewidth=3.5, label='SRM', color='darkorange')

    plt.fill_between(x, vae_means - vae_sems, vae_means + vae_sems, alpha=0.2, color='royalblue')
    plt.plot(x, vae_means, linewidth=3.5, label='VAE', color='royalblue')

    plt.axvline(0, ls='dashed', c='k', alpha=0.3)
    plt.ylim(ylim)
    plt.xlabel('lags (ms)')
    plt.ylabel('Encoding Performance (r)')
    plt.title('Shared Space Encoding')
    plt.legend()
    plt.tight_layout()

    if save_png:
        plt.savefig(save_png, dpi=150)
        print(f"[PLOT] Saved to {save_png}")
    plt.show()

    return {
        'x_requested': x,
        'srm': {'means': srm_means, 'sems': srm_sems, 'chosen_lags': lag_actual_srm},
        'vae': {'means': vae_means, 'sems': vae_sems, 'chosen_lags': lag_actual_vae},
    }


# -----------------------------
# Main
# -----------------------------

if __name__ == '__main__':
    set_global_seed(SEED, deterministic=True)
    try:
        print("[INFO] Loading data... this may take a while.")
        Y_data, elec_num, X, lags = load_all_data(DATA_PATH)
        print("[INFO] Data loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

    sweep_and_plot(
        Y_data=Y_data,
        elec_num=elec_num,
        X=X,
        lags=lags,
        lag_list=LAG_LIST,
        srm_k=SRM_K,
        vae_k=VAE_K,
        train_ratio=TRAIN_RATIO,
        pca_dim=PCA_DIM,
        seed=SEED,
        ylim=YLIM,
        figsize=FIGSIZE,
        save_png=SAVE_PNG,
    )
