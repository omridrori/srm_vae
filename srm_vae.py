# srm_vae_setup.py
# MVP Phase 0: seeds, dataclasses, loader, smoke test — single file, no argparse.

# ====== CONFIG (edit here) ======
DATA_PATH = "./all_data.pkl"  # path to your all_data.pkl
CV        = 10                # KFold splits
LATENT_K  = 5                 # latent dim to carry forward
SEED      = 1234              # random seed
# =================================

import os
import sys
import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Generator, Tuple

import numpy as np
from sklearn.model_selection import KFold

try:
    import torch
except Exception as e:
    print("ERROR: PyTorch not found. Install with: pip install torch")
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
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Data types
# -----------------------------
@dataclass
class SubjectView:
    """Time × Electrodes for one subject, already train/test split & normalized."""
    train: torch.Tensor  # [T_train, E_i]
    test:  torch.Tensor  # [T_test , E_i]
    mask_train: Optional[torch.Tensor] = None  # [E_i]
    mask_test:  Optional[torch.Tensor] = None  # [E_i]

@dataclass
class FoldLagBatch:
    """Container for a single (fold, lag) across all subjects."""
    fold_idx: int
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
# I/O and loader
# -----------------------------
def _zscore_train_apply(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score by train stats only; epsilon to avoid div-by-zero."""
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mu) / sd, (test - mu) / sd

def load_all_data(pkl_path: str):
    """Reads the single pickle created in your codebase."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    Y_data = obj["electrode_data"]       # shape [S, L, T, Emax]
    elec_num = obj["electrode_number"]   # length S
    word_embeddings = obj["word_embeddings"]  # shape [T, D_emb]
    lags = obj["lags"]                   # length L
    return Y_data, elec_num, word_embeddings, lags

def fold_lag_batches(
    pkl_path: str = DATA_PATH,
    cv: int = CV,
    latent_dim: int = LATENT_K,
    seed: int = SEED,
) -> Generator[FoldLagBatch, None, None]:
    """
    Yields a FoldLagBatch for each (fold, lag), mirroring your SRM loops.
    - Splits by time (KFold on word embeddings rows).
    - Normalizes per subject/electrode using train-only stats.
    """
    Y_data, elec_num, X, lags = load_all_data(pkl_path)
    S, L, T, Emax = Y_data.shape
    subjects = list(range(1, S + 1))

    elec_num = np.asarray(elec_num, dtype=int)
    lags = np.asarray(lags).reshape(-1)

    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        # Embeddings train/test + centering by train mean
        X_train = X[train_index, :]
        X_test  = X[test_index, :]
        X_train = X_train - X_train.mean(axis=0, keepdims=True)
        X_test  = X_test  - X_train.mean(axis=0, keepdims=True)

        for lag_i in range(L):
            subject_views: Dict[int, SubjectView] = {}
            per_sub_elec: Dict[int, int] = {}

            for s in range(S):
                e_i = int(elec_num[s])
                # [T, E_i] for subject s at lag lag_i
                mat = Y_data[s, lag_i, :, :e_i]  # numpy [T, E_i]
                train = mat[train_index, :]
                test  = mat[test_index, :]

                train_z, test_z = _zscore_train_apply(train, test)

                # Masks (all True by default; plug real masks if needed)
                mask = np.ones((e_i,), dtype=bool)

                subject_views[subjects[s]] = SubjectView(
                    train=torch.from_numpy(train_z).float(),
                    test=torch.from_numpy(test_z).float(),
                    mask_train=torch.from_numpy(mask),
                    mask_test=torch.from_numpy(mask),
                )
                per_sub_elec[subjects[s]] = e_i

            yield FoldLagBatch(
                fold_idx=fold_idx,
                lag_ms=int(lags[lag_i]),
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
# Smoke test (runs on import)
# -----------------------------
if __name__ == "__main__":
    set_global_seed(SEED, deterministic=True)
    dev = torch_device()
    print(f"[INFO] torch: {torch.__version__}, cuda: {torch.cuda.is_available()}, device: {dev}")
    print(f"[INFO] DATA_PATH={DATA_PATH}  CV={CV}  LATENT_K={LATENT_K}  SEED={SEED}")

    # iterate once to validate shapes
    gen = fold_lag_batches()
    try:
        batch = next(gen)
    except StopIteration:
        print("[ERROR] No batches yielded — check data file and shapes.")
        sys.exit(1)

    print(f"[OK] Fold={batch.fold_idx}  Lag(ms)={batch.lag_ms}  k={batch.latent_dim}")
    print(f"     Subjects: {batch.subjects}")
    s1 = batch.subjects[0]
    print(f"     S{s1} train shape: {tuple(batch.subject_views[s1].train.shape)}")
    print(f"     X_train: {tuple(batch.X_train.shape)}  X_test: {tuple(batch.X_test.shape)}")
    print(f"     Elec S{s1}: {batch.elec_num[s1]}")



# ====== SRM-VAE MVP (append below the existing code) ======
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# ---------- Per-subject encoder & decoder ----------
class PerSubjectEncoder(nn.Module):
    """Linear encoder per subject: R^{E_i} -> (mu, logvar) in R^k"""
    def __init__(self, e_i: int, k: int):
        super().__init__()
        self.lin = nn.Linear(e_i, 2*k, bias=True)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # x: [T, E_i]
        h = self.lin(x)              # [T, 2k]
        mu, logvar = torch.chunk(h, 2, dim=-1)  # [T, k], [T, k]
        return mu, logvar.clamp(min=-8.0, max=8.0)  # numerical safety

class PerSubjectDecoder(nn.Module):
    """Linear decoder per subject: R^k -> R^{E_i} (like W_i)"""
    def __init__(self, k: int, e_i: int):
        super().__init__()
        self.lin = nn.Linear(k, e_i, bias=False)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, zf: torch.Tensor) -> torch.Tensor:
        # zf: [T, k]
        return self.lin(zf)          # [T, E_i]

# ---------- SRM-VAE core ----------
class SRMVAE(nn.Module):
    """
    Aggregates subject posteriors into a group latent z_t (precision-weighted),
    then decodes back to each subject via linear W_i (SRM-like).
    """
    def __init__(self, elec_num: Dict[int, int], k: int):
        super().__init__()
        self.k = k
        # Encoders/decoders per subject id
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for sid, e_i in elec_num.items():
            self.encoders[str(sid)] = PerSubjectEncoder(e_i, k)
            self.decoders[str(sid)] = PerSubjectDecoder(k, e_i)
        # shared nonlinearity f_theta (start as identity-ish MLP)
        self.core = nn.Sequential(
            nn.Linear(k, k), nn.ReLU(),
            nn.Linear(k, k),
        )
        for m in self.core:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def _agg_posteriors(mu_list, logvar_list):
        """
        Precision-weighted aggregation across subjects per timepoint:
        sigma^{-2} = sum_i sigma_i^{-2}, mu = sigma^2 * sum(mu_i * sigma_i^{-2})
        Shapes: each [T, k]
        """
        precisions = [torch.exp(-lv) for lv in logvar_list]       # [T,k]
        prec_sum = torch.stack(precisions, dim=0).sum(dim=0)      # [T,k]
        weighted_mu = torch.stack([m*p for m,p in zip(mu_list, precisions)], dim=0).sum(dim=0)  # [T,k]
        var = 1.0 / (prec_sum + 1e-8)
        mu = var * weighted_mu
        logvar = torch.log(var + 1e-8)
        return mu, logvar

    def encode_group(self, batch_subject_views: Dict[int, 'SubjectView'], split: str):
        """
        Encode all subjects -> aggregate to group posterior.
        split: "train" or "test"
        Returns: mu[T,k], logvar[T,k], per-subject (mu_i, logvar_i) dicts if needed
        """
        mu_list, logvar_list = [], []
        for sid, view in batch_subject_views.items():
            x = getattr(view, split)  # [T, E_i]
            mu_i, logvar_i = self.encoders[str(sid)](x)
            mu_list.append(mu_i)
            logvar_list.append(logvar_i)
        mu, logvar = self._agg_posteriors(mu_list, logvar_list)
        return mu, logvar

    def forward(self, batch_subject_views: Dict[int, 'SubjectView'], split: str, beta: float = 1.0):
        """
        Compute β-VAE loss on the chosen split ("train" usually).
        """
        mu, logvar = self.encode_group(batch_subject_views, split)     # [T,k]
        # reparameterization
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps                         # [T,k]
        zf = self.core(z)                                              # f_theta(z) -> [T,k]

        recon_loss = 0.0
        for sid, view in batch_subject_views.items():
            x = getattr(view, split)                                    # [T,E_i]
            x_hat = self.decoders[str(sid)](zf)                         # [T,E_i]
            recon_loss = recon_loss + F.mse_loss(x_hat, x, reduction='mean')

        # KL to N(0, I): average over T and k
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl
        return loss, recon_loss.detach(), kl.detach()

    @torch.no_grad()
    def infer_z(self, batch_subject_views: Dict[int, 'SubjectView'], split: str, use_mu: bool = True):
        mu, logvar = self.encode_group(batch_subject_views, split)
        if use_mu:
            z = mu
        else:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return z, mu, logvar

    @torch.no_grad()
    def reconstruct_subjects(self, z: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Given latent z (T,k), return reconstructions per subject: sid -> [T, E_i]
        """
        zf = self.core(z)
        out = {}
        for sid, dec in self.decoders.items():
            out[int(sid)] = dec(zf)
        return out


# ---------- Minimal trainer on the FIRST (fold, lag) ----------
def train_one_batch_srmvae(batch: FoldLagBatch, epochs: int = 400, lr: float = 1e-3, beta: float = 1.0, verbose: bool = True):
    dev = torch_device()
    # move tensors to device
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
    patience = 40
    no_imp = 0

    for ep in range(1, epochs+1):
        model.train()
        loss, rec, kl = model(batch.subject_views, split="train", beta=beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # simple val proxy: training loss (no separate val split here)
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


# ---------- Run MVP train & quick inference on the first yielded batch ----------
if __name__ == "__main__":
    # We already printed the smoke info above. Now actually train SRM-VAE on the SAME first batch.
    # Recreate the generator to get the same first (fold, lag)
    gen2 = fold_lag_batches(pkl_path=DATA_PATH, cv=CV, latent_dim=LATENT_K, seed=SEED)
    first_batch = next(gen2)

    print(f"[TRAIN] SRM-VAE on fold={first_batch.fold_idx}, lag={first_batch.lag_ms} ms, k={first_batch.latent_dim}")
    vae = train_one_batch_srmvae(first_batch, epochs=300, lr=1e-3, beta=1.0, verbose=True)

    # Inference: latent z on train/test and reconstructions on test
    dev = torch_device()
    vae.eval()
    z_tr, mu_tr, _ = vae.infer_z(first_batch.subject_views, split="train", use_mu=True)
    z_te, mu_te, _ = vae.infer_z(first_batch.subject_views, split="test", use_mu=True)
    recons_test = vae.reconstruct_subjects(z_te.to(dev))

    # Quick sanity: report MSE per subject on test recon (not the true metric yet)
    with torch.no_grad():
        for sid in first_batch.subjects:
            x_true = first_batch.subject_views[sid].test.to(dev)   # [T,E_i]
            x_hat  = recons_test[sid]                              # [T,E_i]
            mse = F.mse_loss(x_hat, x_true).item()
            print(f"[TEST] Subject {sid}: recon MSE = {mse:.6f}")


# ====== Improved encoding: PCA(50)+L2 for X, RidgeCV, z-standardization ======
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler

def _colwise_pearsonr(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8):
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0)) + eps
    return num / den

@torch.no_grad()
def eval_encoding_shared_and_electrodes_PCA_Ridge(batch: FoldLagBatch, vae: SRMVAE, pca_dim: int = 50):
    dev = torch_device()

    # --- latent z (μ) ---
    z_tr, _, _ = vae.infer_z(batch.subject_views, split="train", use_mu=True)
    z_te, _, _ = vae.infer_z(batch.subject_views, split="test",  use_mu=True)
    z_train = z_tr.cpu().numpy()   # [T_train, k]
    z_test  = z_te.cpu().numpy()   # [T_test,  k]

    # --- standardize z per-dim (train stats only) ---
    z_scaler = StandardScaler(with_mean=True, with_std=True)
    z_train_std = z_scaler.fit_transform(z_train)
    z_test_std  = z_scaler.transform(z_test)

    # --- embeddings: PCA(50) fit on train only + L2 normalize rows ---
    X_train_raw = batch.X_train.cpu().numpy()
    X_test_raw  = batch.X_test.cpu().numpy()
    pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=SEED)
    X_train_p = pca.fit_transform(X_train_raw)  # [T_train, 50]
    X_test_p  = pca.transform(X_test_raw)       # [T_test , 50]
    X_train_p = normalize(X_train_p, axis=1)    # L2 per row
    X_test_p  = normalize(X_test_p, axis=1)

    # --- RidgeCV (alphas לוג-ספייס) ---
    alphas = np.logspace(-3, 3, 13)
    ridge = RidgeCV(alphas=alphas, fit_intercept=True, store_cv_values=False)
    ridge.fit(X_train_p, z_train_std)
    z_hat_test_std = ridge.predict(X_test_p)

    # --- Shared-space r ---
    r_z_dims = _colwise_pearsonr(z_test_std, z_hat_test_std)
    r_z_mean = float(np.nanmean(r_z_dims))

    # --- Electrode-space: decode z_hat_test ---
    z_hat_test_t = torch.from_numpy(z_scaler.inverse_transform(z_hat_test_std)).float().to(dev)  # back to original z scale
    zf_hat = vae.core(z_hat_test_t)
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

# קריאה במקום הפונקציה הקודמת:
if __name__ == "__main__":
    print("[EVAL] PCA(50)+RidgeCV encoding (shared z & electrodes)")
    enc_res2 = eval_encoding_shared_and_electrodes_PCA_Ridge(first_batch, vae, pca_dim=50)
    print(f"[ENC] Shared-space r (mean over k={LATENT_K}): {enc_res2['r_z_mean']:.4f}  | PCA var={enc_res2['pca_explained']:.2f}")
    print(f"[ENC] Electrode-space r (mean over subjects): {enc_res2['r_elec_mean']:.4f}")
    for i, rsub in enumerate(enc_res2["r_elec_by_subject"], start=1):
        print(f"       S{i}: r_elec_mean={rsub:.4f}")


# ====== Quick multi-lag sweep (first fold only) ======
if __name__ == "__main__":
    print("[SWEEP] Testing a few lags on fold 0...")
    # נקח את fold הראשון ונרוץ על רשימת לגים
    target_lags = [-2000, -1000, -500, 0, 100, 200, 300, 500]
    results = []
    gen_all = fold_lag_batches(pkl_path=DATA_PATH, cv=CV, latent_dim=LATENT_K, seed=SEED)

    # נאתר את כל ה-batches של fold=0 ונבחר מהם רק את ה-lags הרלוונטיים
    batches_fold0 = {}
    for b in gen_all:
        if b.fold_idx != 0:
            break  # יצאנו מה-fold הראשון (הגנרטור מדורג לפי folds ואז lags)
        batches_fold0[b.lag_ms] = b

    for lag in target_lags:
        if lag not in batches_fold0:
            continue
        print(f"\n[SWEEP] Fold=0 Lag={lag}ms: training VAE…")
        vae_lag = train_one_batch_srmvae(batches_fold0[lag], epochs=250, lr=1e-3, beta=1.0, verbose=False)
        enc = eval_encoding_shared_and_electrodes_PCA_Ridge(batches_fold0[lag], vae_lag, pca_dim=50)
        print(f"[Lag {lag:>5}] r_z_mean={enc['r_z_mean']:.3f} | r_elec_mean={enc['r_elec_mean']:.3f}")
        results.append((lag, enc['r_z_mean'], enc['r_elec_mean']))

    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        best = results[0]
        print(f"\n[BEST-LAG] by shared r: Lag={best[0]} ms | r_z_mean={best[1]:.3f} | r_elec_mean={best[2]:.3f}")