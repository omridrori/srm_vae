# vae_wandb_sweep_single_lag.py
# Hyperparameter optimization for SRM-VAE at a single lag using Weights & Biases Sweeps.
# Objective: maximize Shared-space r (mean over k).
# One file, no argparse. Edit CONFIG below and run the script.
import math

# ========== CONFIG (edit here) ==========
DATA_PATH        = "./all_data.pkl"
SEED_BASE        = 1234
LAG_MS           = 20             # target lag (ms); closest available is used
TRAIN_RATIO      = 0.8

# Training budget per run
EPOCHS_PER_TRIAL = 1000            # early stopping will usually stop earlier
PATIENCE         = 1000
PCA_DIM          = 50             # PCA dims for X before encoding eval
VERBOSE_EVERY    = 25

# WANDB settings
WANDB_ENTITY     = None           # e.g., "your-team" or None for personal
WANDB_PROJECT    = "srm-vae-hpo"
WANDB_RUN_GROUP  = "lag20"
SWEEP_NAME       = "vae-bayes-lag20"
SWEEP_METHOD     = "bayes"       # "bayes" or "random"
SWEEP_TRIALS     = 60             # total runs to launch sequentially
SWEEP_ID         = None           # set a sweep ID to join an existing sweep instead of creating a new one
SAVE_BEST_MODEL  = "best_vae_wandb.pt"  # saved whenever a run beats the global best r
# ========================================

import os
import sys
import math
import time
import json
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

# Weights & Biases
try:
    import wandb
except Exception:
    print("ERROR: wandb not found. Install with: pip install wandb")
    raise


# -----------------------------
# Repro / device
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
    mask_train: Optional[torch.Tensor] = None
    mask_test:  Optional[torch.Tensor] = None


@dataclass
class LagBatch:
    lag_ms: int
    latent_dim: int
    subjects: List[int]
    subject_views: Dict[int, SubjectView]
    X_train: torch.Tensor
    X_test:  torch.Tensor
    train_index: np.ndarray
    test_index:  np.ndarray
    elec_num: Dict[int, int]


# -----------------------------
# I/O and preprocessing
# -----------------------------

def load_all_data(pkl_path: str):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    Y_data = np.asarray(obj["electrode_data"])          # [S, L, T, Emax]
    elec_num = np.asarray(obj["electrode_number"], int) # [S]
    X = np.asarray(obj["word_embeddings"])              # [T, D_emb]
    lags = np.asarray(obj["lags"]).reshape(-1)          # [L]
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
        mat = Y_data[s, lag_idx, :, :e_i]
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
# VAE model + optional temporal cores
# -----------------------------

class TemporalConvCore(nn.Module):
    def __init__(self, k: int, ks: int = 5):
        super().__init__()
        pad = ks // 2
        self.net = nn.Sequential(
            nn.Conv1d(k, k, kernel_size=ks, padding=pad, groups=1),
            nn.ReLU(),
            nn.Conv1d(k, k, kernel_size=ks, padding=pad, groups=1),
        )
        for m in self.net:
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z1 = z.transpose(0, 1).unsqueeze(0)  # [1,k,T]
        y = self.net(z1)
        return y.squeeze(0).transpose(0, 1)  # [T,k]


class GRUCore(nn.Module):
    def __init__(self, k: int, hidden: Optional[int] = None):
        super().__init__()
        h = hidden if isinstance(hidden, int) else k
        self.rnn = nn.GRU(input_size=k, hidden_size=h, batch_first=True)
        self.out = nn.Linear(h, k)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(z.unsqueeze(0))  # [1,T,h]
        return self.out(y.squeeze(0))    # [T,k]


class PerSubjectEncoder(nn.Module):
    def __init__(self, e_i: int, k: int):
        super().__init__()
        self.lin = nn.Linear(e_i, 2*k, bias=True)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.lin(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar.clamp(min=-8.0, max=8.0)


class PerSubjectDecoder(nn.Module):
    def __init__(self, k: int, e_i: int):
        super().__init__()
        self.lin = nn.Linear(k, e_i, bias=False)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, zf: torch.Tensor) -> torch.Tensor:
        return self.lin(zf)


class SRMVAE(nn.Module):
    def __init__(self, elec_num: Dict[int, int], k: int, core_type: str = "mlp",
                 conv_ks: int = 5, gru_hidden: Optional[int] = None,
                 lambda_ortho: float = 0.0, lambda_smooth: float = 0.0, dropout_p: float = 0.0):
        super().__init__()
        self.k = k
        self.lambda_ortho = lambda_ortho
        self.lambda_smooth = lambda_smooth
        self.dropout_p = dropout_p

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for sid, e_i in elec_num.items():
            self.encoders[str(sid)] = PerSubjectEncoder(e_i, k)
            self.decoders[str(sid)] = PerSubjectDecoder(k, e_i)

        if core_type == "mlp":
            self.core = nn.Sequential(nn.Linear(k, k), nn.ReLU(), nn.Linear(k, k))
            for m in self.core:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
        elif core_type == "conv":
            self.core = TemporalConvCore(k, ks=conv_ks)
        elif core_type == "gru":
            self.core = GRUCore(k, hidden=gru_hidden)
        else:
            raise ValueError(f"Unknown core_type: {core_type}")

    @staticmethod
    def _agg_posteriors(mu_list, logvar_list):
        precisions = [torch.exp(-lv) for lv in logvar_list]
        prec_sum = torch.stack(precisions, dim=0).sum(dim=0)
        weighted_mu = torch.stack([m*p for m,p in zip(mu_list, precisions)], dim=0).sum(dim=0)
        var = 1.0 / (prec_sum + 1e-8)
        mu = var * weighted_mu
        logvar = torch.log(var + 1e-8)
        return mu, logvar

    def _apply_channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.dropout_p <= 0:
            return x
        mask = (torch.rand((x.shape[-1],), device=x.device) > self.dropout_p).float()
        return x * mask.unsqueeze(0)

    def encode_group(self, subject_views: Dict[int, SubjectView], split: str):
        mu_list, logvar_list = [], []
        for sid, view in subject_views.items():
            x = getattr(view, split)
            x = self._apply_channel_dropout(x)
            mu_i, logvar_i = self.encoders[str(sid)](x)
            mu_list.append(mu_i)
            logvar_list.append(logvar_i)
        return self._agg_posteriors(mu_list, logvar_list)

    def _temporal_smoothness(self, z: torch.Tensor) -> torch.Tensor:
        if self.lambda_smooth <= 0:
            return torch.tensor(0.0, device=z.device)
        return self.lambda_smooth * ((z[1:] - z[:-1])**2).mean()

    def _ortho_penalty(self, device) -> torch.Tensor:
        if self.lambda_ortho <= 0:
            return torch.tensor(0.0, device=device)
        pen = torch.tensor(0.0, device=device)
        for dec in self.decoders.values():
            W = dec.lin.weight
            G = W.T @ W
            I = torch.eye(G.shape[0], device=device)
            pen = pen + torch.norm(G - I, p='fro')**2
        return self.lambda_ortho * pen

    def forward(self, subject_views: Dict[int, SubjectView], split: str, beta: float = 1.0):
        mu, logvar = self.encode_group(subject_views, split)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        zf = self.core(z)

        recon_loss = torch.tensor(0.0, device=z.device)
        for sid, view in subject_views.items():
            x = getattr(view, split)
            x_hat = self.decoders[str(sid)](zf)
            recon_loss = recon_loss + F.mse_loss(x_hat, x, reduction='mean')

        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl + self._temporal_smoothness(z) + self._ortho_penalty(z.device)
        return loss, recon_loss.detach(), kl.detach()

    @torch.no_grad()
    def infer_z(self, subject_views: Dict[int, SubjectView], split: str, use_mu: bool = True):
        mu, logvar = self.encode_group(subject_views, split)
        z = mu if use_mu else mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return z, mu, logvar


# -----------------------------
# Eval helpers
# -----------------------------

def _prep_embeddings(X_train: np.ndarray, X_test: np.ndarray, pca_dim: int, seed: int):
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


def eval_shared_r(batch: LagBatch, vae: SRMVAE, pca_dim: int, seed: int):
    vae.eval()
    with torch.no_grad():
        z_tr, _, _ = vae.infer_z(batch.subject_views, split="train", use_mu=True)
        z_te, _, _ = vae.infer_z(batch.subject_views, split="test",  use_mu=True)

    z_train = z_tr.cpu().numpy()
    z_test  = z_te.cpu().numpy()

    z_scaler = StandardScaler(with_mean=True, with_std=True)
    z_train_std = z_scaler.fit_transform(z_train)
    z_test_std  = z_scaler.transform(z_test)

    X_train_raw = batch.X_train.cpu().numpy()
    X_test_raw  = batch.X_test.cpu().numpy()
    X_train_p, X_test_p, pca_var = _prep_embeddings(X_train_raw, X_test_raw, pca_dim=pca_dim, seed=SEED_BASE)

    reg = LinearRegression()
    reg.fit(X_train_p, z_train_std)
    z_hat_test_std = reg.predict(X_test_p)

    r_dims = _colwise_pearsonr(z_test_std, z_hat_test_std)
    r_mean = float(np.nanmean(r_dims))
    return r_mean, r_dims, pca_var


# -----------------------------
# Global data (loaded once)
# -----------------------------
Y_data = None
X = None
elec_num = None
lags = None


def make_batch_for_k(k: int) -> LagBatch:
    return build_lag_batch_from_loaded(
        Y_data=Y_data, elec_num=elec_num, X=X, lags=lags,
        lag_ms=LAG_MS, latent_dim=k, train_ratio=TRAIN_RATIO
    )


# -----------------------------
# One run function (used by W&B agent)
# -----------------------------

def run_once():
    """One training+eval run driven by wandb.config; logs to W&B and returns r_mean."""
    device = torch_device()
    cfg = wandb.config

    # Build batch per latent k
    batch = make_batch_for_k(int(cfg.latent_k))

    # Model
    k = int(cfg.latent_k)
    core_type = cfg.core_type
    conv_ks = int(cfg.conv_ks)
    gru_hidden = None
    if core_type == "gru":
        if str(cfg.gru_hidden) == "k":
            gru_hidden = k
        else:
            try:
                gru_hidden = int(cfg.gru_hidden)
            except Exception:
                gru_hidden = k

    model = SRMVAE(
        elec_num=batch.elec_num, k=k, core_type=core_type,
        conv_ks=conv_ks, gru_hidden=gru_hidden,
        lambda_ortho=float(cfg.lambda_ortho),
        lambda_smooth=float(cfg.lambda_smooth),
        dropout_p=float(cfg.dropout_p),
    ).to(device)

    # Move tensors once
    for sid in batch.subjects:
        sv = batch.subject_views[sid]
        sv.train = sv.train.to(device)
        sv.test  = sv.test.to(device)
        if sv.mask_train is not None: sv.mask_train = sv.mask_train.to(device)
        if sv.mask_test  is not None: sv.mask_test  = sv.mask_test.to(device)

    # Optim
    lr = float(cfg.lr)
    beta = float(cfg.beta)
    warmup_frac = float(cfg.warmup_frac)
    warmup_epochs = max(0, int(round(warmup_frac * EPOCHS_PER_TRIAL)))
    weight_decay = float(cfg.weight_decay)
    grad_clip = float(cfg.grad_clip)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train with early stopping
    best_loss = math.inf
    best_state = None
    no_imp = 0

    for ep in range(1, EPOCHS_PER_TRIAL + 1):
        model.train()
        beta_ep = beta if ep > warmup_epochs else beta * (ep / max(1, warmup_epochs))
        loss, rec, kl = model(batch.subject_views, split="train", beta=beta_ep)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        opt.step()

        L = float(loss.item())
        if L < best_loss - 1e-5:
            best_loss = L
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if (ep % VERBOSE_EVERY == 0) or (ep == 1):
            wandb.log({
                "epoch": ep,
                "loss": L,
                "recon": float(rec),
                "kl": float(kl),
                "beta_ep": float(beta_ep),
            })

        if no_imp >= PATIENCE:
            wandb.log({"early_stop_epoch": ep})
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate shared r
    r_mean, r_dims, pca_var = eval_shared_r(batch, model, pca_dim=PCA_DIM, seed=SEED_BASE)
    wandb.log({
        "r_mean": r_mean,
        "pca_var": pca_var,
        "lag_ms": int(batch.lag_ms),
    })
    wandb.run.summary["r_mean"] = r_mean

    # Optionally save model artifact if it's the global best so far (tracked in sweep state on disk)
    if SAVE_BEST_MODEL is not None:
        # Keep a simple on-disk best tracker
        flag_path = SAVE_BEST_MODEL + ".json"
        prev = {"best": -1.0}
        if os.path.exists(flag_path):
            try:
                with open(flag_path, "r") as f:
                    prev = json.load(f)
            except Exception:
                prev = {"best": -1.0}
        if r_mean > float(prev.get("best", -1.0)):
            torch.save(model.state_dict(), SAVE_BEST_MODEL)
            with open(flag_path, "w") as f:
                json.dump({"best": r_mean, "run_id": wandb.run.id}, f)
            art = wandb.Artifact("best_vae", type="model")
            art.add_file(SAVE_BEST_MODEL)
            wandb.log_artifact(art)

    return r_mean


# -----------------------------
# Sweep config (W&B)
# -----------------------------

def build_sweep_config():
    return {
        "name": SWEEP_NAME,
        "method": SWEEP_METHOD,  # "bayes" or "random"
        "metric": {"name": "r_mean", "goal": "maximize"},
        "parameters": {
            # model size
            "latent_k": {"values": [5, 8, 10]},
            # optimizer
            "lr":   {"distribution": "log_uniform", "min": 1e-4, "max": 5e-3},
            "weight_decay": {"distribution": "uniform", "min": 0.0, "max": 5e-4},
            "grad_clip": {"distribution": "uniform", "min": 0.0, "max": 2.0},
            # KL & regs
            "beta": {"distribution": "log_uniform", "min": 1e-3, "max": 5.0},
            "warmup_frac": {"distribution": "uniform", "min": 0.0, "max": 0.5},
            "lambda_ortho": {"distribution": "uniform", "min": 0.0, "max": 3e-3},
            "lambda_smooth": {"distribution": "uniform", "min": 0.0, "max": 3e-3},
            "dropout_p": {"distribution": "uniform", "min": 0.0, "max": 0.15},
            # core
            "core_type": {"values": ["mlp", "conv", "gru"]},
            "conv_ks": {"values": [3, 5, 7, 9]},
            "gru_hidden": {"values": ["k", 2]},  # "k" means hidden=k, 2 means hidden=2k
        },
        # Early terminate poorly performing runs (optional)
        "early_terminate": {
            "type": "hyperband",
            "min_iter": max(10, PATIENCE),
        },
    }


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    set_global_seed(SEED_BASE, deterministic=True)
    device = torch_device()
    print(f"[INFO] torch: {torch.__version__}, cuda: {torch.cuda.is_available()}, device: {device}")

    # Load data once
    try:
        Y_data, elec_num, X, lags = load_all_data(DATA_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

    # Log a dataset summary in W&B as run notes/summary later
    data_info = {"S": int(Y_data.shape[0]), "L": int(Y_data.shape[1]), "T": int(Y_data.shape[2]), "Emax": int(Y_data.shape[3])}
    print(f"[DATA] {data_info} | target_lag={LAG_MS}")

    # Define sweep
    sweep_config = build_sweep_config()

    # Create or join a sweep
    if SWEEP_ID is None:
        sweep_id = wandb.sweep(sweep=sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY)
        print(f"[W&B] Created sweep: {sweep_id}")
    else:
        sweep_id = SWEEP_ID
        print(f"[W&B] Joining existing sweep: {sweep_id}")

    # Agent entrypoint
    def _agent():
        run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, group=WANDB_RUN_GROUP,
                         config={
                             "lag_ms": LAG_MS,
                             "train_ratio": TRAIN_RATIO,
                             "epochs": EPOCHS_PER_TRIAL,
                             "pca_dim": PCA_DIM,
                             "seed_base": SEED_BASE,
                         })
        # Attach data summary
        wandb.run.summary.update({"data_S": data_info["S"], "data_L": data_info["L"],
                                  "data_T": data_info["T"], "data_Emax": data_info["Emax"]})
        try:
            r = run_once()
        finally:
            run.finish()

    # Launch a single local agent that executes SWEEP_TRIALS runs sequentially
    wandb.agent(sweep_id, function=_agent, count=SWEEP_TRIALS)

    print("[DONE] Sweep finished.")