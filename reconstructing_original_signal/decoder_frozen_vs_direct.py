# -------------------------------------------------------------
# decoder_frozen_vs_direct.py
#
# Compare two strategies for predicting subject ECoG signals
# from word embeddings once a shared SRM-VAE has been trained:
#   1. Freeze the subject-specific decoders and train a small
#      network f: embedding -> latent space, optimized so that
#      d_i(f(h_t)) matches each subject signal s_i,t.
#   2. Train the exact same architecture directly on embeddings
#      to predict each subject's signals without using decoders.
#
# The script loops over lags, reports average correlations, and
# saves subject-level + average lag-vs-correlation plots.
# -------------------------------------------------------------

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# -----------------------------
# Default configuration
# -----------------------------
DATA_PATH = "./all_data.pkl"
SEED = 1234
TRAIN_RATIO = 0.8
LAG_LIST = [-1000,-500,-100,0, 200,1000]
PCA_DIM = 50

# VAE hyperparameters
VAE_LATENT_DIM = 64
VAE_EPOCHS = 800
VAE_LR = 1e-3
VAE_BETA = 1
SELF_RECON_WEIGHT = 0.2
CROSS_RECON_WEIGHT = 2.0
ENC_HIDDEN_DIMS = [256, 128]
DEC_HIDDEN_DIMS = [128, 256]

# Embedding -> latent regressor (f)
F_HIDDEN_DIMS = [128, 64]
F_EPOCHS = 800
F_LR = 5e-3
F_WEIGHT_DECAY = 0
F_BATCH_SIZE = 128

# Direct predictor (same architecture as f)
DIRECT_EPOCHS = 800
DIRECT_LR = 5e-3
DIRECT_WEIGHT_DECAY = 0
DIRECT_BATCH_SIZE = 128

# Logging
LOG_INTERVAL = 50  # set to 1 for per-epoch updates

# Plot directory (relative to this file)
_HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(_HERE, "plots_decoder_vs_direct")


# -----------------------------
# Utilities
# -----------------------------
def set_global_seed(seed: int = 1234) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_device(preferred: Optional[str] = None) -> torch.device:
    """Select device, honoring a preferred string if provided."""
    if preferred:
        pref = preferred.lower()
        if pref in ("cpu",):
            return torch.device("cpu")
        if pref in ("cuda", "gpu"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            print("[WARN] requested GPU but CUDA not available, using CPU.")
            return torch.device("cpu")
        print(f"[WARN] Unknown device '{preferred}', falling back to auto.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


@dataclass
class SubjectView:
    train: torch.Tensor  # [T_train, E_i]
    test: torch.Tensor   # [T_test , E_i]
    mask_train: Optional[torch.Tensor] = None
    mask_test: Optional[torch.Tensor] = None


@dataclass
class LagBatch:
    lag_ms: int
    latent_dim: int
    subjects: List[int]
    subject_views: Dict[int, SubjectView]
    X_train: torch.Tensor  # [T_train, D_emb]
    X_test: torch.Tensor   # [T_test , D_emb]
    train_index: np.ndarray
    test_index: np.ndarray
    elec_num: Dict[int, int]


def load_all_data(pkl_path: str):
    import pickle

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    Y_data = np.asarray(obj["electrode_data"])          # [S, L, T, Emax]
    elec_num = np.asarray(obj["electrode_number"], int) # [S]
    X = np.asarray(obj["word_embeddings"])              # [T, D_emb]
    lags = np.asarray(obj["lags"]).reshape(-1)          # [L]
    return Y_data, elec_num, X, lags


def _zscore_train_apply(train: np.ndarray, test: np.ndarray):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mu) / sd, (test - mu) / sd


def choose_lag_index(lags_ms: np.ndarray, target_ms: int) -> int:
    diffs = np.abs(lags_ms - target_ms)
    return int(np.argmin(diffs))


def time_split_indices(T: int, train_ratio: float):
    n_train = max(1, int(round(T * train_ratio)))
    n_train = min(n_train, T - 1)  # need at least one test sample
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
    S, _, T, _ = Y_data.shape
    lag_idx = choose_lag_index(lags, lag_ms)
    chosen_ms = int(lags[lag_idx])

    train_index, test_index = time_split_indices(T, train_ratio)

    X_train = X[train_index, :]
    X_test = X[test_index, :]
    X_train_mean = X_train.mean(axis=0, keepdims=True)
    X_train_np = X_train - X_train_mean
    X_test_np = X_test - X_train_mean

    subjects = list(range(1, S + 1))
    subject_views: Dict[int, SubjectView] = {}
    per_sub_elec: Dict[int, int] = {}

    for s_idx, sid in enumerate(subjects):
        e_i = int(elec_num[s_idx])
        mat = Y_data[s_idx, lag_idx, :, :e_i]
        tr = mat[train_index, :]
        te = mat[test_index, :]
        tr_z, te_z = _zscore_train_apply(tr, te)
        mask = np.ones((e_i,), dtype=bool)
        subject_views[sid] = SubjectView(
            train=torch.from_numpy(tr_z).float(),
            test=torch.from_numpy(te_z).float(),
            mask_train=torch.from_numpy(mask),
            mask_test=torch.from_numpy(mask),
        )
        per_sub_elec[sid] = e_i

    return LagBatch(
        lag_ms=chosen_ms,
        latent_dim=latent_dim,
        subjects=subjects,
        subject_views=subject_views,
        X_train=torch.from_numpy(X_train_np).float(),
        X_test=torch.from_numpy(X_test_np).float(),
        train_index=train_index,
        test_index=test_index,
        elec_num=per_sub_elec,
    )


def _prep_embeddings(X_train: np.ndarray, X_test: np.ndarray, pca_dim: int, seed: int):
    pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=seed)
    X_train_p = pca.fit_transform(X_train)
    X_test_p = pca.transform(X_test)
    X_train_p = normalize(X_train_p, axis=1)
    X_test_p = normalize(X_test_p, axis=1)
    return X_train_p, X_test_p


def _colwise_pearsonr(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8):
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0)) + eps
    return num / den


def _mean_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    corr_per_dim = _colwise_pearsonr(y_true, y_pred)
    return float(np.nanmean(corr_per_dim))


# -----------------------------
# SRM-VAE model (non-linear enc/dec)
# -----------------------------
class PerSubjectEncoder(nn.Module):
    def __init__(self, e_i: int, k: int, hidden_dims: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        dim = e_i
        for hidden in hidden_dims:
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.ReLU())
            dim = hidden
        layers.append(nn.Linear(dim, 2 * k))
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
        layers: List[nn.Module] = []
        dim = k
        for hidden in hidden_dims:
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.ReLU())
            dim = hidden
        layers.append(nn.Linear(dim, e_i))
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SRMVAE(nn.Module):
    def __init__(
        self,
        elec_num: Dict[int, int],
        k: int,
        enc_hidden_dims: List[int],
        dec_hidden_dims: List[int],
    ):
        super().__init__()
        self.k = k
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for sid, e_i in elec_num.items():
            self.encoders[str(sid)] = PerSubjectEncoder(e_i, k, enc_hidden_dims)
            self.decoders[str(sid)] = PerSubjectDecoder(k, e_i, dec_hidden_dims)
        self.core = nn.Identity()

    @staticmethod
    def _agg_posteriors(mu_list, logvar_list):
        mu = torch.stack(mu_list, dim=0).mean(dim=0)
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

    def forward_cross(
        self,
        subject_views: Dict[int, SubjectView],
        source_sid: int,
        target_sid: int,
        alpha: float,
        gamma: float,
        beta: float,
    ):
        source_view = subject_views[source_sid]
        mu, logvar = self.encoders[str(source_sid)](source_view.train)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        zf = self.core(z)

        source_recon = self.decoders[str(source_sid)](zf)
        loss_self = F.mse_loss(source_recon, source_view.train, reduction="mean")

        target_view = subject_views[target_sid]
        target_recon = self.decoders[str(target_sid)](zf)
        loss_cross = F.mse_loss(target_recon, target_view.train, reduction="mean")

        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
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


def train_srmvae_on_batch(
    batch: LagBatch,
    epochs: int,
    lr: float,
    beta: float,
    alpha: float,
    gamma: float,
    enc_hidden_dims: List[int],
    dec_hidden_dims: List[int],
    device: torch.device,
    verbose: bool = True,
) -> SRMVAE:
    dev = device
    for sid in batch.subjects:
        sv = batch.subject_views[sid]
        batch.subject_views[sid] = SubjectView(
            train=sv.train.to(dev),
            test=sv.test.to(dev),
            mask_train=sv.mask_train.to(dev) if sv.mask_train is not None else None,
            mask_test=sv.mask_test.to(dev) if sv.mask_test is not None else None,
        )

    model = SRMVAE(
        batch.elec_num,
        k=batch.latent_dim,
        enc_hidden_dims=enc_hidden_dims,
        dec_hidden_dims=dec_hidden_dims,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = math.inf
    best_state = None
    subject_ids = list(batch.subjects)
    num_subjects = len(subject_ids)

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_self = 0.0
        epoch_cross = 0.0
        epoch_kl = 0.0

        # ensure every subject is used as target once per epoch
        random.shuffle(subject_ids)
        for target_sid in subject_ids:
            source_sid = random.choice(subject_ids)
            beta_ep = beta * min(1.0, ep / 500)
            loss, rec_self, rec_cross, kl = model.forward_cross(
                batch.subject_views,
                source_sid=source_sid,
                target_sid=target_sid,
                alpha=alpha,
                gamma=gamma,
                beta=beta_ep,
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

        if verbose and (ep % 50 == 0 or ep == 1):
            print(
                f"[VAE ep {ep:03d}] loss={avg_loss:.5f} "
                f"self={epoch_self/num_subjects:.5f} "
                f"cross={epoch_cross/num_subjects:.5f} "
                f"kl={epoch_kl/num_subjects:.5f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -----------------------------
# Embedding -> latent predictor
# -----------------------------
class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.ReLU())
            dim = hidden
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_loop_indices(n_samples: int, batch_size: int, device: torch.device):
    perm = torch.randperm(n_samples, device=device)
    for start in range(0, n_samples, batch_size):
        idx = perm[start : start + batch_size]
        yield idx


def train_decoder_frozen_regressor(
    batch: LagBatch,
    vae_model: SRMVAE,
    X_train_p: np.ndarray,
    X_test_p: np.ndarray,
    hidden_dims: List[int],
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    log_interval: int,
    device: torch.device,
) -> Dict[int, float]:
    dev = device
    vae_model = vae_model.to(dev).eval()

    X_train = torch.from_numpy(X_train_p).float().to(dev)
    X_test = torch.from_numpy(X_test_p).float().to(dev)
    n_train = X_train.shape[0]
    f_net = FeedForwardNet(X_train.shape[1], hidden_dims, vae_model.k).to(dev)

    for dec in vae_model.decoders.values():
        for p in dec.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(f_net.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        f_net.train()
        epoch_loss = 0.0
        for idx in _train_loop_indices(n_train, batch_size, device=dev):
            x_batch = X_train[idx]
            z = f_net(x_batch)

            loss = torch.tensor(0.0, device=dev)
            for sid in batch.subjects:
                y_true = batch.subject_views[sid].train[idx]
                pred = vae_model.decoders[str(sid)](z)
                loss = loss + F.mse_loss(pred, y_true, reduction="mean")
            loss = loss / len(batch.subjects)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * idx.numel()

        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            print(
                f"  [decoder-reg ep {epoch:03d}/{epochs}] "
                f"loss={epoch_loss / n_train:.6f}"
            )

    f_net.eval()
    corrs: Dict[int, float] = {}
    with torch.no_grad():
        z_test = f_net(X_test)
        for sid in batch.subjects:
            preds = vae_model.decoders[str(sid)](z_test).cpu().numpy()
            y_true = batch.subject_views[sid].test.detach().cpu().numpy()
            corrs[sid] = _mean_corr(y_true, preds)
    return corrs


# -----------------------------
# Direct predictor (baseline)
# -----------------------------
def train_direct_predictors(
    batch: LagBatch,
    X_train_p: np.ndarray,
    X_test_p: np.ndarray,
    hidden_dims: List[int],
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    log_interval: int,
    device: torch.device,
) -> Dict[int, float]:
    dev = device
    X_train = torch.from_numpy(X_train_p).float().to(dev)
    X_test = torch.from_numpy(X_test_p).float().to(dev)
    n_train = X_train.shape[0]

    corrs: Dict[int, float] = {}

    for sid in batch.subjects:
        view = batch.subject_views[sid]
        model = FeedForwardNet(X_train.shape[1], hidden_dims, view.train.shape[1]).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for idx in _train_loop_indices(n_train, batch_size, device=dev):
                x_batch = X_train[idx]
                y_true = view.train[idx]
                preds = model(x_batch)
                loss = F.mse_loss(preds, y_true, reduction="mean")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * idx.numel()

            if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
                print(
                    f"  [direct ep {epoch:03d}/{epochs}] "
                    f"sid={sid} loss={epoch_loss / n_train:.6f}"
                )

        model.eval()
        with torch.no_grad():
            preds = model(X_test).cpu().numpy()
        y_true = view.test.detach().cpu().numpy()
        corrs[sid] = _mean_corr(y_true, preds)

    return corrs


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_subject_curves(
    subject_id: int,
    lags_list: List[int],
    decoder_results: List[Dict[int, float]],
    direct_results: List[Dict[int, float]],
    out_dir: str,
):
    decoder_vals = [res[subject_id] for res in decoder_results]
    direct_vals = [res[subject_id] for res in direct_results]

    plt.figure(figsize=(5, 4))
    plt.plot(lags_list, decoder_vals, label="Frozen decoders", color="royalblue", linewidth=3)
    plt.plot(lags_list, direct_vals, label="Direct fit", color="darkorange", linewidth=3)
    plt.axvline(0, ls="--", color="grey", alpha=0.6)
    plt.xlabel("lags (ms)")
    plt.ylabel("Correlation (r)")
    plt.title(f"Subject {subject_id}")
    plt.ylim(bottom=0.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(out_dir, f"subject_{subject_id}_decoder_vs_direct.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved subject plot -> {fname}")


def plot_average_curve(
    lags_list: List[int],
    decoder_results: List[Dict[int, float]],
    direct_results: List[Dict[int, float]],
    out_dir: str,
):
    def _stack(values_list: List[Dict[int, float]]):
        ordered_subjects = sorted(values_list[0].keys())
        arr = np.stack(
            [[res[sid] for sid in ordered_subjects] for res in values_list],
            axis=0,
        )
        mean = np.nanmean(arr, axis=1)
        sem = np.nanstd(arr, axis=1) / np.sqrt(arr.shape[1])
        return mean, sem

    decode_mean, decode_sem = _stack(decoder_results)
    direct_mean, direct_sem = _stack(direct_results)

    plt.figure(figsize=(5, 4))
    plt.plot(lags_list, decode_mean, label="Frozen decoders", color="royalblue", linewidth=3)
    plt.plot(lags_list, direct_mean, label="Direct fit", color="darkorange", linewidth=3)
    plt.fill_between(
        lags_list,
        decode_mean - decode_sem,
        decode_mean + decode_sem,
        color="royalblue",
        alpha=0.25,
    )
    plt.fill_between(
        lags_list,
        direct_mean - direct_sem,
        direct_mean + direct_sem,
        color="darkorange",
        alpha=0.25,
    )
    plt.axvline(0, ls="--", color="grey", alpha=0.6)
    plt.xlabel("lags (ms)")
    plt.ylabel("Correlation (r)")
    plt.title("Average over subjects")
    plt.ylim(bottom=0.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(out_dir, "average_decoder_vs_direct.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved average plot -> {fname}")


# -----------------------------
# Main experiment loop
# -----------------------------
def run_experiment(args):
    set_global_seed(args.seed)
    dev = torch_device(args.device)
    print(f"[INFO] device={dev}, cuda={torch.cuda.is_available()}")

    Y_data, elec_num, X, lags = load_all_data(args.data_path)
    print(f"[DATA] Y={Y_data.shape}, X={X.shape}, lags={len(lags)}")

    ensure_dir(args.output_dir)

    decoder_results_all: List[Dict[int, float]] = []
    direct_results_all: List[Dict[int, float]] = []

    for lag_ms in args.lag_list:
        print(f"\n=== Lag {lag_ms} ms ===")
        batch = build_lag_batch_from_loaded(
            Y_data,
            elec_num,
            X,
            lags,
            lag_ms=lag_ms,
            latent_dim=args.vae_latent_dim,
            train_ratio=args.train_ratio,
        )
        print(f"[INFO] actual lag={batch.lag_ms} ms, latent_dim={batch.latent_dim}")

        vae_model = train_srmvae_on_batch(
            batch,
            epochs=args.vae_epochs,
            lr=args.vae_lr,
            beta=args.vae_beta,
            alpha=args.self_recon_weight,
            gamma=args.cross_recon_weight,
            enc_hidden_dims=args.enc_hidden_dims,
            dec_hidden_dims=args.dec_hidden_dims,
            device=dev,
            verbose=True,
        )

        X_train_p, X_test_p = _prep_embeddings(
            batch.X_train.cpu().numpy(),
            batch.X_test.cpu().numpy(),
            pca_dim=args.pca_dim,
            seed=args.seed,
        )

        decoder_corrs = train_decoder_frozen_regressor(
            batch,
            vae_model,
            X_train_p,
            X_test_p,
            hidden_dims=args.f_hidden_dims,
            lr=args.f_lr,
            weight_decay=args.f_weight_decay,
            epochs=args.f_epochs,
            batch_size=args.f_batch_size,
            log_interval=args.log_interval,
            device=dev,
        )
        decoder_results_all.append(decoder_corrs)
        print(
            f"  Frozen-decoder mean r = {np.nanmean(list(decoder_corrs.values())):.4f}"
        )

        direct_corrs = train_direct_predictors(
            batch,
            X_train_p,
            X_test_p,
            hidden_dims=args.f_hidden_dims,  # same architecture as f
            lr=args.direct_lr,
            weight_decay=args.direct_weight_decay,
            epochs=args.direct_epochs,
            batch_size=args.direct_batch_size,
            log_interval=args.log_interval,
            device=dev,
        )
        direct_results_all.append(direct_corrs)
        print(f"  Direct-fit mean r    = {np.nanmean(list(direct_corrs.values())):.4f}")

    subjects = batch.subjects
    for sid in subjects:
        plot_subject_curves(sid, args.lag_list, decoder_results_all, direct_results_all, args.output_dir)
    plot_average_curve(args.lag_list, decoder_results_all, direct_results_all, args.output_dir)

    print("\nFinished! Plots saved to:", args.output_dir)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compare decoder-frozen embedding regressors vs direct predictors."
    )
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--lag_list", type=int, nargs="+", default=LAG_LIST)
    parser.add_argument("--pca_dim", type=int, default=PCA_DIM)

    parser.add_argument("--vae_latent_dim", type=int, default=VAE_LATENT_DIM)
    parser.add_argument("--vae_epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--vae_lr", type=float, default=VAE_LR)
    parser.add_argument("--vae_beta", type=float, default=VAE_BETA)
    parser.add_argument("--self_recon_weight", type=float, default=SELF_RECON_WEIGHT)
    parser.add_argument("--cross_recon_weight", type=float, default=CROSS_RECON_WEIGHT)
    parser.add_argument("--enc_hidden_dims", type=int, nargs="+", default=ENC_HIDDEN_DIMS)
    parser.add_argument("--dec_hidden_dims", type=int, nargs="+", default=DEC_HIDDEN_DIMS)

    parser.add_argument("--f_hidden_dims", type=int, nargs="+", default=F_HIDDEN_DIMS)
    parser.add_argument("--f_epochs", type=int, default=F_EPOCHS)
    parser.add_argument("--f_lr", type=float, default=F_LR)
    parser.add_argument("--f_weight_decay", type=float, default=F_WEIGHT_DECAY)
    parser.add_argument("--f_batch_size", type=int, default=F_BATCH_SIZE)

    parser.add_argument("--direct_epochs", type=int, default=DIRECT_EPOCHS)
    parser.add_argument("--direct_lr", type=float, default=DIRECT_LR)
    parser.add_argument("--direct_weight_decay", type=float, default=DIRECT_WEIGHT_DECAY)
    parser.add_argument("--direct_batch_size", type=int, default=DIRECT_BATCH_SIZE)

    parser.add_argument("--log_interval", type=int, default=LOG_INTERVAL)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (auto if None)")
    parser.add_argument("--output_dir", type=str, default=PLOTS_DIR)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_experiment(args)

