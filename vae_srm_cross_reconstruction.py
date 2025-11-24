# new_script_reconstruction.py
# --- CONFIG (editable) ---
DATA_PATH   = "./all_data.pkl"
SEED        = 1234
TRAIN_RATIO = 0.8
LAG_LIST    =  [-500,100, 1500] # Expanded list for better plots
PCA_DIM     = 50

# VAE & SRM Hyperparameters
SRM_K       = 5      # Latent dimension for classic SRM
VAE_K       = 5      # Latent dimension for VAE
VAE_EPOCHS  = 1000    # Number of training epochs
VAE_LR      = 1e-3   # Learning rate
VAE_BETA    = 0   # Weight of the KL divergence term

# Non-linear encoder/decoder architecture settings
ENCODER_HIDDEN_DIMS = [64]  # Hidden sizes for per-subject encoders
DECODER_HIDDEN_DIMS = [64]  # Hidden sizes for per-subject decoders

# Plotting settings
PLOTS_FOLDER = "./reconstruction_plots"  # Folder to save plots
PLOT_FORMAT = "png"                      # Plot file format (png, pdf, svg)
PLOT_DPI = 150                           # Plot resolution

# --- Imports and Helper Functions (same as original script) ---
import os
import pickle
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import LinearRegression
from brainiak.funcalign import srm as brainiak_srm

# Settings and helper functions copied from original script for standalone execution
def set_global_seed(seed: int = 1234) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_plots_folder(folder_path):
    """Create plots folder if it doesn't exist"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created plots folder: {folder_path}")
    else:
        print(f"Using existing plots folder: {folder_path}")
    return folder_path

@dataclass
class SubjectView:
    train: torch.Tensor
    test:  torch.Tensor

@dataclass
class LagBatch:
    lag_ms: int
    subjects: List[int]
    subject_views: Dict[int, SubjectView]
    X_train: torch.Tensor
    X_test:  torch.Tensor
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
    return int(np.argmin(np.abs(lags_ms - target_ms)))

def time_split_indices(T: int, train_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    n_train = max(1, int(round(T * train_ratio)))
    n_train = min(n_train, T - 1)
    return np.arange(0, n_train, dtype=int), np.arange(n_train, T, dtype=int)

def build_lag_batch_from_loaded(Y_data, elec_num, X, lags, lag_ms, train_ratio):
    S, _, T, _ = Y_data.shape
    lag_idx = choose_lag_index(lags, lag_ms)
    train_index, test_index = time_split_indices(T, train_ratio)
    
    X_train, X_test = X[train_index, :], X[test_index, :]
    X_train_mean = X_train.mean(axis=0, keepdims=True)
    X_train_np = X_train - X_train_mean
    X_test_np = X_test - X_train_mean
    
    subject_views = {}
    per_sub_elec = {}
    subjects = list(range(1, S + 1))
    
    for s_idx, s_id in enumerate(subjects):
        e_i = int(elec_num[s_idx])
        mat = Y_data[s_idx, lag_idx, :, :e_i]
        tr, te = mat[train_index, :], mat[test_index, :]
        tr_z, te_z = _zscore_train_apply(tr, te)
        subject_views[s_id] = SubjectView(
            train=torch.from_numpy(tr_z).float(),
            test=torch.from_numpy(te_z).float()
        )
        per_sub_elec[s_id] = e_i
        
    return LagBatch(
        lag_ms=int(lags[lag_idx]), subjects=subjects, subject_views=subject_views,
        X_train=torch.from_numpy(X_train_np).float(),
        X_test=torch.from_numpy(X_test_np).float(),
        elec_num=per_sub_elec
    )

def _prep_embeddings(X_train, X_test, pca_dim, seed):
    pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=seed)
    X_train_p = pca.fit_transform(X_train)
    X_test_p = pca.transform(X_test)
    return normalize(X_train_p, axis=1), normalize(X_test_p, axis=1)

def _colwise_pearsonr(y_true, y_pred, eps=1e-8):
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0)) + eps
    return num / den

# --- Helper function for encoding calculation ---
def run_encoding_on_data(Y_train, Y_test, X_train_p, X_test_p):
    """Helper function to run encoding on data and return average correlation"""
    corrs = []
    num_electrodes = Y_train.shape[1]
    for elec_idx in range(num_electrodes):
        reg = LinearRegression().fit(X_train_p, Y_train[:, elec_idx])
        pred = reg.predict(X_test_p)
        corr = np.corrcoef(Y_test[:, elec_idx], pred)[0, 1]
        if not np.isnan(corr):
            corrs.append(corr)
    return np.mean(corrs) if corrs else np.nan

# --- SRM-VAE Model and Functions ---

def _init_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class PerSubjectEncoder(nn.Module):
    """Non-linear encoder per subject with ReLU activations."""

    def __init__(self, e_i: int, k: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        hidden_dims = hidden_dims or []
        self.hidden_layers = nn.ModuleList()
        prev = e_i
        for dim in hidden_dims:
            layer = nn.Linear(prev, dim, bias=True)
            _init_linear(layer)
            self.hidden_layers.append(layer)
            prev = dim
        self.output_layer = nn.Linear(prev, 2 * k, bias=True)
        _init_linear(self.output_layer)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        h = self.output_layer(h)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar.clamp(min=-8.0, max=8.0)


class PerSubjectDecoder(nn.Module):
    """Non-linear decoder per subject with ReLU activations."""

    def __init__(self, k: int, e_i: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        hidden_dims = hidden_dims or []
        self.hidden_layers = nn.ModuleList()
        prev = k
        for dim in hidden_dims:
            layer = nn.Linear(prev, dim, bias=True)
            _init_linear(layer)
            self.hidden_layers.append(layer)
            prev = dim
        self.output_layer = nn.Linear(prev, e_i, bias=False)
        _init_linear(self.output_layer)

    def forward(self, zf: torch.Tensor) -> torch.Tensor:
        h = zf
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        return self.output_layer(h)

class SRMVAE(nn.Module):
    """
    Encoders per subject -> precision-weighted group posterior -> shared core f(z)
    -> decoders per subject (SRM-like).
    """
    def __init__(
        self,
        elec_num: Dict[int, int],
        k: int,
        encoder_hidden: Optional[List[int]] = None,
        decoder_hidden: Optional[List[int]] = None,
    ):
        super().__init__()
        self.k = k
        self.encoder_hidden = encoder_hidden if encoder_hidden is not None else ENCODER_HIDDEN_DIMS
        self.decoder_hidden = decoder_hidden if decoder_hidden is not None else DECODER_HIDDEN_DIMS
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for sid, e_i in elec_num.items():
            self.encoders[str(sid)] = PerSubjectEncoder(e_i, k, self.encoder_hidden)
            self.decoders[str(sid)] = PerSubjectDecoder(k, e_i, self.decoder_hidden)
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
        
        # Orthogonality penalty for decoders
        ortho_pen = torch.tensor(0.0, device=z.device)
        for dec in self.decoders.values():
            W = dec.output_layer.weight  # [E_i, k]
            G = W.T @ W
            I = torch.eye(G.shape[0], device=G.device)
            ortho_pen = ortho_pen + torch.norm(G - I, p='fro')**2
        
        loss = recon_loss + beta * kl + 1e-3 * ortho_pen
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

def train_srmvae_on_batch(
    batch: LagBatch, 
    epochs: int, 
    lr: float, 
    beta: float, 
    alpha: float, 
    gamma: float, 
    verbose: bool = True
) -> SRMVAE:
    dev = torch_device()
    for sid in batch.subjects:
        sv = batch.subject_views[sid]
        sv.train = sv.train.to(dev)
        sv.test  = sv.test.to(dev)

    model = SRMVAE(batch.elec_num, k=VAE_K).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = math.inf
    best_state = None
    
    num_subjects = len(batch.subjects)
    subject_ids = list(batch.subjects)

    for ep in range(1, epochs + 1):
        model.train()
        
        # Fair training loop
        epoch_loss, epoch_self, epoch_cross, epoch_kl = 0, 0, 0, 0
        
        # We can't use batch size in the traditional sense, so we define one training step
        # as one pass through all decoders as targets
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

# --- Stage 1: Adding new functions ---

# Task 1.2: Function to calculate baseline performance (Baseline)
def calculate_original_encoding(batch: LagBatch, pca_dim: int, seed: int) -> Dict[int, float]:
    """
    Calculates encoding performance on original ECoG data.
    The function runs on each subject and each electrode separately and returns the average correlation per subject.
    """
    print(f"  Calculating baseline encoding for lag {batch.lag_ms}ms...")
    
    # 1. Prepare word embeddings (PCA -> L2 norm)
    X_train_p, X_test_p = _prep_embeddings(batch.X_train.cpu().numpy(), batch.X_test.cpu().numpy(), pca_dim, seed)
    
    results_per_subject = {}

    # 2. Loop over all subjects
    for s_id in batch.subjects:
        subject_data = batch.subject_views[s_id]
        Y_train = subject_data.train.cpu().numpy()
        Y_test = subject_data.test.cpu().numpy()
        
        # Use helper function for encoding calculation
        results_per_subject[s_id] = run_encoding_on_data(Y_train, Y_test, X_train_p, X_test_p)
            
    return results_per_subject

# Task 1.1: Full implementation of VAE analysis
def calculate_reconstructed_encoding_vae(batch: LagBatch, pca_dim: int, seed: int, args) -> Dict[int, float]:
    """
    Implements the full process: VAE training, signal reconstruction, and running encoding on reconstructed signals.
    """
    print(f"  Training SRM-VAE for lag {batch.lag_ms}ms...")
    
    # 1. Train SRM-VAE model
    model = train_srmvae_on_batch(
        batch, 
        epochs=args.epochs, 
        lr=args.lr, 
        beta=args.beta,
        alpha=args.self_recon_weight,
        gamma=args.cross_recon_weight,
        verbose=True
    )
    model.eval()
    
    # 2. Inference and reconstruction (using aggregated posteriors for evaluation)
    z_train, _, _ = model.infer_z(batch.subject_views, split="train", use_mu=True)
    z_test, _, _ = model.infer_z(batch.subject_views, split="test", use_mu=True)
    
    reconstructed_train_dict = model.reconstruct_subjects(z_train)
    reconstructed_test_dict = model.reconstruct_subjects(z_test)
    
    # 3. Prepare word embeddings
    X_train_p, X_test_p = _prep_embeddings(batch.X_train.cpu().numpy(), batch.X_test.cpu().numpy(), pca_dim, seed)
    
    results_per_subject = {}
    
    # 4. Run encoding on reconstructed data using helper function
    for s_id in batch.subjects:
        Y_train_recon = reconstructed_train_dict[s_id].cpu().numpy()
        Y_test_recon = reconstructed_test_dict[s_id].cpu().numpy()
        
        results_per_subject[s_id] = run_encoding_on_data(Y_train_recon, Y_test_recon, X_train_p, X_test_p)
        
    return results_per_subject


# --- Task 4.1: New function for classic SRM ---
def calculate_reconstructed_encoding_srm(batch: LagBatch, k: int, pca_dim: int, seed: int) -> Dict[int, float]:
    """
    Calculates encoding performance using classic SRM from brainiak library.
    Fits SRM on training data and reconstructs signals for encoding evaluation.
    """
    print(f"  Fitting classic SRM for lag {batch.lag_ms}ms...")
    
    # Prepare data for brainiak (needs list of [electrodes, time] numpy arrays)
    train_data_list = [v.train.cpu().numpy().T for v in batch.subject_views.values()]
    test_data_list = [v.test.cpu().numpy().T for v in batch.subject_views.values()]

    # Fit SRM on training data
    srm = brainiak_srm.SRM(n_iter=20, features=k)
    srm.fit(train_data_list)

    # Transform data to shared space
    shared_train = srm.transform(train_data_list)
    shared_test = srm.transform(test_data_list)

    # Prepare word embeddings
    X_train_p, X_test_p = _prep_embeddings(batch.X_train.cpu().numpy(), batch.X_test.cpu().numpy(), pca_dim, seed)
    results = {}

    # Reconstruct and run encoding for each subject
    for i, s_id in enumerate(batch.subjects):
        w_subject = srm.w_[i]
        Y_train_recon = (w_subject @ shared_train[i]).T
        Y_test_recon = (w_subject @ shared_test[i]).T
        results[s_id] = run_encoding_on_data(Y_train_recon, Y_test_recon, X_train_p, X_test_p)
        
    return results


# --- Task 4.2: New plotting function ---
def plot_subject_results(subject_id, lags_list, all_original, all_srm, all_vae, elec_num_dict, plots_folder, plot_format, plot_dpi):
    """
    Creates a plot for a specific subject showing encoding performance across lags
    for all three methods: Original, Classic SRM, and SRM-VAE.
    Saves the plot to the specified folder instead of displaying it.
    """
    # Extract results for the specific subject
    original_r = [res[subject_id] for res in all_original]
    srm_r = [res[subject_id] for res in all_srm]
    vae_r = [res[subject_id] for res in all_vae]
    
    num_electrodes = elec_num_dict.get(subject_id, 'N/A')

    plt.figure(figsize=(5, 4))
    plt.plot(lags_list, original_r, label='Original', color='steelblue', linewidth=3)
    plt.plot(lags_list, srm_r, label='SRM (Classic)', color='darkorange', linewidth=3)
    plt.plot(lags_list, vae_r, label='SRM-VAE', color='green', linewidth=3)
    
    plt.axvline(0, ls='--', color='grey', alpha=0.7)
    plt.xlabel("lags (ms)")
    plt.ylabel("Encoding Performance (r)")
    plt.title(f"Encoding S{subject_id} ({num_electrodes} electrodes)")
    plt.legend()
    plt.ylim(bottom=0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Save plot instead of showing it
    plot_filename = f"subject_{subject_id}_encoding.{plot_format}"
    plot_path = os.path.join(plots_folder, plot_filename)
    plt.savefig(plot_path, dpi=plot_dpi, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"  Saved plot for Subject {subject_id} to: {plot_path}")

# --- NEW FUNCTION: Average plot across all subjects ---

def plot_average_results(lags_list, all_original, all_srm, all_vae, plots_folder, plot_format, plot_dpi):
    """
    Creates and saves a plot showing the average encoding performance across **all** subjects
    for the three methods (Original, Classic SRM, SRM-VAE) as a function of lag.
    """
    import numpy as np  # Local import in case file-level import order changes

    # Calculate mean and standard error of the mean (SEM) across subjects for each lag
    avg_original, sem_original = [], []
    avg_srm,      sem_srm      = [], []
    avg_vae,      sem_vae      = [], []

    for res_o, res_s, res_v in zip(all_original, all_srm, all_vae):
        vals_o = np.array(list(res_o.values()))
        vals_s = np.array(list(res_s.values()))
        vals_v = np.array(list(res_v.values()))

        # Helper to compute mean and sem while ignoring NaNs
        def _mean_sem(arr):
            mean = np.nanmean(arr)
            n    = np.count_nonzero(~np.isnan(arr))
            sem  = np.nanstd(arr) / np.sqrt(n) if n > 0 else np.nan
            return mean, sem

        m_o, s_o = _mean_sem(vals_o)
        m_s, s_s = _mean_sem(vals_s)
        m_v, s_v = _mean_sem(vals_v)

        avg_original.append(m_o); sem_original.append(s_o)
        avg_srm.append(m_s);      sem_srm.append(s_s)
        avg_vae.append(m_v);      sem_vae.append(s_v)

    plt.figure(figsize=(5, 4))

    # Plot mean lines
    plt.plot(lags_list, avg_original, label='Original', color='steelblue', linewidth=3)
    plt.plot(lags_list, avg_srm,      label='SRM (Classic)', color='darkorange', linewidth=3)
    plt.plot(lags_list, avg_vae,      label='SRM-VAE', color='green', linewidth=3)

    # Add SEM shaded regions
    plt.fill_between(lags_list,
                     np.array(avg_original) - np.array(sem_original),
                     np.array(avg_original) + np.array(sem_original),
                     color='steelblue', alpha=0.25)
    plt.fill_between(lags_list,
                     np.array(avg_srm) - np.array(sem_srm),
                     np.array(avg_srm) + np.array(sem_srm),
                     color='darkorange', alpha=0.25)
    plt.fill_between(lags_list,
                     np.array(avg_vae) - np.array(sem_vae),
                     np.array(avg_vae) + np.array(sem_vae),
                     color='green', alpha=0.25)

    plt.axvline(0, ls='--', color='grey', alpha=0.7)
    plt.xlabel("lags (ms)")
    plt.ylabel("Encoding Performance (r)")
    plt.title("Average Encoding Performance (All Subjects)")
    plt.legend()
    plt.ylim(bottom=0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    plot_filename = f"average_encoding.{plot_format}"
    plot_path = os.path.join(plots_folder, plot_filename)
    plt.savefig(plot_path, dpi=plot_dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved average plot to: {plot_path}")

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run VAE-SRM Cross-Reconstruction Experiment")
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help="Path to the data pkl file")
    parser.add_argument('--seed', type=int, default=SEED, help="Global random seed")
    parser.add_argument('--train_ratio', type=float, default=TRAIN_RATIO, help="Train/test split ratio")
    parser.add_argument('--pca_dim', type=int, default=PCA_DIM, help="PCA dimensionality for embeddings")
    parser.add_argument('--srm_k', type=int, default=SRM_K, help="Latent dimensions for classic SRM")
    parser.add_argument('--vae_k', type=int, default=VAE_K, help="Latent dimensions for VAE-SRM")
    parser.add_argument('--epochs', type=int, default=VAE_EPOCHS, help="Number of training epochs for VAE")
    parser.add_argument('--lr', type=float, default=VAE_LR, help="Learning rate for VAE")
    parser.add_argument('--beta', type=float, default=VAE_BETA, help="Weight for the KL divergence term")
    parser.add_argument('--self_recon_weight', type=float, default=1.0, help="Weight for the self-reconstruction loss (alpha)")
    parser.add_argument('--cross_recon_weight', type=float, default=1.0, help="Weight for the cross-reconstruction loss (gamma)")
    parser.add_argument('--plots_folder', type=str, default=PLOTS_FOLDER, help="Folder to save plots")
    args = parser.parse_args()

    # Use global variables from args
    DATA_PATH, SEED, TRAIN_RATIO, PCA_DIM, SRM_K, VAE_K, VAE_EPOCHS, VAE_LR, VAE_BETA, PLOTS_FOLDER = \
        args.data_path, args.seed, args.train_ratio, args.pca_dim, args.srm_k, args.vae_k, \
        args.epochs, args.lr, args.beta, args.plots_folder

    set_global_seed(SEED)
    
    # Load data
    try:
        Y_data, elec_num, X, lags = load_all_data(DATA_PATH)
        print(f"Data loaded successfully. Y_data shape: {Y_data.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load data from {DATA_PATH}: {e}")
        exit()

    # Create subject list and electrode number dictionary
    subjects_list = list(range(1, len(elec_num) + 1))
    elec_num_dict = {s_id: num for s_id, num in zip(subjects_list, elec_num)}

    # Data structures to store results for all three methods
    all_original_results, all_srm_results, all_vae_results = [], [], []

    # Main loop over lags
    for lag_ms in LAG_LIST:
        print(f"\nProcessing Lag: {lag_ms}ms")
        
        # Prepare batch for current lag
        batch = build_lag_batch_from_loaded(Y_data, elec_num, X, lags, lag_ms, TRAIN_RATIO)
        
        # 1. Original encoding (baseline)
        original_results = calculate_original_encoding(batch, PCA_DIM, SEED)
        all_original_results.append(original_results)
        print(f"  > Original Avg r: {np.nanmean(list(original_results.values())):.4f}")

        # 2. Classic SRM encoding
        srm_results = calculate_reconstructed_encoding_srm(batch, SRM_K, PCA_DIM, SEED)
        all_srm_results.append(srm_results)
        print(f"  > Classic SRM Avg r: {np.nanmean(list(srm_results.values())):.4f}")

        # 3. SRM-VAE encoding
        vae_results = calculate_reconstructed_encoding_vae(batch, PCA_DIM, SEED, args)
        all_vae_results.append(vae_results)
        print(f"  > SRM-VAE Avg r: {np.nanmean(list(vae_results.values())):.4f}")

    print("\n--- Analysis Complete ---")
    print("Generating plots for each subject...")

    # Ensure plots folder exists
    plots_folder = ensure_plots_folder(PLOTS_FOLDER)

    # Generate a plot for each subject
    for s_id in subjects_list:
        plot_subject_results(
            s_id, LAG_LIST, all_original_results, all_srm_results, all_vae_results, 
            elec_num_dict, plots_folder, PLOT_FORMAT, PLOT_DPI
        )
    
    # Generate and save an aggregate plot across all subjects
    print("Generating average plot over all subjects...")
    plot_average_results(
        LAG_LIST, all_original_results, all_srm_results, all_vae_results,
        plots_folder, PLOT_FORMAT, PLOT_DPI
    )
    
    print(f"\nAll plots saved to: {plots_folder}")
    print("Analysis and plotting complete!") 