"""
Word-identity decoding experiments for SRM-VAE and classic SRM representations.

This script mirrors the data loading utilities from `vae_srm_cross_reconstruction.py`
but flips the direction of the predictive mapping: instead of predicting neural
activity from language embeddings, we predict the discrete word identity from
neural measurements (original, shared, and reconstructed spaces).

Comparisons:
    1) Original z-scored electrode activity per subject.
    2) SRM-VAE shared latent (mean aggregated posterior across subjects).
    3) SRM-VAE reconstructed signals per subject (decoder outputs).
    4) Classic SRM shared space (brainiak).
    5) Classic SRM reconstructed per-subject signals.

For every representation we fit a multi-class classifier that maps neural
features to discrete word IDs supplied by the user (or inferred from embeddings
if explicitly requested).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from brainiak.funcalign import srm as brainiak_srm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# --- Editable defaults -----------------------------------------------------
# Change these values instead of passing long argument lists on the CLI.
DATA_PATH_DEFAULT = "./all_data.pkl"
SEED_DEFAULT = 1234
TRAIN_RATIO_DEFAULT = 0.8
LAG_LIST_DEFAULT = "-500,100,1500"

VAE_K_DEFAULT = 5
VAE_EPOCHS_DEFAULT = 750
VAE_LR_DEFAULT = 1e-3
VAE_BETA_DEFAULT = 0.1
VAE_SELF_WEIGHT_DEFAULT = 1.0
VAE_CROSS_WEIGHT_DEFAULT = 1.0

SRM_K_DEFAULT = 5
SRM_ITERS_DEFAULT = 25

CLF_HIDDEN_DIMS_DEFAULT = "256,256"
CLF_DROPOUT_DEFAULT = 0.1
CLF_LR_DEFAULT = 2e-3
CLF_WEIGHT_DECAY_DEFAULT = 1e-4
CLF_EPOCHS_DEFAULT = 150
CLF_BATCH_SIZE_DEFAULT = 256
CLF_LOG_INTERVAL_DEFAULT = 25

WORD_LABELS_PATH_DEFAULT: Optional[str] = None
INFER_WORD_LABELS_DEFAULT = False
WORD_LABEL_ROUND_DECIMALS_DEFAULT = 4

FORCE_CPU_DEFAULT = False
RESULTS_DIR_DEFAULT = "./results/word_decoding"
SAVE_JSON_DEFAULT = False
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vae_srm_cross_reconstruction as core


@dataclass
class WordLabelSet:
    """Holds train/test word identity labels."""

    train: np.ndarray
    test: np.ndarray
    num_classes: int
    vocab: Optional[List[str]] = None


@dataclass
class ClassifierConfig:
    hidden_dims: List[int]
    dropout: float
    lr: float
    weight_decay: float
    epochs: int
    batch_size: int
    log_interval: int
    device: torch.device


class WordClassifier(nn.Module):
    """Simple MLP for multi-class word decoding."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for hid in hidden_dims:
            layers.append(nn.Linear(prev, hid))
            layers.append(nn.LayerNorm(hid))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hid
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_hidden_dims(arg: str) -> List[int]:
    arg = arg.strip()
    if not arg or arg.lower() in {"none", "null"}:
        return []
    return [int(part) for part in arg.split(",") if part.strip()]


def parse_lag_list(arg: str) -> List[int]:
    arg = arg.strip()
    if not arg:
        raise ValueError("Lag list cannot be empty.")
    return [int(tok) for tok in arg.split(",") if tok.strip()]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def standardize_features(
    train_feat: np.ndarray, test_feat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    train_std = scaler.fit_transform(train_feat)
    test_std = scaler.transform(test_feat)
    return train_std.astype(np.float32), test_std.astype(np.float32)


def train_word_classifier(
    train_feat: np.ndarray,
    test_feat: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    cfg: ClassifierConfig,
    prefix: str,
) -> float:
    if train_feat.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {train_feat.shape}")

    X_train, X_test = standardize_features(train_feat, test_feat)
    y_train = y_train.astype(np.int64, copy=False)
    y_test = y_test.astype(np.int64, copy=False)

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = WordClassifier(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        if cfg.log_interval > 0 and (epoch == 1 or epoch % cfg.log_interval == 0):
            avg_loss = epoch_loss / len(train_ds)
            print(f"    [{prefix}] epoch {epoch:03d}/{cfg.epochs} loss={avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).to(cfg.device, non_blocking=True)
        y_test_tensor = torch.from_numpy(y_test).to(cfg.device, non_blocking=True)
        logits = model(X_test_tensor)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_test_tensor).float().mean().item()
    return float(acc)


def load_word_labels_from_file(path: str) -> Tuple[np.ndarray, Optional[List[str]]]:
    _, ext = os.path.splitext(path.lower())
    if ext == ".npy":
        raw = np.load(path)
    elif ext == ".npz":
        raw = np.load(path)["arr_0"]
    elif ext in {".pkl", ".pickle"}:
        with open(path, "rb") as handle:
            raw = pickle.load(handle)
    else:
        with open(path, "r", encoding="utf-8") as handle:
            raw = [line.strip() for line in handle if line.strip()]

    arr = np.asarray(raw)
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    vocab: Optional[List[str]] = None
    if arr.dtype.kind in {"U", "S", "O"}:
        vocab, encoded = np.unique(arr, return_inverse=True)
        arr = encoded
        vocab = [str(v) for v in vocab]
    arr = arr.astype(np.int64, copy=False)
    return arr, vocab


def infer_word_labels_from_embeddings(
    embeddings: np.ndarray, decimals: int
) -> np.ndarray:
    labels = np.empty(embeddings.shape[0], dtype=np.int64)
    lookup: Dict[bytes, int] = {}
    for idx, vec in enumerate(embeddings):
        key = np.round(vec, decimals=decimals).tobytes()
        if key not in lookup:
            lookup[key] = len(lookup)
        labels[idx] = lookup[key]
    unique = len(lookup)
    if unique == embeddings.shape[0]:
        raise ValueError(
            "Every embedding vector is unique even after rounding; "
            "word labels cannot be inferred. Please provide --word_labels_path."
        )
    return labels


def build_word_label_set(
    embeddings: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    labels_path: Optional[str],
    infer_from_embeddings: bool,
    rounding_decimals: int,
) -> WordLabelSet:
    if labels_path:
        all_labels, vocab = load_word_labels_from_file(labels_path)
        if len(all_labels) != embeddings.shape[0]:
            raise ValueError(
                f"Label file length ({len(all_labels)}) "
                f"does not match number of time points ({embeddings.shape[0]})."
            )
    elif infer_from_embeddings:
        all_labels = infer_word_labels_from_embeddings(
            embeddings, decimals=rounding_decimals
        )
        vocab = None
        print(
            f"  Inferred {all_labels.max() + 1} unique labels "
            f"using rounding={rounding_decimals}."
        )
    else:
        raise ValueError(
            "Word labels are required. Provide --word_labels_path "
            "or enable --infer_word_labels."
        )

    y_train = all_labels[train_idx]
    y_test = all_labels[test_idx]
    num_classes = int(all_labels.max() + 1)
    return WordLabelSet(train=y_train, test=y_test, num_classes=num_classes, vocab=vocab)


def extract_original_features(batch: core.LagBatch) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    feats: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for sid in batch.subjects:
        view = batch.subject_views[sid]
        feats[sid] = (
            view.train.detach().cpu().numpy(),
            view.test.detach().cpu().numpy(),
        )
    return feats


def train_vae_model(
    batch: core.LagBatch,
    epochs: int,
    lr: float,
    beta: float,
    alpha: float,
    gamma: float,
) -> core.SRMVAE:
    model = core.train_srmvae_on_batch(
        batch,
        epochs=epochs,
        lr=lr,
        beta=beta,
        alpha=alpha,
        gamma=gamma,
        verbose=True,
    )
    model.eval()
    return model


def extract_vae_features(
    batch: core.LagBatch, model: core.SRMVAE
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    device = next(model.parameters()).device
    backups = _subject_views_to_device(batch.subject_views, device)
    try:
        with torch.no_grad():
            z_train, _, _ = model.infer_z(batch.subject_views, split="train", use_mu=True)
            z_test, _, _ = model.infer_z(batch.subject_views, split="test", use_mu=True)

            recon_train = model.reconstruct_subjects(z_train)
            recon_test = model.reconstruct_subjects(z_test)
    finally:
        _restore_subject_views(batch.subject_views, backups)

    shared_train = z_train.detach().cpu().numpy()
    shared_test = z_test.detach().cpu().numpy()

    recon_feats: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for sid in batch.subjects:
        recon_feats[sid] = (
            recon_train[sid].detach().cpu().numpy(),
            recon_test[sid].detach().cpu().numpy(),
        )
    return shared_train, shared_test, recon_feats


def fit_classic_srm(
    batch: core.LagBatch, features: int, n_iter: int
) -> Tuple[brainiak_srm.SRM, np.ndarray, List[np.ndarray]]:
    train_data = [view.train.detach().cpu().numpy().T for view in batch.subject_views.values()]
    test_data = [view.test.detach().cpu().numpy().T for view in batch.subject_views.values()]
    srm = brainiak_srm.SRM(n_iter=n_iter, features=features)
    srm.fit(train_data)
    shared_test_list = srm.transform(test_data)
    shared_train = srm.s_.T  # [T_train, k]
    return srm, shared_train, shared_test_list


def extract_srm_reconstructions(
    batch: core.LagBatch,
    srm: brainiak_srm.SRM,
    shared_train: np.ndarray,
    shared_test_list: Sequence[np.ndarray],
) -> Tuple[np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    shared_test = np.mean(np.stack(shared_test_list, axis=0), axis=0).T

    recon_feats: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for idx, sid in enumerate(batch.subjects):
        w_i = srm.w_[idx]
        y_train = (w_i @ shared_train.T).T
        y_test = (w_i @ shared_test_list[idx]).T
        recon_feats[sid] = (y_train, y_test)

    return shared_test, recon_feats


def decode_per_subject(
    feature_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    labels: WordLabelSet,
    cfg: ClassifierConfig,
    prefix: str,
) -> Dict[int, float]:
    results: Dict[int, float] = {}
    for sid, (train_feat, test_feat) in feature_dict.items():
        acc = train_word_classifier(
            train_feat,
            test_feat,
            labels.train,
            labels.test,
            labels.num_classes,
            cfg,
            prefix=f"{prefix}-S{sid}",
        )
        results[int(sid)] = acc
    return results


def _subject_views_to_device(
    subject_views: Dict[int, core.SubjectView], device: torch.device
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    backups: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for sid, view in subject_views.items():
        backups[int(sid)] = (view.train, view.test)
        view.train = view.train.to(device)
        view.test = view.test.to(device)
    return backups


def _restore_subject_views(
    subject_views: Dict[int, core.SubjectView],
    backups: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for sid, view in subject_views.items():
        orig_train, orig_test = backups[int(sid)]
        view.train = orig_train
        view.test = orig_test


def summarize_subject_metrics(metrics: Dict[int, float]) -> float:
    return float(np.mean(list(metrics.values()))) if metrics else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Word decoding experiments for SRM-VAE vs SRM.")
    parser.add_argument("--data_path", type=str, default=DATA_PATH_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO_DEFAULT)
    parser.add_argument("--lag_list", type=str, default=LAG_LIST_DEFAULT)

    # VAE hyper-parameters
    parser.add_argument("--vae_k", type=int, default=VAE_K_DEFAULT)
    parser.add_argument("--vae_epochs", type=int, default=VAE_EPOCHS_DEFAULT)
    parser.add_argument("--vae_lr", type=float, default=VAE_LR_DEFAULT)
    parser.add_argument("--vae_beta", type=float, default=VAE_BETA_DEFAULT)
    parser.add_argument("--vae_self_weight", type=float, default=VAE_SELF_WEIGHT_DEFAULT)
    parser.add_argument("--vae_cross_weight", type=float, default=VAE_CROSS_WEIGHT_DEFAULT)

    # SRM hyper-parameters
    parser.add_argument("--srm_k", type=int, default=SRM_K_DEFAULT)
    parser.add_argument("--srm_iters", type=int, default=SRM_ITERS_DEFAULT)

    # Classifier hyper-parameters
    parser.add_argument("--clf_hidden_dims", type=str, default=CLF_HIDDEN_DIMS_DEFAULT)
    parser.add_argument("--clf_dropout", type=float, default=CLF_DROPOUT_DEFAULT)
    parser.add_argument("--clf_lr", type=float, default=CLF_LR_DEFAULT)
    parser.add_argument("--clf_weight_decay", type=float, default=CLF_WEIGHT_DECAY_DEFAULT)
    parser.add_argument("--clf_epochs", type=int, default=CLF_EPOCHS_DEFAULT)
    parser.add_argument("--clf_batch_size", type=int, default=CLF_BATCH_SIZE_DEFAULT)
    parser.add_argument("--clf_log_interval", type=int, default=CLF_LOG_INTERVAL_DEFAULT)

    # Word labels
    parser.add_argument("--word_labels_path", type=str, default=WORD_LABELS_PATH_DEFAULT)
    parser.add_argument("--infer_word_labels", action="store_true", default=INFER_WORD_LABELS_DEFAULT)
    parser.add_argument(
        "--word_label_round_decimals",
        type=int,
        default=WORD_LABEL_ROUND_DECIMALS_DEFAULT,
    )

    # Misc
    parser.add_argument("--force_cpu", action="store_true", default=FORCE_CPU_DEFAULT)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--save_json", action="store_true", default=SAVE_JSON_DEFAULT)

    args = parser.parse_args()

    lag_list = parse_lag_list(args.lag_list)

    # Override shared hyper-parameters inside the imported module.
    core.VAE_K = args.vae_k
    if args.force_cpu:
        core.torch_device = lambda: torch.device("cpu")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    core.set_global_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Y_data, elec_num, X_all, lags = core.load_all_data(args.data_path)
    _, _, T, _ = Y_data.shape
    train_idx, test_idx = core.time_split_indices(T, args.train_ratio)

    label_set = build_word_label_set(
        embeddings=np.asarray(X_all),
        train_idx=train_idx,
        test_idx=test_idx,
        labels_path=args.word_labels_path,
        infer_from_embeddings=args.infer_word_labels,
        rounding_decimals=args.word_label_round_decimals,
    )

    clf_cfg = ClassifierConfig(
        hidden_dims=parse_hidden_dims(args.clf_hidden_dims),
        dropout=args.clf_dropout,
        lr=args.clf_lr,
        weight_decay=args.clf_weight_decay,
        epochs=args.clf_epochs,
        batch_size=args.clf_batch_size,
        log_interval=args.clf_log_interval,
        device=device,
    )

    ensure_dir(args.results_dir)
    all_results: Dict[int, Dict[str, object]] = {}

    for lag_ms in lag_list:
        print(f"\n=== Lag {lag_ms} ms ===")
        batch = core.build_lag_batch_from_loaded(
            Y_data, elec_num, X_all, lags, lag_ms, args.train_ratio
        )

        # 1) Original data per subject
        original_feats = extract_original_features(batch)
        original_results = decode_per_subject(
            original_feats, label_set, clf_cfg, prefix="orig"
        )
        print(
            f"  Original (mean accuracy): {summarize_subject_metrics(original_results):.4f}"
        )

        # 2 & 3) Train SRM-VAE for shared space + recon.
        print("  Training SRM-VAE...")
        vae_model = train_vae_model(
            batch,
            epochs=args.vae_epochs,
            lr=args.vae_lr,
            beta=args.vae_beta,
            alpha=args.vae_self_weight,
            gamma=args.vae_cross_weight,
        )

        # Move subject tensors back to CPU for downstream SRM fitting
        for sid in batch.subjects:
            view = batch.subject_views[sid]
            view.train = view.train.detach().cpu()
            view.test = view.test.detach().cpu()

        shared_train, shared_test, vae_recon_feats = extract_vae_features(batch, vae_model)
        vae_shared_acc = train_word_classifier(
            shared_train,
            shared_test,
            label_set.train,
            label_set.test,
            label_set.num_classes,
            clf_cfg,
            prefix="vae-shared",
        )
        print(f"  VAE shared accuracy: {vae_shared_acc:.4f}")

        vae_recon_results = decode_per_subject(
            vae_recon_feats, label_set, clf_cfg, prefix="vae-recon"
        )
        print(
            f"  VAE recon (mean accuracy): {summarize_subject_metrics(vae_recon_results):.4f}"
        )

        # 4 & 5) Classic SRM shared + reconstructed
        print("  Fitting classic SRM...")
        srm_model, srm_shared_train, srm_shared_test_list = fit_classic_srm(
            batch, features=args.srm_k, n_iter=args.srm_iters
        )
        srm_shared_test, srm_recon_feats = extract_srm_reconstructions(
            batch, srm_model, srm_shared_train, srm_shared_test_list
        )

        srm_shared_acc = train_word_classifier(
            srm_shared_train,
            srm_shared_test,
            label_set.train,
            label_set.test,
            label_set.num_classes,
            clf_cfg,
            prefix="srm-shared",
        )
        print(f"  SRM shared accuracy: {srm_shared_acc:.4f}")

        srm_recon_results = decode_per_subject(
            srm_recon_feats, label_set, clf_cfg, prefix="srm-recon"
        )
        print(
            f"  SRM recon (mean accuracy): {summarize_subject_metrics(srm_recon_results):.4f}"
        )

        all_results[int(lag_ms)] = {
            "original": original_results,
            "vae_shared": vae_shared_acc,
            "vae_recon": vae_recon_results,
            "srm_shared": srm_shared_acc,
            "srm_recon": srm_recon_results,
        }

    if args.save_json:
        out_path = os.path.join(
            args.results_dir,
            f"word_decoding_seed{args.seed}.json",
        )
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": vars(args),
                    "results": all_results,
                },
                handle,
                indent=2,
            )
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()

