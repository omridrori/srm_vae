"""
Word-embedding regression experiments for SRM-VAE and classic SRM representations.

This script mirrors the infrastructure in `vae_srm_cross_reconstruction.py`
but flips the direction of prediction: we fit regressors that map neural
features (original signals, shared latent, or reconstructions) back to the
continuous word-embedding vectors used as stimuli.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from brainiak.funcalign import srm as brainiak_srm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# --- Editable defaults -----------------------------------------------------
DATA_PATH_DEFAULT = "./all_data.pkl"
SEED_DEFAULT = 1234
TRAIN_RATIO_DEFAULT = 0.8
LAG_LIST_DEFAULT = "-500,100,1500"

VAE_K_DEFAULT = 10
VAE_EPOCHS_DEFAULT = 1500
VAE_LR_DEFAULT = 1e-4
VAE_BETA_DEFAULT = 0.0
VAE_SELF_WEIGHT_DEFAULT = 1.0
VAE_CROSS_WEIGHT_DEFAULT = 5.0

SRM_K_DEFAULT = 5
SRM_ITERS_DEFAULT = 25

RIDGE_ALPHA_DEFAULT = 1.0
TARGET_STANDARDIZE_DEFAULT = True
DEBUG_MODE_DEFAULT = True

FORCE_CPU_DEFAULT = False
RESULTS_DIR_DEFAULT = "./results/embedding_regression"
SAVE_JSON_DEFAULT = False
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vae_srm_cross_reconstruction as core


@dataclass
class RegressionConfig:
    alpha: float = 1.0
    standardize_targets: bool = True
    debug: bool = False


def standardize_pair(
    X_train: np.ndarray,
    X_test: np.ndarray,
    with_mean: bool = True,
    with_std: bool = True,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std.astype(np.float32), X_test_std.astype(np.float32), scaler


def regress_embeddings(
    feat_train: np.ndarray,
    feat_test: np.ndarray,
    embed_train: np.ndarray,
    embed_test: np.ndarray,
    cfg: RegressionConfig,
) -> Tuple[float, float]:
    if feat_train.shape[0] != embed_train.shape[0]:
        raise ValueError("Feature and embedding training sets must align in time.")

    X_train_std, X_test_std, feat_scaler = standardize_pair(feat_train, feat_test)
    if cfg.standardize_targets:
        Y_train_std, Y_test_std, target_scaler = standardize_pair(embed_train, embed_test)
        target = Y_train_std
    else:
        target = embed_train
        Y_test_std = embed_test
        target_scaler = None

    if cfg.debug:
        print("\n--- DEBUG: Regression Inputs ---")
        print(f"  Feat Train: mean={X_train_std.mean():.4f}, std={X_train_std.std():.4f}, min={X_train_std.min():.4f}, max={X_train_std.max():.4f}")
        print(f"  Target Train: mean={target.mean():.4f}, std={target.std():.4f}, min={target.min():.4f}, max={target.max():.4f}")
        if target_scaler:
             print(f"  Target Scaler: means (first 5)={target_scaler.mean_[:5]}, vars (first 5)={target_scaler.var_[:5]}")

    reg = Ridge(alpha=cfg.alpha)
    reg.fit(X_train_std, target)

    preds_std = reg.predict(X_test_std)
    if target_scaler is not None:
        preds = target_scaler.inverse_transform(preds_std)
        y_true = embed_test
    else:
        preds = preds_std
        y_true = Y_test_std

    mse = float(np.mean((y_true - preds) ** 2))
    corr = core._colwise_pearsonr(y_true, preds)
    
    if cfg.debug:
         print(f"  Prediction Stats: mean={preds.mean():.4f}, std={preds.std():.4f}, min={preds.min():.4f}, max={preds.max():.4f}")
         print(f"  True Embed Stats: mean={y_true.mean():.4f}, std={y_true.std():.4f}, min={y_true.min():.4f}, max={y_true.max():.4f}")
         print(f"  MSE={mse:.6f}, Corr={np.nanmean(corr):.4f}")
         print("--------------------------------")
         
    return float(np.nanmean(corr)), mse


def extract_original_features(batch: core.LagBatch) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    feats: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for sid in batch.subjects:
        view = batch.subject_views[sid]
        feats[int(sid)] = (
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
        recon_feats[int(sid)] = (
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
        recon_feats[int(sid)] = (y_train, y_test)
    return shared_test, recon_feats


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


def summarize_metrics(metrics: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
    if not metrics:
        return float("nan"), float("nan")
    cors = [val[0] for val in metrics.values()]
    mses = [val[1] for val in metrics.values()]
    return float(np.mean(cors)), float(np.mean(mses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Regress word embeddings from neural representations.")
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

    # Regression options
    parser.add_argument("--ridge_alpha", type=float, default=RIDGE_ALPHA_DEFAULT)
    parser.add_argument(
        "--no_target_standardize",
        action="store_true",
        default=not TARGET_STANDARDIZE_DEFAULT,
    )
    parser.add_argument("--debug", action="store_true", default=DEBUG_MODE_DEFAULT)

    parser.add_argument("--force_cpu", action="store_true", default=FORCE_CPU_DEFAULT)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--save_json", action="store_true", default=SAVE_JSON_DEFAULT)
    parser.add_argument("--use_contrastive", action="store_true", help="Use Contrastive (InfoNCE) training instead of Reconstruction")

    args = parser.parse_args()

    lag_list = [int(tok) for tok in args.lag_list.split(",") if tok.strip()]

    core.VAE_K = args.vae_k
    if args.force_cpu:
        core.torch_device = lambda: torch.device("cpu")

    core.set_global_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Y_data, elec_num, X_all, lags = core.load_all_data(args.data_path)

    cfg = RegressionConfig(
        alpha=args.ridge_alpha,
        standardize_targets=not args.no_target_standardize,
        debug=args.debug,
    )

    os.makedirs(args.results_dir, exist_ok=True)
    all_results: Dict[int, Dict[str, object]] = {}

    for lag_ms in lag_list:
        print(f"\n=== Lag {lag_ms} ms ===")
        batch = core.build_lag_batch_from_loaded(
            Y_data, elec_num, X_all, lags, lag_ms, args.train_ratio
        )

        embed_train = batch.X_train.detach().cpu().numpy()
        embed_test = batch.X_test.detach().cpu().numpy()

        original_feats = extract_original_features(batch)
        original_results: Dict[int, Tuple[float, float]] = {}
        
        if cfg.debug:
            print("--- Debugging Original Features Regression ---")
            
        for sid, (feat_train, feat_test) in original_feats.items():
            acc, mse = regress_embeddings(feat_train, feat_test, embed_train, embed_test, cfg)
            original_results[sid] = (acc, mse)
        mean_corr, mean_mse = summarize_metrics(original_results)
        print(f"  Original mean corr={mean_corr:.4f}  mse={mean_mse:.6f}")

        if args.use_contrastive:
            print("  Training SRM-VAE (Contrastive Mode)...")
            vae_model = train_contrastive_model(
                batch,
                epochs=args.vae_epochs,
                lr=args.vae_lr,
                temp=0.1,
                batch_size=256,
                verbose=True
            )
        else:
            print("  Training SRM-VAE (Reconstruction Mode)...")
            vae_model = train_vae_model(
                batch,
                epochs=args.vae_epochs,
                lr=args.vae_lr,
                beta=args.vae_beta,
                alpha=args.vae_self_weight,
                gamma=args.vae_cross_weight,
            )
        
        # Restore subject views to CPU immediately after training
        for sid in batch.subjects:
            view = batch.subject_views[sid]
            view.train = view.train.detach().cpu()
            view.test = view.test.detach().cpu()

        vae_shared_train, vae_shared_test, vae_recon_feats = extract_vae_features(batch, vae_model)
        
        if cfg.debug:
            print("--- Debugging VAE Shared Regression ---")
        vae_shared_corr, vae_shared_mse = regress_embeddings(
            vae_shared_train, vae_shared_test, embed_train, embed_test, cfg
        )
        print(f"  VAE shared corr={vae_shared_corr:.4f}  mse={vae_shared_mse:.6f}")

        vae_recon_results: Dict[int, Tuple[float, float]] = {}
        
        if cfg.debug:
            print("--- Debugging VAE Reconstruction Regression ---")
            
        for sid, (feat_train, feat_test) in vae_recon_feats.items():
            corr, mse = regress_embeddings(feat_train, feat_test, embed_train, embed_test, cfg)
            vae_recon_results[sid] = (corr, mse)
        vae_recon_corr, vae_recon_mse = summarize_metrics(vae_recon_results)
        print(f"  VAE recon mean corr={vae_recon_corr:.4f}  mse={vae_recon_mse:.6f}")

        print("  Fitting classic SRM...")
        srm_model, srm_shared_train, srm_shared_test_list = fit_classic_srm(
            batch, features=args.srm_k, n_iter=args.srm_iters
        )
        srm_shared_test, srm_recon_feats = extract_srm_reconstructions(
            batch, srm_model, srm_shared_train, srm_shared_test_list
        )

        if cfg.debug:
            print("--- Debugging SRM Shared Regression ---")
            
        srm_shared_corr, srm_shared_mse = regress_embeddings(
            srm_shared_train, srm_shared_test, embed_train, embed_test, cfg
        )
        print(f"  SRM shared corr={srm_shared_corr:.4f}  mse={srm_shared_mse:.6f}")

        srm_recon_results: Dict[int, Tuple[float, float]] = {}
        
        if cfg.debug:
            print("--- Debugging SRM Reconstruction Regression ---")
        
        for sid, (feat_train, feat_test) in srm_recon_feats.items():
            corr, mse = regress_embeddings(feat_train, feat_test, embed_train, embed_test, cfg)
            srm_recon_results[sid] = (corr, mse)
        srm_recon_corr, srm_recon_mse = summarize_metrics(srm_recon_results)
        print(f"  SRM recon mean corr={srm_recon_corr:.4f}  mse={srm_recon_mse:.6f}")

        all_results[int(lag_ms)] = {
            "original": original_results,
            "vae_shared": {"corr": vae_shared_corr, "mse": vae_shared_mse},
            "vae_recon": vae_recon_results,
            "srm_shared": {"corr": srm_shared_corr, "mse": srm_shared_mse},
            "srm_recon": srm_recon_results,
        }

    if args.save_json:
        out_path = os.path.join(
            args.results_dir,
            f"embedding_regression_seed{args.seed}.json",
        )
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump({"config": vars(args), "results": all_results}, handle, indent=2)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
