"""
plot_experiment_curves.py

Given a directory of experiment results (JSON and/or CSV produced by the
beta ablation script), generate a single plot with one curve per experiment,
showing lag (ms) vs correlation (mean r). Each experiment uses a different color.

Usage:
  python experiments/plot_experiment_curves.py --input_dir ./results/beta_ablation \
      --output ./results/beta_ablation/combined_lag_correlation.png

Options:
  --show  Display the figure after saving.
"""

import os
import json
import argparse
import re
from typing import Dict, List, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt


def read_json_experiment(path: str) -> Tuple[str, List[int], List[float], List[float]]:
    """Return (label, x_lags_ms, y_mean, y_sem). Supports single-run or multi-run format."""
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    label = os.path.splitext(os.path.basename(path))[0]
    # Prefer explicit labels if present
    if isinstance(state, dict):
        if "ortho_weight" in state:
            label = f"ortho={state['ortho_weight']}"
        elif "beta" in state:
            label = f"beta={state['beta']}"

    x_req = state.get("x_requested", [])
    runs = state.get("runs")
    if runs:
        # Aggregate across runs → mean and SEM per lag
        x_vals: List[int] = []
        y_mean: List[float] = []
        y_sem: List[float] = []
        for lag in x_req:
            key = str(lag)
            vals: List[float] = []
            for rk, rstate in runs.items():
                lr = rstate.get("lag_results", {})
                if key in lr:
                    vals.append(float(lr[key]["r_mean"]))
            if vals:
                x_vals.append(int(lag))
                arr = np.asarray(vals, dtype=float)
                y_mean.append(float(np.nanmean(arr)))
                if arr.size > 1:
                    y_sem.append(float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size)))
                else:
                    y_sem.append(float('nan'))
        return label, x_vals, y_mean, y_sem
    else:
        # Single-run
        lag_results = state.get("lag_results", {})
        x_vals: List[int] = []
        y_vals: List[float] = []
        for lag in x_req:
            key = str(lag)
            if key in lag_results:
                x_vals.append(int(lag))
                y_vals.append(float(lag_results[key]["r_mean"]))
        # Fallback: if x_requested is missing, derive from keys
        if not x_vals and isinstance(lag_results, dict):
            items = []
            for k, v in lag_results.items():
                try:
                    items.append((int(k), float(v.get("r_mean", np.nan))))
                except Exception:
                    continue
            items.sort(key=lambda t: t[0])
            if items:
                x_vals = [t[0] for t in items]
                y_vals = [t[1] for t in items]
        return label, x_vals, y_vals, [float('nan')] * len(x_vals)


def read_csv_experiment(path: str) -> Tuple[str, List[int], List[float]]:
    import csv
    label = os.path.splitext(os.path.basename(path))[0]
    x_vals: List[int] = []
    y_vals: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                x_vals.append(int(row["requested_lag_ms"]))
                y_vals.append(float(row["r_mean"]))
            except Exception:
                continue
    # Ensure sorted by lag
    order = np.argsort(x_vals)
    x_vals = [x_vals[i] for i in order]
    y_vals = [y_vals[i] for i in order]
    return label, x_vals, y_vals


def read_csv_mean_experiment(path: str) -> Tuple[str, List[int], List[float], List[float]]:
    """Read aggregated mean±SEM CSV: requested_lag_ms, r_mean_mean, r_mean_sem"""
    import csv
    label = os.path.splitext(os.path.basename(path))[0]
    x_vals: List[int] = []
    y_mean: List[float] = []
    y_sem: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                x_vals.append(int(row["requested_lag_ms"]))
                y_mean.append(float(row["r_mean_mean"]))
                y_sem.append(float(row["r_mean_sem"]))
            except Exception:
                continue
    order = np.argsort(x_vals)
    x_vals = [x_vals[i] for i in order]
    y_mean = [y_mean[i] for i in order]
    y_sem = [y_sem[i] for i in order]
    return label, x_vals, y_mean, y_sem


def aggregate_run_csvs(paths: Iterable[str], label: str) -> Tuple[str, List[int], List[float], List[float]]:
    """Aggregate multiple per-run CSVs (requested_lag_ms, r_mean) into mean±SEM per lag."""
    import csv
    lag_to_values: Dict[int, List[float]] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    lag = int(row["requested_lag_ms"])
                    val = float(row["r_mean"])
                except Exception:
                    continue
                lag_to_values.setdefault(lag, []).append(val)
    x_vals = sorted(lag_to_values.keys())
    y_mean: List[float] = []
    y_sem: List[float] = []
    for lag in x_vals:
        vals = np.asarray(lag_to_values[lag], dtype=float)
        y_mean.append(float(np.nanmean(vals)))
        if vals.size > 1:
            y_sem.append(float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)))
        else:
            y_sem.append(float('nan'))
    return label, x_vals, y_mean, y_sem


def collect_experiments(input_dir: str, prefer: str = 'json') -> List[Tuple]:
    """
    Collect experiments as one aggregated curve per experiment.
    Group by canonical base name (strip trailing _run\d+). Priority:
      1) JSON (multi-run aware)
      2) mean CSV (beta_X.csv with mean±SEM)
      3) aggregate per-run CSVs (beta_X_run*.csv)
    """
    assert prefer in {'json', 'csv'}
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    def canonical_base(fname: str) -> str:
        base, _ = os.path.splitext(fname)
        return re.sub(r"_run\d+$", "", base)

    groups: Dict[str, Dict[str, object]] = {}
    for fname in files:
        path = os.path.join(input_dir, fname)
        base, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext not in {'.json', '.csv'}:
            continue
        canon = canonical_base(fname)
        g = groups.setdefault(canon, {"json": None, "mean_csv": None, "run_csvs": []})
        if ext == '.json':
            g["json"] = path
        else:
            # CSV: detect if it's a run CSV
            if re.search(r"_run\d+$", base):
                g["run_csvs"].append(path)
            else:
                g["mean_csv"] = path

    exps: List[Tuple] = []
    for canon in sorted(groups.keys()):
        g = groups[canon]
        try:
            if prefer == 'json' and g["json"]:
                exps.append(read_json_experiment(g["json"]))
            elif g["mean_csv"]:
                exps.append(read_csv_mean_experiment(g["mean_csv"]))
            elif g["run_csvs"]:
                exps.append(aggregate_run_csvs(g["run_csvs"], label=os.path.basename(canon)))
            elif g["json"]:
                # fallback if prefer='csv' but only json exists
                exps.append(read_json_experiment(g["json"]))
        except Exception as e:
            print(f"[WARN] Failed to read group {canon}: {e}")
            continue

    return exps


def get_distinct_colors(n: int) -> List[tuple]:
    """Return n visually distinct colors (as RGBA tuples)."""
    colors: List[tuple] = []
    if n <= 0:
        return colors
    # Prefer qualitative palette for up to 20
    if n <= 20:
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(n)]
    else:
        # Evenly spaced hues for many curves
        cmap = plt.get_cmap('hsv')
        colors = [cmap(i / float(n)) for i in range(n)]
    return colors


def plot_experiments(experiments: List[Tuple],
                     title: str = "Lag vs Correlation (mean r)",
                     ylim: Tuple[float, float] = None,
                     figsize: Tuple[float, float] = (6.0, 4.0)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    # Filter out empty experiments while preserving order
    non_empty = [(label, x_vals, y_vals, y_sem) for (label, x_vals, y_vals, y_sem) in experiments if x_vals]
    for label, x_vals, *_ in experiments:
        if not x_vals:
            print(f"[SKIP] {label} has no data")
    colors = get_distinct_colors(len(non_empty))
    for (label, x_vals, y_vals, y_sem), color in zip(non_empty, colors):
        ax.plot(x_vals, y_vals, marker='o', linewidth=1.5, label=label, color=color)
        # SEM shading if available
        if y_sem and not all(np.isnan(y_sem)):
            y_arr = np.asarray(y_vals)
            sem_arr = np.asarray(y_sem)
            ax.fill_between(x_vals, y_arr - sem_arr, y_arr + sem_arr, color=color, alpha=0.2)
    ax.axvline(0, ls='dashed', c='k', alpha=0.3)
    ax.set_xlabel('lags (ms)')
    ax.set_ylabel('Encoding Performance (r)')
    ax.set_title(title)
    ax.legend()
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Folder with experiment result files (JSON/CSV)')
    parser.add_argument('--output', type=str, default=None, help='Path to save the combined plot PNG')
    parser.add_argument('--title', type=str, default='Lag vs Correlation (mean r)')
    parser.add_argument('--ylim', type=float, nargs=2, default=None, help='y-axis limits, e.g., --ylim 0 0.4')
    parser.add_argument('--figsize', type=float, nargs=2, default=(6.0, 4.0), help='Figure size, e.g., --figsize 6 4')
    parser.add_argument('--show', action='store_true', help='Show plot after saving')
    parser.add_argument('--prefer', type=str, choices=['json', 'csv'], default='json', help='Prefer JSON or CSV when both exist for same experiment')
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"[ERROR] Not a directory: {input_dir}")
        return

    experiments = collect_experiments(input_dir, prefer=args.prefer)
    if not experiments:
        print(f"[ERROR] No experiments found in: {input_dir}")
        return

    fig = plot_experiments(experiments, title=args.title, ylim=tuple(args.ylim) if args.ylim else None, figsize=tuple(args.figsize))

    out_path = args.output
    if out_path is None:
        out_path = os.path.join(input_dir, 'combined_lag_correlation.png')
    fig.savefig(out_path, dpi=150)
    print(f"[PLOT] Saved to {out_path}")

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()


