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
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_json_experiment(path: str) -> Tuple[str, List[int], List[float]]:
    """Return (label, x_lags_ms, y_means). Uses requested lags from file."""
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    label = os.path.splitext(os.path.basename(path))[0]
    # Prefer explicit beta label if present
    if isinstance(state, dict) and "beta" in state:
        label = f"beta={state['beta']}"

    x_req = state.get("x_requested", [])
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

    return label, x_vals, y_vals


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


def collect_experiments(input_dir: str, prefer: str = 'json') -> List[Tuple[str, List[int], List[float]]]:
    """
    Collect experiments ensuring at most one curve per base filename.
    If both JSON and CSV exist for the same base, prefer according to `prefer`.
    """
    assert prefer in {'json', 'csv'}
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    # Group by base name (without extension)
    by_base: Dict[str, Dict[str, str]] = {}
    for fname in files:
        base, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext not in {'.json', '.csv'}:
            continue
        by_base.setdefault(base, {})[ext] = os.path.join(input_dir, fname)

    exps: List[Tuple[str, List[int], List[float]]] = []
    for base in sorted(by_base.keys()):
        paths = by_base[base]
        chosen_path = None
        if prefer == 'json' and '.json' in paths:
            chosen_path = paths['.json']
        elif prefer == 'csv' and '.csv' in paths:
            chosen_path = paths['.csv']
        else:
            # Fallback to whichever exists
            chosen_path = paths.get('.json') or paths.get('.csv')

        if not chosen_path:
            continue

        try:
            if chosen_path.lower().endswith('.json'):
                exps.append(read_json_experiment(chosen_path))
            else:
                exps.append(read_csv_experiment(chosen_path))
        except Exception as e:
            print(f"[WARN] Failed to read {chosen_path}: {e}")
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


def plot_experiments(experiments: List[Tuple[str, List[int], List[float]]],
                     title: str = "Lag vs Correlation (mean r)",
                     ylim: Tuple[float, float] = None,
                     figsize: Tuple[float, float] = (6.0, 4.0)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    # Filter out empty experiments while preserving order
    non_empty = [(label, x_vals, y_vals) for (label, x_vals, y_vals) in experiments if x_vals]
    for label, x_vals, _ in experiments:
        if not x_vals:
            print(f"[SKIP] {label} has no data")
    colors = get_distinct_colors(len(non_empty))
    for (label, x_vals, y_vals), color in zip(non_empty, colors):
        ax.plot(x_vals, y_vals, marker='o', linewidth=1.5, label=label, color=color)
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


