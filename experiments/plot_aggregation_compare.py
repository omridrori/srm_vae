"""
plot_aggregation_compare.py

Plot lag vs correlation for the three aggregation methods produced by
experiments/aggregation_compare.py.

Usage:
  python experiments/plot_aggregation_compare.py \
    --input_csv ./results/aggregation_compare/aggregation_compare.csv \
    --output ./results/aggregation_compare/aggregation_compare_plot.png \
    --ylim 0 0.4 --show
"""

import os
import argparse
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_csv(path: str):
    import csv
    x_vals: List[int] = []
    mean_vals: List[float] = []
    prec_vals: List[float] = []
    minv_vals: List[float] = []
    mean_sem: List[float] = []
    prec_sem: List[float] = []
    minv_sem: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                x_vals.append(int(row["requested_lag_ms"]))
                # Support both single-run and aggregated columns
                if "mean_mean" in row:
                    mean_vals.append(float(row.get("mean_mean", np.nan)))
                    prec_vals.append(float(row.get("precision_mean", np.nan)))
                    minv_vals.append(float(row.get("minvar_mean", np.nan)))
                    mean_sem.append(float(row.get("mean_sem", np.nan)))
                    prec_sem.append(float(row.get("precision_sem", np.nan)))
                    minv_sem.append(float(row.get("minvar_sem", np.nan)))
                else:
                    mean_vals.append(float(row.get("mean", np.nan)))
                    prec_vals.append(float(row.get("precision", np.nan)))
                    minv_vals.append(float(row.get("minvar", np.nan)))
                    mean_sem.append(float("nan"))
                    prec_sem.append(float("nan"))
                    minv_sem.append(float("nan"))
            except Exception:
                continue
    order = np.argsort(x_vals)
    x_vals = [x_vals[i] for i in order]
    mean_vals = [mean_vals[i] for i in order]
    prec_vals = [prec_vals[i] for i in order]
    minv_vals = [minv_vals[i] for i in order]
    mean_sem = [mean_sem[i] for i in order]
    prec_sem = [prec_sem[i] for i in order]
    minv_sem = [minv_sem[i] for i in order]
    return x_vals, mean_vals, prec_vals, minv_vals, mean_sem, prec_sem, minv_sem


def plot_curves(x, mean_y, prec_y, minv_y, mean_sem, prec_sem, minv_sem,
                title: str, ylim: Tuple[float, float] = None, figsize: Tuple[float, float] = (6.0, 4.0)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, mean_y, marker='o', linewidth=1.5, label='mean', color='royalblue')
    ax.plot(x, prec_y, marker='o', linewidth=1.5, label='precision', color='darkorange')
    ax.plot(x, minv_y, marker='o', linewidth=1.5, label='minvar', color='seagreen')
    if not all(np.isnan(mean_sem)):
        arr = np.asarray(mean_y)
        sem = np.asarray(mean_sem)
        ax.fill_between(x, arr - sem, arr + sem, color='royalblue', alpha=0.2)
    if not all(np.isnan(prec_sem)):
        arr = np.asarray(prec_y)
        sem = np.asarray(prec_sem)
        ax.fill_between(x, arr - sem, arr + sem, color='darkorange', alpha=0.2)
    if not all(np.isnan(minv_sem)):
        arr = np.asarray(minv_y)
        sem = np.asarray(minv_sem)
        ax.fill_between(x, arr - sem, arr + sem, color='seagreen', alpha=0.2)
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
    parser.add_argument('--input_csv', type=str, required=True, help='Path to aggregation_compare.csv')
    parser.add_argument('--output', type=str, default=None, help='Path to save the plot PNG')
    parser.add_argument('--title', type=str, default='Aggregation Methods: Lag vs Correlation')
    parser.add_argument('--ylim', type=float, nargs=2, default=None, help='y-axis limits, e.g., --ylim 0 0.4')
    parser.add_argument('--figsize', type=float, nargs=2, default=(6.0, 4.0), help='Figure size, e.g., --figsize 6 4')
    parser.add_argument('--show', action='store_true', help='Show plot after saving')
    args = parser.parse_args()

    data = read_csv(args.input_csv)
    x, mean_y, prec_y, minv_y, mean_sem, prec_sem, minv_sem = data
    if not x:
        print(f"[ERROR] No data found in: {args.input_csv}")
        return

    fig = plot_curves(x, mean_y, prec_y, minv_y, mean_sem, prec_sem, minv_sem,
                      title=args.title, ylim=tuple(args.ylim) if args.ylim else None, figsize=tuple(args.figsize))
    out_path = args.output or os.path.join(os.path.dirname(args.input_csv), 'aggregation_compare_plot.png')
    fig.savefig(out_path, dpi=150)
    print(f"[PLOT] Saved to {out_path}")
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()


