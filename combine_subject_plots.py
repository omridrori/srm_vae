#!/usr/bin/env python
"""Combine per-subject VAE-vs-SRM plots into a single panel.

Run this script from the project root (or adjust IMG_DIR if needed).
It searches for PNG files named ``subject_<id>_vae_vs_srm.png`` inside
``shared_space_plots/`` and produces ``all_subjects_combined.png`` in the
same folder.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Iterable, Tuple


def combine_subject_plots(
    img_dir: str = "shared_space_plots",
    subject_ids: Iterable[int] = range(1, 9),
    output_name: str = "all_subjects_combined.png",
    grid_shape: Tuple[int, int] = (2, 4),
    figsize: Tuple[int, int] = (16, 8),
    dpi: int = 300,
) -> str:
    """Load each subject image and arrange them into a single figure.

    Parameters
    ----------
    img_dir : str
        Directory containing the individual subject PNG files.
    subject_ids : Iterable[int]
        Iterable of subject indices (1-based) to include.
    output_name : str
        Filename for the combined figure (saved inside ``img_dir``).
    grid_shape : Tuple[int, int]
        (rows, cols) for the subplot grid.
    figsize : Tuple[int, int]
        Size of the matplotlib figure in inches.
    dpi : int
        Resolution of the saved image.

    Returns
    -------
    str
        Full path to the saved combined figure.
    """

    n_rows, n_cols = grid_shape
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for idx, sid in enumerate(subject_ids):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        img_path = os.path.join(img_dir, f"subject_{sid}_vae_vs_srm.png")
        if not os.path.exists(img_path):
            ax.set_visible(False)
            print(f"[WARN] {img_path} not found – skipping.")
            continue

        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Subject {sid}")

    # Hide any unused axes (in case fewer than rows*cols images)
    total_plots = len(subject_ids)
    for extra_idx in range(total_plots, n_rows * n_cols):
        row, col = divmod(extra_idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()

    out_path = os.path.join(img_dir, output_name)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Combined figure saved to {out_path}")

    return out_path


if __name__ == "__main__":
    combine_subject_plots()