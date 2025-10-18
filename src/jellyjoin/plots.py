# jellyjoin/plots.py
from __future__ import annotations

from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "plot_similarity_matrix",
    "plot_associations",
]


def plot_similarity_matrix(
    similarity_matrix: np.ndarray,
    *,
    ax: Axes | None = None,
    figsize: Tuple[float, float] | None = None,
    left_labels: Iterable[str] | None = None,
    right_labels: Iterable[str] | None = None,
    cmap: str = "Blues",
    annotate: bool = True,
    annotation_fontsize: int = 6,
    label_fontsize: int = 9,
    title: str = "Similarity Matrix",
    show_colorbar: bool = True,
) -> Tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sim = np.asarray(similarity_matrix)
    if sim.ndim != 2:
        raise ValueError("similarity_matrix must be 2D.")

    n_rows, n_cols = sim.shape
    im = ax.imshow(sim, cmap=cmap)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))

    if right_labels is not None:
        ax.set_xticklabels(
            list(right_labels), rotation=45, ha="right", fontsize=label_fontsize
        )
    if left_labels is not None:
        ax.set_yticklabels(list(left_labels), fontsize=label_fontsize)

    if annotate:
        thresh = float(np.percentile(sim, 90)) if sim.size else 0.0
        for i in range(n_rows):
            for j in range(n_cols):
                val = sim[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=("black" if val < thresh else "white"),
                    fontsize=annotation_fontsize,
                )

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Right")
    ax.set_ylabel("Left")
    ax.set_title(title)
    fig.tight_layout()

    return fig, ax


def plot_associations(
    association_df: pd.DataFrame,
    *,
    ax: Axes | None = None,
    figsize: Tuple[float, float] | None = None,
    indent: float = 0.2,
    text_gap: float = 0.02,
    left_column: str = "Left Value",
    right_column: str = "Right Value",
    left_index_column: str = "Left",
    right_index_column: str = "Right",
    marker_color: str = "black",
    line_color: str = "gray",
    line_width: float = 0.8,
    label_fontsize: int = 10,
    title: str | None = None,
) -> Tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    required = {left_column, right_column, left_index_column, right_index_column}
    missing = required - set(association_df.columns)
    if missing:
        raise ValueError(
            f"association_df is missing required columns: {sorted(missing)}"
        )

    left_labels = association_df[left_column].tolist()
    right_labels = association_df[right_column].tolist()
    left_indices = association_df[left_index_column].tolist()
    right_indices = association_df[right_index_column].tolist()

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(len(left_labels), len(right_labels)) + 1)
    ax.axis("off")

    for label, idx in zip(left_labels, left_indices):
        y = len(left_labels) - idx
        ax.plot(indent, y, "o", color=marker_color)
        ax.text(
            indent - text_gap,
            y,
            str(label),
            ha="right",
            va="center",
            fontsize=label_fontsize,
        )

    for label, idx in zip(right_labels, right_indices):
        y = len(right_labels) - idx
        ax.plot(1 - indent, y, "o", color=marker_color)
        ax.text(
            1 - indent + text_gap,
            y,
            str(label),
            ha="left",
            va="center",
            fontsize=label_fontsize,
        )

    for li, ri in zip(left_indices, right_indices):
        y_left = len(left_labels) - li
        y_right = len(right_labels) - ri
        ax.plot(
            [indent, 1 - indent],
            [y_left, y_right],
            color=line_color,
            lw=line_width,
            zorder=0,
        )

    if title:
        ax.set_title(title)
    fig.tight_layout()

    return fig, ax
