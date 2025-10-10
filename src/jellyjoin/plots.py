import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "plot_similarity_matrix",
    "plot_associations",
]


def plot_similarity_matrix(
    similarity_matrix, figsize=(6, 6), left_labels=None, right_labels=None
):
    """
    Displays a similarity matrix as a heatmap with labels.
    """
    # Display the similarity matrix visually
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(similarity_matrix, cmap="Blues")

    # Show all ticks and label them with the respective list entries
    n_rows, n_columns = similarity_matrix.shape
    ax.set_xticks(np.arange(n_rows))
    ax.set_yticks(np.arange(n_columns))
    if right_labels:
        ax.set_xticklabels(right_labels, rotation=45, ha="right", fontsize=9)
    if left_labels:
        ax.set_yticklabels(left_labels, fontsize=9)

    # Loop over data dimensions and create text annotations
    score_threshold = np.percentile(similarity_matrix, 90)
    for i in range(n_rows):
        for j in range(n_columns):
            score = similarity_matrix[i, j]
            text_color = "black" if score < score_threshold else "white"
            ax.text(
                j,
                i,
                f"{score:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=6,
            )

    # Add colorbar, labels, and title
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Right")
    ax.set_ylabel("Left")
    ax.set_title("Similarity Matrix")
    plt.tight_layout()

    return fig


def plot_associations(
    association_df: pd.DataFrame,
    figsize=(10, 6),
    indent=0.2,
    text_gap=0.02,
    left_column="Left Value",
    right_column="Right Value",
):
    """
    Displays a "connect-the-dots" style plot showing labeled dots
    on the left and right connected by lines.
    """
    # Extract left and right labels and indices
    left_labels = association_df[left_column]
    right_labels = association_df[right_column]
    left_indices = association_df["Left"]
    right_indices = association_df["Right"]

    # Set up plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.axis("off")  # No axes or labels

    # Plot left dots and labels
    for i, (label, idx) in enumerate(zip(left_labels, left_indices)):
        ax.plot(indent, len(left_labels) - idx, "o", color="black")
        ax.text(
            indent - text_gap,
            len(left_labels) - idx,
            label,
            ha="right",
            va="center",
            fontsize=10,
        )

    # Plot right dots and labels
    for i, (label, idx) in enumerate(zip(right_labels, right_indices)):
        ax.plot(1 - indent, len(right_labels) - idx, "o", color="black")
        ax.text(
            1 - indent + text_gap,
            len(right_labels) - idx,
            label,
            ha="left",
            va="center",
            fontsize=10,
        )

    # Draw lines connecting matches
    for left_idx, right_idx in zip(left_indices, right_indices):
        ax.plot(
            [indent, 1 - indent],
            [len(left_labels) - left_idx, len(right_labels) - right_idx],
            color="gray",
            lw=0.8,
            zorder=0,
        )

    return fig
