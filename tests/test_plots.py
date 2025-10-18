# tests/test_plots.py
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import jellyjoin.plots as plots


@pytest.fixture(params=[False, True], ids=["ax=None", "ax=provided"])
def pre_ax(request):
    if request.param:
        fig, ax = plt.subplots()
        try:
            yield ax
        finally:
            plt.close(fig)
    else:
        yield None


def test_plot_similarity_matrix(tmp_path, pre_ax):
    # test data
    sim = np.array([[0.1, 0.9, 0.3], [0.7, 0.2, 0.8]])
    left_labels = ["L0", "L1"]
    right_labels = ["R0", "R1", "R2"]

    fig, ax = plots.plot_similarity_matrix(
        sim,
        ax=pre_ax,
        left_labels=left_labels,
        right_labels=right_labels,
        annotate=True,
        show_colorbar=True,
        title="Similarity Matrix",
    )

    assert fig is not None
    assert ax is not None

    if pre_ax:
        assert ax is pre_ax

    assert len(ax.images) == 1
    assert ax.get_title() == "Similarity Matrix"

    # annotations should match n_rows * n_cols
    n_rows, n_cols = sim.shape
    texts = [t for t in ax.texts if t.get_text()]
    assert len(texts) == n_rows * n_cols

    out = tmp_path / "similarity_matrix.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0

    plt.close(fig)


def test_plot_associations(tmp_path, pre_ax):
    df = pd.DataFrame(
        {
            "Left Value": ["apple", "banana", "cherry"],
            "Right Value": ["red", "yellow", "red"],
            "Left": [0, 1, 2],
            "Right": [0, 1, 2],
        }
    )

    fig, ax = plots.plot_associations(
        df,
        ax=pre_ax,
        indent=0.2,
        text_gap=0.02,
        title="Associations",
    )

    assert fig is not None
    assert ax is not None

    if pre_ax:
        assert ax is pre_ax

    assert ax.get_title() == "Associations"

    # there should be as many connecting lines as rows
    line_count = sum(1 for line in ax.lines if len(line.get_xdata()) == 2)
    assert line_count == len(df)

    out = tmp_path / "associations.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0

    plt.close(fig)


def test_plot_similarity_matrix_invalid_ndim():
    with pytest.raises(ValueError, match=r"similarity_matrix must be 2D\."):
        plots.plot_similarity_matrix(np.array([0.1, 0.2]))

    with pytest.raises(ValueError, match=r"similarity_matrix must be 2D\."):
        plots.plot_similarity_matrix(np.array([[[0.1, 0.2]]]))


def test_plot_associations_missing_columns():
    df = pd.DataFrame(
        {
            "Left Value": ["x"],
            "Right Value": ["y"],
            "Left": [0],
            "Right": [0],
        }
    )

    for column in df.columns:
        bad_df = df.drop(columns=[column])
        with pytest.raises(
            ValueError, match=r"association_df is missing required columns:"
        ):
            plots.plot_associations(bad_df)
