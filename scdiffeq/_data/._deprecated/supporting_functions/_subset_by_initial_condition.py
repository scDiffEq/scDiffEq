import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from ...plotting._figure_presets import subplot_presets


def get_index_all_y0(adata):

    """"""

    y0_index = adata.obs.loc[adata.obs.timepoint == 0.0].index.astype(int)

    return y0_index


def subset_y0s(y0_points, x_coord, y_coord, x_cut, y_cut):

    """"""

    if x_cut == "<":

        if y_cut == ">":

            subset_idx = np.intersect1d(
                np.where(y0_points[:, 0] < x_coord), np.where(y0_points[:, 1] > y_coord)
            )

        else:
            subset_idx = np.intersect1d(
                np.where(y0_points[:, 0] < x_coord), np.where(y0_points[:, 1] < y_coord)
            )

    if x_cut == ">":

        if y_cut == ">":

            subset_idx = np.intersect1d(
                np.where(y0_points[:, 0] > x_coord), np.where(y0_points[:, 1] > y_coord)
            )

        else:
            subset_idx = np.intersect1d(
                np.where(y0_points[:, 0] > x_coord), np.where(y0_points[:, 1] < y_coord)
            )

    y0_subset = y0_points[subset_idx]

    return y0_subset, subset_idx


def select_y0(adata, x_coord, y_coord, x_cut="<", y_cut=">"):

    """"""

    y0_index = get_index_all_y0(adata)
    y0_points = adata.X[y0_index]
    y0_subset, idx = subset_y0s(y0_points, x_coord, y_coord, x_cut, y_cut)

    return idx, y0_points, y0_subset


def subset_adata_by_initial_condition(adata, x_coord, y_coord, x_cut="<", y_cut=">"):

    subset_initial_conditions_idx, y0_points, y0_subset = select_y0(
        adata, -0.5, 0.45, x_cut="<", y_cut=">"
    )

    adata = adata[
        adata.obs.loc[
            adata.obs.trajectory.isin(subset_initial_conditions_idx)
        ].index.astype(int)
    ]

    print(adata)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor="white")

    fig.suptitle("Data subset", y=1.02, fontsize=20)  # , fontweight="semibold")

    ax1, ax2 = axes[0], axes[1]
    subplot_presets(
        ax1,
        y0_points[:, 0],
        y0_points[:, 1],
        xlab=None,
        ylab=None,
        size=15,
        color="lightgrey",
        alpha=0.2,
    )
    subplot_presets(
        ax1,
        y0_subset[:, 0],
        y0_subset[:, 1],
        xlab="$X$",
        ylab="$Y$",
        size=15,
        color="orange",
        alpha=1.0,
    )
    subplot_presets(
        ax2,
        adata.X[:, 0],
        adata.X[:, 1],
        xlab="$X$",
        ylab="$Y$",
        size=15,
        color=adata.obs.timepoint,
        alpha=0.5,
    )
    fig.tight_layout(pad=1.5)

    adata.obs = adata.obs.reset_index()

    return adata
