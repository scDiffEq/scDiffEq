
import numpy as np
import vinplots
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

def _build_layout(figsize=vinplots.build.default_plt_dims() * 3):

    fig = plt.Figure(figsize)
    gs = GridSpec(3, 3, hspace=0.5, wspace=0.4, width_ratios=[1.5, 1.5, 1])
    ax = fig.add_subplot(gs[:3, :2])

    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[1, 2])
    ax3 = fig.add_subplot(gs[2, 2])

    for _ax in [ax, ax1, ax2, ax3]:
        vinplots.style.modify_spines(
            ax=_ax,
            spines_positioning_amount=10,
            spines_to_move=["bottom", "left"],
            spines_to_delete=["top", "right"],
        )
        _ax.grid(True, c="lightgrey", alpha=0.25)

    return fig, [ax, ax1, ax2, ax3]

def _get_true_lineage(adata, adata_clonal, clone_idx):

    """"""
    
    idx_ = np.where(adata[clone_idx].obsm["X_clone"].toarray() == True)
    idx = np.array(idx_).flatten()[np.array(idx_).flatten() != 0]
    lineage_idx = adata_clonal[adata_clonal.obsm["X_clone"][:, idx].toarray()].obs.index.astype(
        int
    )
    true_lineage = adata_clonal.obsm["X_umap"][lineage_idx]

    return true_lineage, lineage_idx

def _plot_predicted(adata, adata_clonal, umap, prediction, idx):

    """"""

    X_umap = adata.obsm["X_umap"]

    fig, axes = _build_layout()
    axes[0].scatter(X_umap[:, 0], X_umap[:, 1], c="lightgrey", s=5, zorder=2)
    colors = ["#FFD500", "darkorange", "red"]

    titles = ["Day 2", "Day 4", "Day 6"]

    for timepoint in range(3):
        day = umap.transform(prediction[timepoint].detach().numpy()).reshape(
            [prediction.shape[1], 2]
        )
        axes[0].scatter(day[:, 0], day[:, 1], c=colors[timepoint], zorder=3, s=15)
        axes[0].set_title("Simulated Predictions", fontsize=16)
        axes[0].set_xlabel("UMAP-1", fontsize=14)
        axes[0].set_ylabel("UMAP-2", fontsize=14)
        axes[timepoint + 1].scatter(
            X_umap[:, 0], X_umap[:, 1], c="lightgrey", s=3, zorder=2
        )
        axes[timepoint + 1].scatter(
            day[:, 0], day[:, 1], c=colors[timepoint], s=10, zorder=2
        )
        axes[timepoint + 1].set_title(titles[timepoint], fontsize=16)
        axes[timepoint + 1].set_xlabel("UMAP-1", fontsize=14)
        axes[timepoint + 1].set_ylabel("UMAP-2", fontsize=14)

#     plot the true lineage
    if idx:
        true_lin_coords, lineage_idx = _get_true_lineage(adata, adata_clonal, idx)

        df = adata_clonal[lineage_idx].obs.rename({"index": "set_index"}, axis=1)
        c = ["#8ECAE6", "#219EBC", "#023047"]
        DayCoords = {}

        for n, i in enumerate([2, 4, 6]):
            idx_ = df.loc[df["Time point"] == i].index.astype(int)
            DayCoords[i] = v = adata_clonal.obsm["X_umap"][idx_]
            axes[0].scatter(v[:, 0], v[:, 1], c=c[n], s=45, zorder=10, label=i)
        axes[0].legend(edgecolor='w')

    return fig