
import matplotlib.pyplot as plt
import os
import vinplots


def _Weinreb2020_color_palette():

    palette = {
        "Neutrophil": "#023047",
        "Eos": "#005f73",
        "Baso": "#0a9396",
        "Mast": "#94d2bd",
        "Erythroid": "#e9d8a6",
        "Lymphoid": "#ee9b00",
        "Monocyte": "#F08700",
        "pDC": "#bb3e03",
        "Ccr7_DC": "#ae2012",
        "Meg": "#9b2226",
        "undiff": "#f0efeb",
    }

    return palette

def _build_umap_plot(figsize=1.2):

    """"""

    fig = vinplots.Plot()
    fig.construct(nplots=1, ncols=1, figsize=figsize)
    fig.modify_spines(ax="all", spines_to_delete=["top", "right", "bottom", "left"])
    ax = fig.AxesDict[0][0]
    xt = ax.set_xticks([])
    yt = ax.set_yticks([])

    xl = ax.set_xlabel("UMAP-1", fontsize=14)
    yl = ax.set_ylabel("UMAP-2", fontsize=14)

    return fig, ax

def _plot_dataset(adata, figsize=1.5, save=False):

    """"""
    fig, ax = _build_umap_plot(figsize)

    X_umap = adata.obsm["X_umap"]
    colors = _Weinreb2020_color_palette()

    for n, celltype in enumerate(list(colors.keys())):
        celltype_color = colors[celltype]
        idx = adata.obs.loc[adata.obs["Annotation"] == celltype].index.astype(int)
        ax.scatter(
            X_umap[idx, 0],
            X_umap[idx, 1],
            s=2,
            label=celltype,
            c=celltype_color,
            zorder=100 - n,
        )

    plt.legend(markerscale=3, edgecolor="white", loc=[1, 0.3])
    if save:
        savename = os.path.join(save, "Weinreb2020.png")
        plt.savefig(savename, dpi=500)