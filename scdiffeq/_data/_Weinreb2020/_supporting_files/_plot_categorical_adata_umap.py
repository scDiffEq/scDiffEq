
import matplotlib.pyplot as plt
import numpy as np
import vinplots

# need to finish from k-means


def _plot_categorical_adata_umap(adata, groupby, save_as=False, dpi=200):

    """"""

    grouped = adata.obs.groupby(groupby)
    color_palette = vinplots.palettes.SHAREseq()
    
    color_palette = np.repeat(
        color_palette, np.ceil(grouped.ngroups / len(color_palette)).astype(int)
    )
    
    fig = vinplots.Plot()
    fig.construct(nplots=1, ncols=1, figsize=1.2)
    fig.modify_spines(ax="all", spines_to_delete=["top", "right", "left", "bottom"])
    
    ax = fig.AxesDict[0][0]
    ax.set_xticks([])
    ax.set_yticks([])
    fig.fig.dpi = dpi

    for n, (group, group_df) in enumerate(grouped):
        idx = group_df.index.astype(int)
        ax.scatter(
            adata.obsm["X_umap"][idx, 0],
            adata.obsm["X_umap"][idx, 1],
            s=1,
            c=palette[n],
        )
    if save_as:
        plt.savefig(save_as)