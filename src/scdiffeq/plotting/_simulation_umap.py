# -- import packages: ---------------------------------------------------------
import anndata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# -- set typing: --------------------------------------------------------------
from typing import Dict, List, Optional, Union


# -- API-facing function: -----------------------------------------------------
def simulation_umap(
    adata_sim: anndata.AnnData,
    color: str = "t",
    use_key: str = "X_umap",
    gene_key: str = "X_gene_inv",
    gene_ids_key: str = "gene_ids",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (4, 4),
    cmap: Union[str, mcolors.Colormap] = "viridis",
    categorical_cmap: Optional[Dict[str, str]] = None,
    s: float = 1.0,
    alpha: float = 0.8,
    title: Optional[str] = None,
    show_colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    show_legend: bool = True,
    legend_loc: str = "best",
    x_label: str = "UMAP 1",
    y_label: str = "UMAP 2",
    save: bool = False,
    savename: Optional[str] = None,
    save_format: str = "svg",
    dpi: int = 300,
    **kwargs,
) -> plt.Axes:
    """
    Plot UMAP embedding of simulated data, colored by obs attribute or gene expression.

    Parameters
    ----------
    adata_sim : anndata.AnnData
        Simulated data from ``sdq.tl.simulate()``, with UMAP coordinates
        in obsm and optionally gene expression in obsm after calling
        ``sdq.tl.invert_scaled_gex()``.
    color : str, default="t"
        What to color points by. Can be:
        - Column name in ``adata_sim.obs`` (e.g., "t", "fate", "sim_i")
        - Gene name (will look up in gene_ids_key and extract from gene_key)
    use_key : str, default="X_umap"
        Key in ``adata_sim.obsm`` containing UMAP coordinates.
    gene_key : str, default="X_gene_inv"
        Key in ``adata_sim.obsm`` containing gene expression matrix.
    gene_ids_key : str, default="gene_ids"
        Key in ``adata_sim.uns`` containing gene names.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
    figsize : tuple, default=(4, 4)
        Figure size (width, height) in inches if creating new figure.
    cmap : str or Colormap, default="viridis"
        Colormap for continuous values.
    categorical_cmap : Dict[str, str], optional
        Mapping from category names to colors for categorical data.
    s : float, default=1.0
        Point size.
    alpha : float, default=0.8
        Point transparency.
    title : str, optional
        Plot title. If None, uses the color parameter.
    show_colorbar : bool, default=True
        Whether to show colorbar for continuous values.
    colorbar_label : str, optional
        Label for colorbar. If None, uses the color parameter.
    show_legend : bool, default=True
        Whether to show legend for categorical values.
    legend_loc : str, default="best"
        Legend location.
    x_label : str, default="UMAP 1"
        Label for x-axis.
    y_label : str, default="UMAP 2"
        Label for y-axis.
    save : bool, default=False
        Whether to save the figure.
    savename : str, optional
        Filename for saving. If None, auto-generates from color parameter.
    save_format : str, default="svg"
        Format for saving figure.
    dpi : int, default=300
        Resolution for saving figure.
    **kwargs
        Additional keyword arguments passed to ``ax.scatter()``
        (e.g., ``zorder``, ``edgecolors``, ``linewidths``).

    Returns
    -------
    plt.Axes
        The matplotlib axes object.

    Examples
    --------
    >>> import scdiffeq as sdq
    >>> # Color by time
    >>> sdq.pl.simulation_umap(adata_sim, color="t")
    >>> # Color by fate
    >>> sdq.pl.simulation_umap(adata_sim, color="fate", categorical_cmap={"Mon.": "orange", "Neu.": "blue"})
    >>> # Color by gene expression
    >>> sdq.pl.simulation_umap(adata_sim, color="Myc")
    """
    # -- Get UMAP coordinates -------------------------------------------------
    umap_coords = adata_sim.obsm[use_key]
    if isinstance(umap_coords, pd.DataFrame):
        x = umap_coords.iloc[:, 0].values
        y = umap_coords.iloc[:, 1].values
    else:
        x = umap_coords[:, 0]
        y = umap_coords[:, 1]

    # -- Determine color values -----------------------------------------------
    is_categorical = False
    color_values = None

    # Check if color is in obs
    if color in adata_sim.obs.columns:
        color_values = adata_sim.obs[color].values
        # Check if categorical
        if adata_sim.obs[color].dtype.name == 'category' or isinstance(color_values[0], str):
            is_categorical = True
    else:
        # Try to find as gene name
        gene_ids_raw = adata_sim.uns.get(gene_ids_key, {})

        # Handle dict format: {index: gene_name}
        if isinstance(gene_ids_raw, dict):
            gene_names = list(gene_ids_raw.values())
            if color in gene_names:
                gene_idx = gene_names.index(color)
                expr_matrix = adata_sim.obsm[gene_key]
                if isinstance(expr_matrix, pd.DataFrame):
                    color_values = expr_matrix.iloc[:, gene_idx].values
                else:
                    color_values = expr_matrix[:, gene_idx]
        else:
            # Handle list-like formats
            if isinstance(gene_ids_raw, (pd.Index, pd.Series)):
                gene_ids = gene_ids_raw.tolist()
            elif isinstance(gene_ids_raw, np.ndarray):
                gene_ids = gene_ids_raw.tolist()
            elif isinstance(gene_ids_raw, list):
                gene_ids = gene_ids_raw
            else:
                gene_ids = list(gene_ids_raw) if gene_ids_raw else []

            if color in gene_ids:
                gene_idx = gene_ids.index(color)
                expr_matrix = adata_sim.obsm[gene_key]
                if isinstance(expr_matrix, pd.DataFrame):
                    color_values = expr_matrix.iloc[:, gene_idx].values
                else:
                    color_values = expr_matrix[:, gene_idx]

    if color_values is None:
        raise ValueError(
            f"'{color}' not found in adata_sim.obs columns or gene names. "
            f"Available obs columns: {list(adata_sim.obs.columns)[:5]}..."
        )

    # -- Setup figure ---------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # -- Plot -----------------------------------------------------------------
    if is_categorical:
        unique_cats = np.unique(color_values)

        # Default categorical colors
        if categorical_cmap is None:
            default_colors = plt.cm.tab10.colors
            categorical_cmap = {cat: default_colors[i % len(default_colors)]
                               for i, cat in enumerate(unique_cats)}

        for cat in unique_cats:
            mask = color_values == cat
            cat_color = categorical_cmap.get(cat, "gray")
            ax.scatter(x[mask], y[mask], c=[cat_color], s=s, alpha=alpha, label=cat, **kwargs)

        if show_legend:
            ax.legend(loc=legend_loc, frameon=True, facecolor="white",
                     edgecolor="lightgray", fontsize=8, markerscale=3)
    else:
        # Continuous
        scatter = ax.scatter(x, y, c=color_values, cmap=cmap, s=s, alpha=alpha, **kwargs)

        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            if colorbar_label is None:
                colorbar_label = color
            cbar.set_label(colorbar_label, fontsize=10)

    # -- Formatting -----------------------------------------------------------
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)

    if title is None:
        title = color
    ax.set_title(title, fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    # -- Save -----------------------------------------------------------------
    if save:
        if savename is None:
            savename = f"scDiffEq.simulation_umap.{color}.{save_format}"
        plt.savefig(savename, format=save_format, dpi=dpi, bbox_inches="tight")

    return ax
