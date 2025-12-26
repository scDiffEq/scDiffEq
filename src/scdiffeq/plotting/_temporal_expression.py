# -- import packages: ---------------------------------------------------------
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- set typing: --------------------------------------------------------------
from typing import Dict, List, Optional, Union


# -- API-facing function: -----------------------------------------------------
def temporal_expression(
    adata_sim: anndata.AnnData,
    gene: str,
    groupby: str = "final_state",
    use_key: str = "X_gene_inv",
    time_key: str = "t",
    gene_ids_key: str = "gene_ids",
    show_std: bool = True,
    std_alpha: float = 0.2,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (3, 2.5),
    cmap: Optional[Dict[str, str]] = None,
    linewidth: float = 2.0,
    x_label: str = "t(d)",
    y_label: str = "Log-norm. expression",
    title: Optional[str] = None,
    show_legend: bool = True,
    legend_loc: Union[str, tuple] = "best",
    grid: bool = True,
    grid_alpha: float = 0.3,
    save: bool = False,
    savename: Optional[str] = None,
    save_format: str = "svg",
    dpi: int = 300,
) -> plt.Axes:
    """
    Plot gene expression over simulated time, grouped by fate.

    Computes mean and standard deviation at each time step and plots as
    line (mean) with shaded fill-between region (±1 std).

    Parameters
    ----------
    adata_sim : anndata.AnnData
        Simulated data from ``sdq.tl.simulate()``, with gene expression
        stored in obsm after calling ``sdq.tl.invert_scaled_gex()``.
    gene : str
        Gene name to plot.
    groupby : str, default="final_state"
        Column in ``adata_sim.obs`` for grouping trajectories (e.g., cell fate).
    use_key : str, default="X_gene_inv"
        Key in ``adata_sim.obsm`` containing the gene expression matrix.
    time_key : str, default="t"
        Column in ``adata_sim.obs`` containing time values.
    gene_ids_key : str, default="gene_ids"
        Key in ``adata_sim.uns`` containing gene names array.
    show_std : bool, default=True
        Whether to show standard deviation as shaded fill-between region.
    std_alpha : float, default=0.2
        Transparency of the standard deviation shading.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
    figsize : tuple, default=(3, 2.5)
        Figure size (width, height) in inches if creating new figure.
    cmap : Dict[str, str], optional
        Mapping from group names to colors. If None, uses default colormap.
    linewidth : float, default=2.0
        Width of the mean line.
    x_label : str, default="t(d)"
        Label for x-axis.
    y_label : str, default="Log-norm. expression"
        Label for y-axis.
    title : str, optional
        Plot title. If None, uses gene name in italic.
    show_legend : bool, default=True
        Whether to show legend.
    legend_loc : str or tuple, default="best"
        Legend location.
    grid : bool, default=True
        Whether to show grid.
    grid_alpha : float, default=0.3
        Transparency of grid lines.
    save : bool, default=False
        Whether to save the figure.
    savename : str, optional
        Filename for saving. If None, auto-generates from gene name.
    save_format : str, default="svg"
        Format for saving figure.
    dpi : int, default=300
        Resolution for saving figure.

    Returns
    -------
    plt.Axes
        The matplotlib axes object.

    Examples
    --------
    >>> import scdiffeq as sdq
    >>> adata_sim = sdq.tl.simulate(adata, diffeq=model, idx=idx)
    >>> sdq.tl.invert_scaled_gex(adata_sim, ...)
    >>> sdq.tl.annotate_cell_fate(adata_sim, ...)
    >>> sdq.pl.temporal_expression(
    ...     adata_sim,
    ...     gene="Spi1",
    ...     groupby="final_state",
    ...     cmap={"Mon.": "orange", "Neu.": "#4a7298"}
    ... )
    """
    # -- Get gene index -------------------------------------------------------
    gene_ids = adata_sim.uns[gene_ids_key]
    if isinstance(gene_ids, (pd.Index, pd.Series)):
        gene_ids = gene_ids.tolist()
    elif isinstance(gene_ids, np.ndarray):
        gene_ids = gene_ids.tolist()
    elif not isinstance(gene_ids, list):
        gene_ids = list(gene_ids)

    if gene not in gene_ids:
        preview = gene_ids[:5] if len(gene_ids) >= 5 else gene_ids
        raise ValueError(
            f"Gene '{gene}' not found in adata_sim.uns['{gene_ids_key}']. "
            f"Available genes: {preview}..."
        )
    gene_idx = gene_ids.index(gene)

    # -- Extract expression and metadata --------------------------------------
    expression = adata_sim.obsm[use_key][:, gene_idx]
    time = adata_sim.obs[time_key].values
    groups = adata_sim.obs[groupby].values

    # -- Build dataframe for groupby operations -------------------------------
    df = pd.DataFrame({
        "expression": expression,
        "time": time,
        "group": groups,
    })

    # -- Compute mean and std per (time, group) -------------------------------
    stats = df.groupby(["time", "group"])["expression"].agg(["mean", "std"])
    stats = stats.reset_index()

    # -- Setup figure ---------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # -- Default colormap -----------------------------------------------------
    unique_groups = df["group"].unique()
    if cmap is None:
        default_colors = plt.cm.tab10.colors
        cmap = {g: default_colors[i % len(default_colors)] for i, g in enumerate(unique_groups)}

    # -- Plot each group ------------------------------------------------------
    for group in unique_groups:
        group_data = stats[stats["group"] == group].sort_values("time")
        t = group_data["time"].values
        mean = group_data["mean"].values
        std = group_data["std"].values

        color = cmap.get(group, "gray")

        # Plot mean line
        ax.plot(t, mean, color=color, linewidth=linewidth, label=group, zorder=2)

        # Plot std fill-between
        if show_std:
            ax.fill_between(
                t,
                mean - std,
                mean + std,
                color=color,
                alpha=std_alpha,
                linewidth=0,
                zorder=1,
            )

    # -- Formatting -----------------------------------------------------------
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)

    if title is None:
        title = f"$\\it{{{gene}}}$"
    ax.set_title(title, fontsize=11)

    if grid:
        ax.grid(True, alpha=grid_alpha, zorder=0)

    if show_legend:
        ax.legend(
            loc=legend_loc,
            frameon=True,
            facecolor="white",
            edgecolor="lightgray",
            fontsize=8,
        )

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -- Save -----------------------------------------------------------------
    if save:
        if savename is None:
            savename = f"scDiffEq.temporal_expression.{gene}.{save_format}"
        plt.savefig(savename, format=save_format, dpi=dpi, bbox_inches="tight")

    return ax
