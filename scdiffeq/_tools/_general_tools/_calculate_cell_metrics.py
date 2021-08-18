import numpy as np
from ..plotting._plot_potency_metrics import _plot_potency_metrics


def _count_genes(adata):

    """
    Count number of unique genes detected in each cell.

    Parameters:
    -----------
    adata
        AnnData

    Returns:
    --------
    None
        AnnData is modified in place.
    """

    adata.obs["gene_count"] = np.array(adata.X.toarray() > 0).sum(axis=1)


def _calculate_potency(adata, plot=False, save_plot=False, plot_savename=None):

    """
    Calculate cell potency as a correlate of genes detected per cell.

    Parameters:
    -----------
    adata

    Returns:
    --------
    None
        AnnData is modified in place.
    """

    try:
        adata.obs.gene_count.values[0]
    except:
        _count_genes(adata)

    adata.obs["potency"] = 1 - (adata.obs.gene_count / adata.obs.gene_count.max())

    if plot == True:
        _plot_potency_metrics(adata, save=save_plot, plot_savename=plot_savename)
