
__module_name__ = "_read_h5ad.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
from anndata import read_h5ad


def _read_h5ad(h5ad_path, silent=False):

    """
    Read an AnnData object from a path to a .h5ad file.

    Parameters:
    -----------
    h5ad_path [ required ]
        path to .h5ad file.
        type: str


    silent [ optional ]
        If True, the anndata object is not printed.
        default: False
        type: bool

    Returns:
    --------
    adata
        type: anndata._core.anndata.AnnData

    Notes:
    ------
    (1) Documentation for AnnData: https://anndata.readthedocs.io/en/stable/

    """

    adata = read_h5ad(h5ad_path)

    if not silent:
        print(adata)

    return adata