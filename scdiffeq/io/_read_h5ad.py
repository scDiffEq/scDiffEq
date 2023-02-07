
__module_name__ = "__init__.py"
__doc__ = """Main __init__ module - most user-visible API."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(["mvinyard@broadinstitute.org", "arasmuss@broadinstitute.org", "ruitong@broadinstitute.org"])


# specify version: -----------------------------------------------------------------------
__version__ = "0.0.44"


# import packages: -----------------------------------------------------------------------
import anndata


# API-facing function: -------------------------------------------------------------------
def read_h5ad(h5ad_path: str, silent: bool=False)->anndata.AnnData:

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

    adata = anndata.read_h5ad(h5ad_path)

    if not silent:
        print(adata)

    return adata