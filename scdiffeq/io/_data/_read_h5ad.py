
__module_name__ = "__init__.py"
__doc__ = """I/O Functions: h5ad interactions."""
__author__ = ", ".join(["Michael E. Vinyard",])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages: -----------------------------------------------------------------------
import anndata


# API-facing function: -------------------------------------------------------------------
def read_h5ad(h5ad_path: str, silent: bool = False, annotate_path: bool = True) -> anndata.AnnData:

    """
    Read an AnnData object from a path to a .h5ad file.

    Parameters:
    -----------
    h5ad_path: str
        path to .h5ad file.

    silent: bool, default = False
        If True, the anndata object is not printed.
        
    annotate_path: bool, default = True
        If True, adds the h5ad_path to the read adata object in adata.uns['h5ad_path'].

    Returns:
    --------
    adata (anndata._core.anndata.AnnData). The (annotated) single-cell data matrix of shape
        n_obs × n_vars. Rows correspond to cells and columns to genes. 
        For more: https://anndata.readthedocs.io/en/latest/.

    Notes:
    ------
    (1) Documentation for AnnData: https://anndata.readthedocs.io/en/stable/

    """

    adata = anndata.read_h5ad(h5ad_path)
    adata.uns['h5ad_path'] = h5ad_path

    if not silent:
        print(adata)

    return adata