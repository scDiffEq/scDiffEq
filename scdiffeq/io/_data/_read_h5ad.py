# import packages: ------------------------------------------------------------
import anndata


# API-facing function: --------------------------------------------------------
def read_h5ad(
    h5ad_path: str, silent: bool = False, annotate_path: bool = True
) -> anndata.AnnData:
    """Read an AnnData object from a path to a .h5ad file.

    Args:
        h5ad_path (str): Path to .h5ad file.
        silent (bool): If True, the AnnData object is not printed.
        annotate_path (bool): If True, adds the h5ad_path to the read AnnData object in adata.uns['h5ad_path'].

    Returns:
        anndata.AnnData
            The (annotated) single-cell data matrix of shape n_obs × n_vars. Rows correspond to cells and columns to genes.
        For more information, visit: https://anndata.readthedocs.io/en/latest/.

    Notes:
        Documentation for AnnData: https://anndata.readthedocs.io/en/stable/
    """

    adata = anndata.read_h5ad(h5ad_path)
    adata.uns["h5ad_path"] = h5ad_path

    if not silent:
        print(adata)

    return adata
