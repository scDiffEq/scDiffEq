from glob import glob
import anndata as a
import pickle


def _read_AnnData():

    """
    Writes h5ad and pkl file for outputs of scdiffeq method.
    Mainly written to fill the gaps of AnnData (i.e., cannot save pca result from sklearn).
    Creates output directory if needed.

    Parameters:
    -----------
    adata
        AnnData object.

    label
        experiment-specific label.

    outpath
        directory where outs_directory should be placed.

    scdiffeq_outs_dir
        scdiffeq-specific outs directory

    Returns:
    --------
    None
    """

    h5ad_search_path = os.path.join(outpath, scdiffeq_outs_dir, (label + "*")) + ".h5ad"
    pkl_search_path = os.path.join(outpath, scdiffeq_outs_dir, (label + "*")) + ".pkl"

    adata = a.read_h5ad(h5ad_search_path)
    adata.uns["pca"] = pickle.load(open(pkl_search_path, "rb"))

    return adata