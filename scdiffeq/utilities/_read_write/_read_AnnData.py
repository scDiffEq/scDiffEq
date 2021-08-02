from glob import glob
import anndata as a
import os, pickle


def _choose_path_from_glob_list(glob_path_list):

    """"""

    if len(glob_path_list) > 1:
        print(
            glob_path_list,
            "; please narrow your search criteria using the `label` parameter.",
        )
    else:
        glob_path_item = glob_path_list[0]

    return glob_path_item


def _read_AnnData(
    label="testing_results",
    outpath="./",
    scdiffeq_outs_dir="scdiffeq_adata",
    silence=False,
):

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

    h5ad_search_path = _choose_path_from_glob_list(
        glob(os.path.join(outpath, scdiffeq_outs_dir, (label + "*h5ad")))
    )
    pkl_search_path = _choose_path_from_glob_list(
        glob(os.path.join(outpath, scdiffeq_outs_dir, (label + "*.pkl")))
    )

    adata = a.read_h5ad(h5ad_search_path)
    adata.uns["pca"] = pickle.load(open(pkl_search_path, "rb"))

    if not silence:
        print(adata)

    return adata