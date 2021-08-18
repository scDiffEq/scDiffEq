import os
import pickle


def _write_AnnData(
    adata, label="testing_results", outpath="./", scdiffeq_outs_dir="scdiffeq_adata"
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

    scdiffeq_outs_dir = os.path.join(outpath, scdiffeq_outs_dir)
    if not os.path.exists(scdiffeq_outs_dir):
        os.mkdir(scdiffeq_outs_dir)

    h5ad_out_path = os.path.join(scdiffeq_outs_dir, (label + ".h5ad"))
    pca_pkl_out_path = os.path.join(scdiffeq_outs_dir, (label + ".pkl"))

    print("Writing results to:\n\n\t{}".format(h5ad_out_path))
    print("\t{}".format(pca_pkl_out_path))

    tmp_uns_pca = adata.uns["pca"]

    pickle.dump(adata.uns["pca"], open(pca_pkl_out_path, "wb"))
    del adata.uns["pca"]
    adata.write_h5ad(h5ad_out_path)
    adata.uns["pca"] = tmp_uns_pca
    print("\n\t...Done.")
