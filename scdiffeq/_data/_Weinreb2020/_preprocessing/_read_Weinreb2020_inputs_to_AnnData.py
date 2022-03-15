
# import packages #
# --------------- #
from anndata import AnnData
import numpy as np
import pandas as pd
import scipy.sparse


def _read_Weinreb2020_inputs_to_AnnData(PathDict):

    """"""

    adata = AnnData(scipy.sparse.load_npz(PathDict["X"]).tocsc())
    adata.obs = pd.read_csv(PathDict["obs"], sep="\t")
    adata.var = pd.read_csv(PathDict["var"], header=None, names=["gene_id"])
    adata.obsm["X_spring"] = np.loadtxt(PathDict["X_spring"], dtype=float).reshape(
        adata.shape[0], 2
    )
    adata.obsm["X_clone"] = scipy.sparse.load_npz(PathDict["clonal"]).tocsc()
    adata.raw = adata

    return adata