
__module_name__ = "_io.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
from anndata import AnnData
import licorice
import numpy as np
import pandas as pd
import scipy.sparse


def _read_files_downloaded_from_GitHub(downloaded_files, converted_files, outpath):

    """"""

    DataDict = {}

    for file, path in downloaded_files.items():
        if file in ["normed_counts", "clone_matrix"]:
            if not converted_files[file] == None:
                DataDict[file] = converted_files[file]
                del converted_files[file]
            else:
                DataDict[file] = scipy.sparse.load_npz("./{}.npz".format(file))
        elif file == "gene_names":
            DataDict[file] = pd.read_csv(path, sep="\t", header=None)
        elif file in [
            "metadata",
            "neutrophil_pseudotime",
            "neutrophil_monocyte_trajectory",
        ]:
            DataDict[file] = pd.read_csv(path, sep="\t")
        else:
            print("There is a problem with: {}".format(path))

    return DataDict

def _to_adata(DataDict):

    """"""

    adata = AnnData(DataDict["normed_counts"])
    adata.var["gene_ids"] = DataDict["gene_names"][0].values
    adata.obsm["X_clone"] = DataDict["clone_matrix"]
    adata.obs = DataDict["metadata"]
    neu_pseudotime = np.zeros(len(adata))
    _tmp_df = DataDict["neutrophil_pseudotime"]
    neu_idx = _tmp_df["Cell index"].values
    neu_pseudotime[neu_idx] = _tmp_df["pseudotime"].values
    adata.obs["neutrophil_pseudotime"] = neu_pseudotime
    adata.obs["neutrophil_monocyte_trajectory"] = DataDict[
        "neutrophil_monocyte_trajectory"
    ]
    adata.obs = adata.obs.fillna(-1)

    for col in adata.obs.columns:
        adata.obs[col] = pd.Categorical(adata.obs[col])

    return adata