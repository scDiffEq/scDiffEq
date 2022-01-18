
import numpy as np
import pandas as pd

def _annotate_clonal_barcodes(adata):

    """
    Requires `adata.obsm['X_clone']` to be present. 
    """

    df = pd.DataFrame(
        np.array(adata.obsm["X_clone"].nonzero()).T, columns=["cell_idx", "clone_idx"]
    )
    adata.obs["cell_idx"] = pd.Categorical(adata.obs.index.astype(str))
    df["cell_idx"] = pd.Categorical(df.cell_idx.astype(str))
    df_obs = adata.obs.merge(df, on="cell_idx", how="left")
    df_obs.index = df_obs.index.astype(str)
    
    return df_obs