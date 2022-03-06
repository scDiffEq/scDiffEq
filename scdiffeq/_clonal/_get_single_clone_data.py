
import numpy as np
import torch

from ._subset_single_clonal_lineage import _subset_single_clonal_lineage

def _get_single_clone_data(
    adata,
    clone_idx=False,
    clone_adata=False,
    use_key="X_pca",
    timepoint_key="Time point",
):

    """Returns y, y0, and t as tensors"""

    X_use = adata.obsm[use_key]

    if not clone_adata:
        clone_adata = _subset_single_clonal_lineage(adata, clone_idx)

    clone_obs = clone_adata.obs
    t = np.sort(clone_obs[timepoint_key].unique())
    clone_y0_idx = clone_obs.loc[clone_obs[timepoint_key] == t[0]].index.astype(int)

    y = torch.Tensor(X_use[clone_adata.obs.index.astype(int)])
    y0 = torch.Tensor(X_use[clone_y0_idx])

    return y, y0, torch.Tensor(t)