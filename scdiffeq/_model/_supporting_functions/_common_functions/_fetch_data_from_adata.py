import torch
import numpy as np

def _get_y0_idx(df, time_key):

    """"""

    y0_idx = df.index[np.where(df[time_key] == df[time_key].min())]

    return y0_idx

def _get_adata_y0(adata, time_key):

    y0_idx = _get_y0_idx(adata.obs, time_key)
    adata_y0 = adata[y0_idx].copy()

    return adata_y0

def _get_y0(adata, use, time_key):

    adata_y0 = _get_adata_y0(adata, time_key)

    if use == "X":
        return torch.Tensor(adata_y0.X)

    elif use in adata.obsm_keys():
        return torch.Tensor(adata_y0.obsm[use])

    else:
        print("y0 not properly defined!")
        
def _fetch_data(adata, use="X", time_key="time"):

    """

    Assumes parallel time.
    """
        
    if use == "X":
        y = torch.Tensor(adata.X)
    else:
        y = torch.Tensor(adata.obsm[use])
    
    y0 = _get_y0(adata, use, time_key)
    t = torch.Tensor(adata.obs[time_key].unique())

    return y, y0, t

def _fetch_adata(adata_group, network_model, device, use, time_key):

    """"""
    
    batch_size = int(
        adata_group.shape[0] / adata_group.obs[time_key].nunique()
    )
    y, y0, t = _fetch_data(adata_group, use, time_key)

    return y.to(device), y0.to(device), t.to(device), batch_size