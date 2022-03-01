
import numpy as np
import torch
from torchsde import sdeint

def _isolate_example_cell(adata, batch_size=1, idx=False, device="cpu"):

    """"""

    if not idx:
        d2_idx = adata.obs.loc[adata.obs["Time point"] == 2].index.astype(int)
        idx = np.random.choice(d2_idx, batch_size)
    y0 = torch.Tensor(adata.obsm["X_pca"][idx]).to(device)

    return y0, idx


def _formulate_simulation_inputs(
    adata, diffeq, n_simulations=5, batch_size=1, idx=False, device="cpu"
):

    """"""

    t = torch.Tensor([2, 4, 6])

    y0_single, idx = _isolate_example_cell(adata, batch_size, idx, device)
    y0 = torch.Tensor(
        np.tile(y0_single, n_simulations).reshape(n_simulations, y0_single.shape[1])
    )
    diffeq.network_model.batch_size = n_simulations

    return diffeq.network_model.to(device), y0.to(device), t.to(device), idx


def _simulate_predictions(
    adata, umap, diffeq, device="cpu", n_simulations=5, batch_size=1, idx=False
):

    """"""

    func, y0, t, idx = _formulate_simulation_inputs(
        adata, diffeq, n_simulations, batch_size, idx, device
    )
    return sdeint(func, y0, t).to(device), idx


def _isolate_clonal_data(adata):
    
    """Assumes clonal matrix is defined"""
    
    Xc = adata.obsm["X_clone"].toarray()
    adata_clonal = adata[np.where(Xc.sum(axis=1) > 0)].copy()
    adata_clonal.obs = adata_clonal.obs.reset_index()
    
    return adata_clonal
        