import numpy as np
import torch


def _prepare_data_no_lineages(
    adata, t_key="Time point", use_key="X_pca", save=True, save_path="X_train.pt"
):

    t_array = np.sort(adata.obs[t_key].unique())

    X_train = {}
    for _t in t_array:
        _idx = adata.obs.loc[adata.obs[t_key] == _t].index
        X_train[_t] = torch.Tensor(adata[_idx].obsm[use_key])

    if save:
        torch.save(X_train, save_path)

    t_train = torch.Tensor(np.sort(adata.obs["t"].unique()))

    return X_train, t_train