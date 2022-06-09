import numpy as np
import torch


def _prepare_data_no_lineages(
    adata,
    time_key="Time point",
    use_key="X_pca",
    save=True,
    save_path="X_train.pt",
    silent=False,
):

    """AnnData to tensor format (not batched yet)"""

    t_array = np.sort(adata.obs[time_key].unique())

    X_data = {}
    for _t in t_array:
        _idx = adata.obs.loc[adata.obs[time_key] == _t].index
        if "matrix" in type(adata[_idx].obsm[use_key]).__name__:
            X_data[_t] = torch.Tensor(adata[_idx].obsm[use_key].toarray())
        else:
            X_data[_t] = torch.Tensor(adata[_idx].obsm[use_key])         
            
            
    if save:
        torch.save(_idx, save_path)
        if not silent:
            print("Saving data to: {}".format(save_path))

    t = torch.Tensor(np.sort(adata.obs["t_augmented"].unique()))

    return X_data, t

def _prepare_lineage_data():
    print("Using lineage data... function to be defined...")
    return None, None


def _prepare_data(adata, TrainingProgram, save, save_path, silent):
    
    """"""

    if TrainingProgram["use_lineages"]:
        return _prepare_lineage_data()
        
    else:
        print("Not using lineages...")
        return _prepare_data_no_lineages(
            adata,
            TrainingProgram["time_key"],
            TrainingProgram["use_key"],
            save,
            save_path,
            silent,
        )
    