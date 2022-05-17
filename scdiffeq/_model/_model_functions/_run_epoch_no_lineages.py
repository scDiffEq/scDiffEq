
import numpy as np
import torch


from ..._tools._forward_integrate import _forward_integrate


def _prepare_data_no_lineages(
    adata, t_key="Time point", use_key="X_pca", save=True, save_path="X_train.pt"
):

    t_array = np.sort(model._adata.obs[t_key].unique())

    X_train = {}
    for _t in t_array:
        _idx = adata.obs.loc[adata.obs[t_key] == _t].index
        X_train[_t] = torch.Tensor(adata[_idx].obsm[use_key])

    if save:
        torch.save(X_train, save_path)

    t_train = torch.Tensor(np.sort(adata.obs["t"].unique()))

    return X_train, t_train


def _create_batch_indices_no_lineages(X_train, batch_size):

    n_cells_max = max([v.shape[0] for v in X_train.values()])
    n_batches = int(n_cells_max / batch_size)

    batches = {}

    for _t, _X in X_train.items():
        if _X.shape[0] == n_cells_max:
            replace_flag = False
        else:
            replace_flag = True

        _idx = np.random.choice(
            _X.shape[0], [n_batches, int(batch_size)], replace=replace_flag
        )
        batches[_t] = _idx

    return batches


def _fetch_batched_data_no_lineages(X_train, batch_size=2000):

    batch_indices = _create_batch_indices_no_lineages(X_train, batch_size=batch_size)

    X_batched = []
    for _t, batches in batch_indices.items():
        X_batched.append(torch.stack([X_train[_t][batch] for batch in batches]))

    return torch.stack(X_batched)

def _time_resolve_batch_loss_no_lineages(X_batch, X_pred, LossFunc):

    batch_loss = []

    for _t in range(len(X_batch)):
        batch_loss_t = LossFunc(X_batch[_t], X_pred[_t])
        batch_loss.append(batch_loss_t)

    return torch.stack(batch_loss)

def _run_epoch_no_lineages(X_train, t_train, func, optimizer, LossFunc, device, batch_size=2000):
    
    """
    
    Returns:
    --------
    epoch_loss
        The mean epoch loss at each time.
        type: numpy.ndarray
        
    """
    
    X_batched = _fetch_batched_data_no_lineages(X_train, batch_size=batch_size)
    
    epoch_loss = []
    optimizer.zero_grad()
    for batch in range(X_batched.shape[1]):
        X_batch = X_batched[:, batch, :, :]
        X_pred = _forward_integrate(func, X_batch, t_train, device=device)
        batch_loss = _time_resolve_batch_loss_no_lineages(X_batch, X_pred, LossFunc)
        batch_loss.sum().backward()
        optimizer.step()
        batch_loss = batch_loss.detach().cpu().numpy()
        epoch_loss.append(batch_loss)

    return np.stack(epoch_loss).mean(0)