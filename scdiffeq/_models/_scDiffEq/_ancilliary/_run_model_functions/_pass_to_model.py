
import numpy as np
import torch


def _pass_to_VAE_scDiffEq_model(model, X0, t):

    model.encode(X0)
    model.reparameterize()
    model.forward_int(t)

    return model.decode(), model._mu, model._log_var


def _pass_to_scDiffEq_model(model, X0, t):

    return model.forward_int(X0, t)


def _pass_to_model(model, X0, t):

    if not model['VAE'] == None:
        return _pass_to_VAE_scDiffEq_model(model["VAE"], X0, t)
    else:
        return _pass_to_scDiffEq_model(model["NeuralDiffEq"], X0, t)


def _model_forward(Model, X, t, reconst_loss_func, reparam_loss_func, device):

    X0 = X[0]

    X_pred, mu, log_var = _pass_to_model(Model, X0, t)
    loss = _calculate_loss(
        X_pred, mu, log_var, reconst_loss_func, reparam_loss_func, device
    )

    return X_pred, loss
    
def _batched_no_grad_model_pass(
    Model,
    X,
    t,
    reconst_loss_func,
    reparam_loss_func,
    device,
):

    batched_data = _get_batches(X)
    batched_loss = []
    with torch.no_grad():
        for X_batch in batched_data:
            X_pred, loss = _model_forward(
                self._model,
                X_batch,
                t,
                self._VAE,
                self._reconst_loss_func,
                self._reparam_loss_func,
                self._device,
            )

            loss = loss.detach().cpu()
            batched_loss.append(loss)

    return batched_loss

def _prepare_data_no_lineages(
    adata, t_key="Time point", use_key="X_pca", save=True, save_path="X_train.pt"
):

    t_array = np.sort(model._adata.obs[t_key].unique())

    X_train = {}
    for _t in t_array:
        _idx = adata.obs.loc[adata.obs[t_key] == _t].index
        X_train[_t] = torch.Tensor(adata[_idx].obsm[use_key])

    if save:
        torch.save(X_data, save_path)

    t_train = torch.Tensor(np.sort(adata.obs["t"].unique()))

    return X_data, t_train


def _create_batch_indices_no_lineages(X_data, batch_size):

    n_cells_max = max([v.shape[0] for v in X_data.values()])
    n_batches = int(n_cells_max / batch_size)

    batches = {}

    for _t, _X in X_data.items():
        if _X.shape[0] == n_cells_max:
            replace_flag = False
        else:
            replace_flag = True

        _idx = np.random.choice(
            _X.shape[0], [n_batches, int(batch_size)], replace=replace_flag
        )
        batches[_t] = _idx

    return batches


def _fetch_batched_data_no_lineages(X_data, batch_size=2000):

    batch_indices = _create_batch_indices_no_lineages(X_data, batch_size=batch_size)

    X_batched = []
    for _t, batches in batch_indices.items():
        X_batched.append(torch.stack([X_data[_t][batch] for batch in batches]))

    return torch.stack(X_batched)

def _batched_training_model_pass(
    X_data, Model, optimizer, t, reconst_loss_func, reparam_loss_func, batch_size, device
):
    
    batched_data = _fetch_batched_data_no_lineages(X_data, batch_size=batch_size)
        
    for X_batch in batched_data:
        print("batch...")
        X_pred, loss = _model_forward(
            Model,
            X_batch,
            t,
            reconst_loss_func,
            reparam_loss_func,
            device,
        )
        loss.sum().backward()
        optimizer.step()
        loss = loss.detach().cpu()

        return X_pred, loss.item()