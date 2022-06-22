
import numpy as np
import torch

from .. import _loss_functions as loss_funcs

def _pass_to_VAE_scDiffEq_model(model, NeuralDiffEq, X, t, pretrain_VAE):
    
    X0 = X[0]
    
    if pretrain_VAE:
        print("pre-training VAE")
        model.encode(X)
    else:
        model.encode(X0)
                
    model.reparameterize()
    
    if not pretrain_VAE:
        model.forward_integrate(NeuralDiffEq, t)
    else:
        print("Training only the VAE as part of pre-training. Not performing forward_int with NDE")
    
    
    n_t = X.shape[0]
    n_dim = X.shape[-1]
    
    return model.decode().reshape(n_t, -1, n_dim), model._mu, model._log_var

import torchsde

def _pass_to_scDiffEq_model(model, X, t):
    
    X0 = X[0]
    
    return torchsde.sdeint(model, X0, t), None, None

#     return model.forward_integrate(X0, t), None, None


def _pass_to_model(model, X, t, pretrain_VAE):
    
    if not model['VAE'] == None:
        print("passing training the right thing")
        return _pass_to_VAE_scDiffEq_model(model["VAE"], model["NeuralDiffEq"], X, t, pretrain_VAE)
    else:
        return _pass_to_scDiffEq_model(model["NeuralDiffEq"], X, t)


def _model_forward(Model, X, t, reconst_loss_func, reparam_loss_func, pretrain_VAE_epochs, device):

    X_pred, mu, log_var = _pass_to_model(Model, X, t, pretrain_VAE_epochs)
    X_pred = torch.nan_to_num(X_pred, 0)
    
    loss = loss_funcs.calculate_loss(
        X_pred, X, mu, log_var, reconst_loss_func, reparam_loss_func, device
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

    return torch.stack(batched_loss)

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

# import matplotlib.pyplot as plt
# import pydk
# umap_model = pydk.load_pickled(
#     "/home/mvinyard/data/scdiffeq_data/Weinreb2020_preprocessed/loaded/Weinreb2020.adata.umap"
# )


def _batched_training_model_pass(
    X_data, Model, optimizer, t, reconst_loss_func, reparam_loss_func, batch_size, pretrain_VAE, device
):
    
    Model['NeuralDiffEq'] = Model['NeuralDiffEq'].to(device)
    batched_data = _fetch_batched_data_no_lineages(X_data, batch_size=batch_size).to(device)
    epoch_loss = []
    for i in range(batched_data.shape[1]):
        X_batch = batched_data[:, i, :, :]
        X_pred, loss = _model_forward(
                Model,
                X_batch,
                t,
                reconst_loss_func,
                reparam_loss_func,
                pretrain_VAE,
                device,
            )
        loss.total_loss.sum().backward()
        optimizer.step()
        loss = loss.total_loss.detach().cpu()
        epoch_loss.append(loss)

    return X_pred, torch.stack(epoch_loss)
    

# def _batched_training_model_pass(
#     X_data, Model, optimizer, t, reconst_loss_func, reparam_loss_func, batch_size, pretrain_VAE, device
# ):
    
#     Model['VAE'] = Model['VAE'].to(device)
#     Model['NeuralDiffEq'] = Model['NeuralDiffEq'].to(device)
    
#     batched_data = _fetch_batched_data_no_lineages(X_data, batch_size=batch_size).to(device)
        
#     epoch_loss = []
        
#     for i in range(batched_data.shape[1]):
# #         print("-- Current Batch: {} --".format(i+1))
#         X_batch = batched_data[:, i, :, :] # t x batch x cell x dim
        
#         if i == 1:
# #             print("THIS IS THE ONE I AM PAYING ATTENTION TO")
#             Model['VAE'].encode(X_batch)
#             return Model, None
            
#         else:
#             X_pred, loss = _model_forward(
#                 Model,
#                 X_batch,
#                 t,
#                 reconst_loss_func,
#                 reparam_loss_func,
#                 pretrain_VAE,
#                 device,
#             )

#### plotting module #### - silenced only to speed things up now that I can see it's working
#         X_batch_umap = [umap_model.transform(X_batch[_t].cpu().numpy().reshape(-1, 2)) for _t in range(len(X_batch))]
#         X_pred_umap = [umap_model.transform(X_pred[_t].detach().cpu().numpy().reshape(-1, 2)) for _t in range(len(X_pred))]
#         for _t, xut in enumerate(X_batch_umap):
#             plt.scatter(xut[:,0], xut[:,1], c="lightgrey")
#         for _t, xut in enumerate(X_pred_umap):
#             plt.scatter(xut[:,0], xut[:,1], label=_t)
#         plt.title("pred")
#         plt.show()
#### plotting module #### - silenced only to speed things up now that I can see it's working
        
#             print(loss.total_loss)
#             loss.total_loss.sum().backward()
#             optimizer.step()
#             loss = loss.total_loss.detach().cpu()
#     #         print("loss... {}".format(loss))
#             epoch_loss.append(loss)

#     return X_pred, epoch_loss