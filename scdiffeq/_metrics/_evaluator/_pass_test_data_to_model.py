
import numpy as np
import os
import time
import torch
import torchsde

from ._evaluator_utilities import _specify_device
from ..._model._scDiffEq_Model import _scDiffEq_Model
from ..._data._Weinreb2020._RetrieveData import _RetrieveData


def _reshape(X_pred, N, n_cells, dim):
    
    X_pred_ = []
    for X_t in X_pred:
        X_t_ = []
        for sample in range(N):
            X_t_.append(X_t.reshape(n_cells, N, dim)[:, sample, :])
        X_pred_.append(torch.stack(X_t_))
    
    return torch.stack(X_pred_)

def _forward_integrate_model(model, X0, N=1, t=[0, 0.01, 0.02], device=0):
    
    device = _specify_device(device)
    n_cells = X0.shape[0]
    dim = X0.shape[1]
    
    X0_ = X0[:, np.newaxis, :].expand(n_cells, N, -1).reshape(-1, dim).to(device)
    
    func = model._nn_func.to(device)
    t = torch.Tensor(t).to(device)
    
    func.batch_size = X0_.shape[0]
    
    start = time.time()
    with torch.no_grad():
        X_pred = torchsde.sdeint(func, X0_, t)
    end = time.time()

    return _reshape(X_pred, N, n_cells, dim)

def _return_input_data(adata, use_key="X_pca"):
    
    """"""
    
    data = _RetrieveData(adata)
    data.neu_mo_test_set_early()
    df = data._df
    X0 = torch.Tensor(adata[df.index].obsm[use_key])
    
    return X0

def _echo_model_status(n, epoch, epochs_to_evaluate, N):
    
    if n is 0:
        print("Evaluating model (using N={} generative simulations) at epoch: {}".format(N, epoch), end=", ")
    elif n == len(epochs_to_evaluate) - 1:
        print("{}".format(epoch), end="")
    else:
        print("{}".format(epoch), end=", ")
        
def _return_model_path(run_path, epoch):
    
    model_specification = "model/{}_epochs.model".format(epoch)
    model_path = os.path.join(run_path, model_specification)
    
    return model_path

def _load_trained_model(adata, run_path, epoch, layers, nodes, dim=50):
    
    """"""
    
    model_path = _return_model_path(run_path, epoch)
    model = _scDiffEq_Model(
        adata,
        in_dim=dim,
        out_dim=dim,
        layers=layers,
        nodes=nodes,
        device="cpu",
        silent=True,
        evaluate_only=True,
    )
    model.load(model_path)
    
    return model
    
def _pass_test_data_to_model(adata,
                             epochs_to_evaluate,
                             run_path,
                             layers,
                             nodes,
                             seed,
                             evaluation_outpath,
                             N=2000,
                             use_key="X_pca",
                             device=0):
    
    X0 = _return_input_data(adata, use_key)
    dim = X0.shape[-1]
    
    X_model_pred = {}
    
    for n, epoch in enumerate(epochs_to_evaluate):
        _echo_model_status(n, epoch, epochs_to_evaluate, N)
        model = _load_trained_model(adata, run_path, epoch, layers, nodes, dim=50)
        X_pred = _forward_integrate_model(model, X0, N=N, device=device)
        X_model_pred[epoch] = X_pred.cpu().detach()        
        X_pred_outpath = os.path.join(evaluation_outpath,
                                      "X_pred.N{}.{}layers{}nodes.seed{}.epoch{}".format(N, layers, nodes, seed, epoch))
        torch.save(X_model_pred[epoch], X_pred_outpath)
        
    return X_model_pred
