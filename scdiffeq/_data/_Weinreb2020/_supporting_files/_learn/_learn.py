
import anndata as a
import adata_sdk
import argparse
import numpy as np
import os
import pyrequisites as pyrex
import torch
import torch.optim as optim
import torchsde
# from tqdm import tqdm
from tqdm import tqdm

from ....._model._supporting_functions._training._OptimalTransportLoss import _OptimalTransportLoss
from ....._model._scDiffEq_model import _scDiffEq
from . import _supporting_functions as funcs


class _ModelTrainer:
    def __init__(
        self,
        adata,
        func,
        LossFunc,
        outdir="./",
        clonal_idx_key="clone_idx",
        time_key="Time point",
        lr=1e-5,
        device="cpu",
        batch_size=55,
    ):

        """training class"""
        
        self._LossFunc = LossFunc
        self._obs = adata.obs.copy()
        self._X_pca = adata.obsm["X_pca"]
        self._clonal_idx_key = clonal_idx_key
        self._time_key = time_key
        self._t = torch.Tensor(np.sort(adata.obs[self._time_key].unique()))
        func.batch_size = batch_size
        func.noise_type = "general"
        func.sde_type = "ito"
        self._func = func.network_model
        self._optimizer = optim.RMSprop(self._func.parameters(), lr=lr)
        self._epoch_count = 0
        self._device = device
        self._batch_size = self._func.batch_size = batch_size
        self._clones = funcs._return_clones_present_at_all_timepoints(
            adata, self._clonal_idx_key, self._time_key
        )
        self._outdir = outdir
        self._status_file = funcs._setup_logfile(
            self._outdir, columns=["epoch", "d2", "d4", "d6", "total"]
        )
        self._best_loss = np.inf

    def update_optimizer(self, lr=1e-5):

        """ optional """

        self._optimizer = optim.RMSprop(self._func.parameters(), lr=lr)

    def run_epoch(self):

        self._optimizer.zero_grad()
        self._epoch_data = funcs._sample_clonal_lineages(
            self._X_pca, self._obs, self._clones, self._clonal_idx_key, self._time_key
        )

        self._batch_idx_list = funcs._make_epoch_batches(self._epoch_data, self._batch_size)
        self._epoch_loss = torch.zeros(
            len(self._batch_idx_list), self._epoch_data.shape[0]
        )
#         for n, batch in enumerate(tqdm(self._batch_idx_list, desc="Epoch {} Progress".format(1 + self._epoch_count))):
        for n, batch in enumerate(self._batch_idx_list):
            self._func.batch_size = batch.shape[0]
            self._batch_data = self._epoch_data[:, batch, :]
            self._batch_pred = torchsde.sdeint(
                self._func.to(self._device), self._batch_data[0].to(self._device), self._t.to(self._device)
            ).to(self._device)
            self._epoch_loss[n] = batch_loss = self._LossFunc(self._batch_pred, self._batch_data)
            batch_loss.sum().backward()
            self._optimizer.step()

        self._epoch_count += 1

        if self._epoch_loss.sum().item() < self._best_loss:
            self._best_loss = self._epoch_loss.sum().item()
            torch.save(
                self._func.state_dict(),
                os.path.join(
                    self._outdir, "best.model.epoch_{}".format(self._epoch_count)
                ),
            )
        by_day_epoch_loss = self._epoch_loss.sum(0)
        printable_loss = [l.item() for l in by_day_epoch_loss]
#         print(printable_loss)
        funcs._update_logfile(self._status_file, self._epoch_count, self._epoch_loss)
        
def _create_outdir(version, seed, nodes, layers, device, lr):
    
    """"""
    signature = ".".join(["seed_{}".format(seed),"nodes_{}".format(nodes),"layers_{}".format(layers),"device_{}".format(device),"lr_{}".format(lr),])
    outdir = os.path.join(version, signature)
    pyrex.mkdir_flex(version)
    pyrex.mkdir_flex(outdir)
    
    return outdir

class _ModelHandle:
    
    def __init__(self, adata, layers=2, nodes=5, seed=18, lr=1e-4, device=0, batch_size=40):
                 
        torch.manual_seed(seed)
        version="v1.0.5"
    
        self._adata = adata        
        self._func = _scDiffEq(in_dim=50, out_dim=50, layers=layers, nodes=nodes)
        self._outdir = _create_outdir(version, seed, nodes, layers, device, lr)
        self._model = _ModelTrainer(self._adata,
                                    self._func,
                                    _OptimalTransportLoss(device),
                                    outdir=self._outdir,
                                    batch_size=batch_size,
                                    device=device,
                                    lr=lr,
                                    )

        
    
    def learn(self, n_epochs=5000):
        
        for epoch in tqdm(range(n_epochs)):
            self._model.run_epoch()
                 