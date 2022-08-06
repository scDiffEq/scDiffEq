
__module_name__ = "_CustomModel.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
from pytorch_lightning import Trainer, utilities, loggers
import numpy as np
import torch
import os
import pydk


# import local dependencies #
# ------------------------- #
from ._core._BaseModel import BaseModel
from ._core._lightning_callbacks import SaveHyperParamsYAML, timepoint_recovery_checkpoint, TrainingSummary

#### ------------------- ####

def _timespan(t):
    return (t.max() - t.min()).item()

def _get_t_final(adata, device, time_key="Time point", use_key="X_pca"):

    df = adata.obs.copy()

    time_cell_counts = df[time_key].value_counts()
    t_final = time_cell_counts.idxmax()
    n_cells_t_final = time_cell_counts.max()
    X = torch.Tensor(
        adata[df.loc[df[time_key] == t_final].index].obsm[use_key]
    )

    return {"X": X, "n_cells": n_cells_t_final, "t": t_final}


class CustomModel(BaseModel):
    def __init__(
        self,
        adata,
        func,
        lr=1e-3,
        seed=0,
        dt=0.1,
        alpha=0.5,
        t_scale=0.02,
        regularize=False,
        epochs=2500,
        optimizer=torch.optim.RMSprop,
        time_key="Time point",
        checkpoint_callback=False,
        gpus=torch.cuda.device_count(),
        log_path="./",
        flush_logs_every_n_steps=1,
        log_every_n_steps=1,
        logger_kwargs={},
        trainer_kwargs={},
        checkpoint_kwargs={},
    ):
        
        pydk.mkdir_flex(log_path)
        
        train_adata = adata[adata.obs['train']]
        test_adata  = adata[adata.obs['test']]
        
        train_t = torch.Tensor(np.sort(train_adata.obs[time_key].unique()))
        test_t  = torch.Tensor(np.sort(test_adata.obs[time_key].unique()))
                
        super(CustomModel, self).__init__(func,
                                          train_t=train_t,
                                          test_t=test_t,
                                          optimizer=optimizer,
                                          dt=dt,
                                          t_scale=t_scale,
                                          lr=lr,
                                          seed=seed,
                                         )
        
        self._callback_list = [SaveHyperParamsYAML(), TrainingSummary()]
        
        self._logger = loggers.CSVLogger(
            log_path, flush_logs_every_n_steps=flush_logs_every_n_steps, **logger_kwargs
        )
        if checkpoint_callback:
            ckpt_path = os.path.join(log_path, "version_{}".format(self._logger.version))
            model_checkpoint = timepoint_recovery_checkpoint(ckpt_path, **checkpoint_kwargs)
            self._callback_list.append(model_checkpoint)
            
        accelerator = None
        if torch.cuda.is_available():
            accelerator = "gpu"
            
        self.trainer = Trainer(
            logger=self._logger,
            max_epochs=epochs,
            accelerator=accelerator,
            devices=1,
            log_every_n_steps=log_every_n_steps,
            callbacks=self._callback_list,
            **trainer_kwargs
        )
        
        self._alpha = alpha
        self._regularize = regularize
        self._burn_t_final=16
        self._burn_in_steps=2
        self._tspan = {}
        self._tspan['train'] = _timespan(self._train_t)
        self._tspan['test']  = _timespan(self._test_t)
        self._X_final = _get_t_final(train_adata, self.device, time_key="Time point", use_key="X_pca")

    def fit(self, dataset):

        self._dataset = dataset
        self.trainer.fit(self, self._dataset["train"], self._dataset["val"])

    def evaluate(self):

        self.test_loss = self.trainer.test(self, self._dataset["test"], verbose=False)