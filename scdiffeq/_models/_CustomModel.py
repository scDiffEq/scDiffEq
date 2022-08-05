
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

class CustomModel(BaseModel):
    def __init__(
        self,
        adata,
        func,
        lr=1e-3,
        seed=0,
        dt=0.5,
        t_scale=0.02,
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
        
#         log_path = os.path.join(log_path) # , "lightning_logs")
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
            
        self.trainer = Trainer(
            logger=self._logger,
            max_epochs=epochs,
            gpus=gpus,
            log_every_n_steps=log_every_n_steps,
            callbacks=self._callback_list,
            **trainer_kwargs
        )

    def fit(self, dataset):

        self._dataset = dataset
        self.trainer.fit(self, self._dataset["train"], self._dataset["val"])

    def evaluate(self):

        self.test_loss = self.trainer.test(self, self._dataset["test"], verbose=False)