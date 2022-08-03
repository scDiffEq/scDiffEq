
__module_name__ = "_CustomModel.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
from pytorch_lightning import Trainer, utilities, loggers
import numpy as np
import torch


# import local dependencies #
# ------------------------- #
from ._core._BaseModel import BaseModel

from ._core._lightning_callbacks import SaveHyperParamsYAML


class CustomModel(BaseModel):
    def __init__(
        self,
        adata,
        func,
        lr=1e-3,
        seed=0,
        dt=0.5,
        epochs=2500,
        time_key="Time point",
        gpus=torch.cuda.device_count(),
        log_path="./",
        flush_logs_every_n_steps=1,
        log_every_n_steps=1,
        logger_kwargs={},
        trainer_kwargs={},
    ):
        train_adata = adata[adata.obs['train']]
        test_adata  = adata[adata.obs['test']]
        
        train_t = torch.Tensor(np.sort(train_adata.obs[time_key].unique()))
        test_t  = torch.Tensor(np.sort(test_adata.obs[time_key].unique()))
                
        super(CustomModel, self).__init__(func,
                                          train_t=train_t,
                                          test_t=test_t,
                                          dt=dt,
                                          lr=lr,
                                          seed=seed,
                                         )

        logger = loggers.CSVLogger(
            log_path, flush_logs_every_n_steps=flush_logs_every_n_steps, **logger_kwargs
        )
        self.trainer = Trainer(
            logger=logger,
            max_epochs=epochs,
            gpus=gpus,
            log_every_n_steps=log_every_n_steps,
            callbacks=[SaveHyperParamsYAML()],
            **trainer_kwargs
        )

    def fit(self, dataset):

        self._dataset = dataset
        self.trainer.fit(self, self._dataset["train"], self._dataset["val"])

    def evaluate(self):

        self.test_loss = self.trainer.test(self, self._dataset["test"], verbose=False)