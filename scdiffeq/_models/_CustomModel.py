
__module_name__ = "_CustomModel.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
from pytorch_lightning import Trainer, utilities, loggers
import torch


# import local dependencies #
# ------------------------- #
from ._core._BaseModel import BaseModel


class CustomModel(BaseModel):
    def __init__(
        self,
        func,
        epochs=2500,
        gpus=torch.cuda.device_count(),
        log_path="./",
        flush_logs_every_n_steps=1,
        log_every_n_steps=1,
        logger_kwargs={},
        trainer_kwargs={},
    ):
        super(CustomModel, self).__init__(func)

        logger = loggers.CSVLogger(
            log_path, flush_logs_every_n_steps=flush_logs_every_n_steps, **logger_kwargs
        )
        self.trainer = Trainer(
            logger=logger,
            max_epochs=epochs,
            gpus=gpus,
            log_every_n_steps=log_every_n_steps,
            **trainer_kwargs
        )

    def fit(self, dataset):

        self._dataset = dataset
        self.trainer.fit(self, self._dataset["train"], self._dataset["val"])

    def evaluate(self):

        self.test_loss = self.trainer.test(self, self._dataset["test"], verbose=False)