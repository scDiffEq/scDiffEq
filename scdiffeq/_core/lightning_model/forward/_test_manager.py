import os
from tqdm.notebook import tqdm
import scdiffeq as sdq
import warnings
from pytorch_lightning import Trainer #, seed_everything
from torch.utils.data import DataLoader
from torch_adata import AnnDataset
import torch
import numpy as np


warnings.filterwarnings("ignore")

from ...utils import AutoParseBase

class TestManager(AutoParseBase):

#     global_seet = seed_everything(0)

    predicted = []

    def __init__(
        self,
        diffeq,
        n_predictions=25,
        test_adata=None,
        use_key="X_pca",
        groupby="Time point",
        batch_size=10_000,
        shuffle=True,
    ):

        self.__parse__(locals(), ignore=["diffeq"])
        self.__configure__(diffeq=diffeq)

    def _trainer(self):
        #         if not hasattr(self, "_trainer"):
        self._trainer = Trainer(
            accelerator="gpu",
            devices=1,
            limit_test_batches=1,
            enable_progress_bar=False,
            reload_dataloaders_every_n_epochs=1,
#             logger=self.fit_trainer.logger,
        )

    def __configure__(self, diffeq):
        
        self.fit_trainer = diffeq.LightningTrainer
        
        self.model = diffeq.LightningModel
        self.potential = self.model.mu_is_potential

        if self.potential:
            self._trainer = diffeq.LightningTestTrainer
            self.forward = getattr(self._trainer, "fit")
            self._train_loader = diffeq.LightningDataModule.train_dataloader()
            if not isinstance(self.test_adata, type(None)):
                self._val_loader = self.test_dataloader
            else:
                self._val_loader = diffeq.LightningDataModule.test_dataloader()
        else:
            self._trainer()
            self.forward = getattr(self._trainer, "test")
            if not isinstance(self.test_adata, type(None)):
                self._loader = self.test_dataloader
            else:
                self._loader = diffeq.LightningDataModule
                
    @property
    def test_dataloader(self):
        self._configure_test_dataset()
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=os.cpu_count(),
        )

    def _configure_test_dataset(self):

        if not isinstance(self.test_adata, type(None)):
            self.test_dataset = AnnDataset(
                self.test_adata, use_key=self.use_key, groupby=self.groupby, silent=True
            )

    @property
    def save_fname(self):
        return os.path.join(self.fit_trainer.logger.log_dir, self.savename)

    def __call__(self, savename="test_predictions.npy"):

        self.__parse__(locals())

        for i in tqdm(range(self.n_predictions)):
            if self.potential:
                predicted = self.forward(
                    self.model,
                    train_dataloaders=self._train_loader,
                    val_dataloaders=self._val_loader,
                )
            else:
                predicted = self.forward(self.model, self._loader, verbose=False)[0][
                    "test_1_positional"
                ]
            self.predicted.append(predicted)

        self.predicted = np.array(self.predicted)
        
        if savename:
            print("Saving to: {}".format(self.save_fname))
            np.save(self.save_fname, self.predicted)

        return self.predicted
