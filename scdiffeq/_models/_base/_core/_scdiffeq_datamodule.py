
import torch_adata
import os

from torch_adata import BaseLightningDataModule


def _augment_obs_with_W(adata, w_key="W"):
    w_hat_key = "{}_hat".format(w_key)
    if not w_key in adata.obs.columns.tolist():
        adata.obs[w_key] = 1
    adata.obs[w_hat_key] = 1
    return [w_key, w_hat_key]


class scDiffEqDataModule(BaseLightningDataModule):
    """
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningDataModule.html
    """
    def prepare_data(self):
        """
        download, split, etc...
        only called on 1 GPU/TPU in distributed
        """
        self.groupby = self.time_key
        self._w_keys = _augment_obs_with_W(self.adata, w_key="W")
        
        split = torch_adata.tl.AnnDatasetSplit(
            self.adata,
            groupby=self.groupby,
            obs_keys=self._w_keys,
        )
        split.on_test_train()
        split.allocate_validation()

        self.train_dataset = split.train_dataset
        self.val_dataset = split.val_dataset
        self.test_dataset = split.test_dataset

    def setup(self, stage: str = None):
        """
        Make assignments here (val/train/test split). Called on every process in DDP.
        """
        pass


def configure_data(adata, time_key, batch_size=2000, num_workers=os.cpu_count(), percent_val=0.2):
    
    """"""
    
    kwargs = {"time_key": time_key, "percent_val": percent_val}
    
    return scDiffEqDataModule(adata, batch_size, num_workers, **kwargs)
