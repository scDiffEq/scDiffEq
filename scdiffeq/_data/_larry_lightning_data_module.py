

import torch_adata

from ._base_lightning_anndata_module import BaseLightningAnnDataModule

class LARRY_LightningDataModule(BaseLightningAnnDataModule):
    def prepare_data(self):
        """fetch the data. do any required preprocessing."""
        adata = self.adata
        adata.obs["W"] = adata.obs["v"] = 1
        self.dataset = torch_adata.AnnDataset(
            self.adata, use_key="X_pca", groupby="Time point", obs_keys=["W"]
        )

    def setup(self, stage=None):
        """Setup the data for feeding towards a specific stage"""
        if (stage == None) or (stage == "fit"):
            train, val = torch_adata.split(self.dataset, percentages=[0.8, 0.2])
            self.train_dataset = train.dataset
            self.val_dataset = val.dataset
