

import torch_adata

from ._lightning_anndata_module import LightningAnnDataModule

class LARRY_LightningDataModule(LightningAnnDataModule):
    def prepare_data(self):
        """fetch the data. do any required preprocessing."""
        adata = self.adata
        adata.obs["W"] = adata.obs["v"] = 1
        self.dataset = torch_adata.AnnDataset(
            self.adata, use_key=self.hparams["use_key"], groupby="Time point", obs_keys=["W"]
        )

    def setup(self, stage=None):
        """Setup the data for feeding towards a specific stage"""
        if (stage == None) or (stage == "fit"):
            self.train_dataset, self.val_dataset = torch_adata.tl.split(self.dataset, percentages=[0.8, 0.2])