
import torch_adata

from ._base_lightning_anndata_module import BaseLightningAnnDataModule

class LARRY_LightningDataModule(BaseLightningAnnDataModule):
    def prepare_data(self):
        # download here, do not assign states
        pass

    def setup(self, stage):

        if stage == "fit":
            train_dataset = torch_adata.AnnDataset(
                self.adata[self.adata.obs["train"]],
                use_key="X_pca",
                groupby="Time point",
                obs_keys=["W", "v"],
            )
            self.train_dataset, self.val_dataset = torch_adata.split(
                train_dataset, percentages=[0.8, 0.2]
            )

        elif stage == "test":
            self.test_dataset = torch_adata.AnnDataset(
                self.adata[self.adata.obs["test"]],
                use_key="X_pca",
                groupby="Time point",
                obs_keys=["W", "v"],
            ).dataset

        elif stage == "predict":
            self.predict_dataset = torch_adata.AnnDataset(
                adata, use_key="X_pca",
                groupby="Time point",
                obs_keys=["W", "v"],
            ).dataset
            
