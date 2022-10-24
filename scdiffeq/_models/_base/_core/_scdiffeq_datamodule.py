
import torch_adata

BaseLightningDataModule = torch_adata.tl.BaseLightningDataModule


def _augment_obs_with_W(adata, w_key="W"):
    w_hat_key = "{}_hat".format(w_key)
    if not w_key in adata.obs.columns.tolist():
        adata.obs[w_key] = 1
    adata.obs[w_hat_key] = 1
    return [w_key, w_hat_key]


class scDiffEqDataModule(BaseLightningDataModule):
    def prepare_data(self):
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
        pass
