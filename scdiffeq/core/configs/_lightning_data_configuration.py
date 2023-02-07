
# -- import packages: -----------------------------------
from torch_adata import LightningAnnDataModule
import os


# -- import local dependencies: -------------------------
from .. import utils


class LightningData(LightningAnnDataModule, utils.AutoParseBase):
    def __init__(
        self,
        adata=None,
        h5ad_path=None,
        batch_size=2000,
        num_workers=os.cpu_count(),
        train_val_split=[0.8, 0.2],
        use_key="X_pca",
        groupby="Time point",  # TODO: make optional
        train_key="train",
        val_key="val",
        test_key="test",
        predict_key="predict",
        silent=True,
        **kwargs,
    ):
        super(LightningData, self).__init__()
        self.__parse__(locals(), public=[None])
        self.configure_train_val_split()

    def prepare_data(self):
        ...

    def setup(self, stage=None):
        ...