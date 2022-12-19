
# -- import packages: --------------------------------------------------------------------
from abc import abstractmethod
import os


from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from licorice_font import font_format
import anndata


# -- import local dependencies: ----------------------------------------------------------
from .._io._read_h5ad import read_h5ad

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from licorice_font import font_format
import anndata
import os


# -- main module: ------------------------------------------------------------------------
class BaseLightningAnnDataModule(LightningDataModule):
    def __init__(
        self,
        adata: anndata.AnnData = None,
        h5ad_path: str = None,
        use_key: str = None,
        groupby: str = None,
        obs_keys: [str, ..., str] = None,
        attr_names: dict({"obs": [], "aux": []}) = {"obs": [], "aux": []},
        one_hot: [bool, ..., bool] = False,
        aux_keys: [str, ..., str] = None,
        num_workers: int = os.cpu_count(),
        batch_size: int = None,
        silent: bool = False,
        percentages=[0.8, 0.1, 0.1],
        **kwargs,
    ):
        """
        Parameters:
        -----------

        Returns:
        --------

        Notes:
        ------
        (1) Calling batch size during __init__ prompts loading adata from .h5ad
        (2) This should eventually be moved to `torch-adata`
        """
        super(BaseLightningAnnDataModule, self).__init__()

        self.save_hyperparameters(ignore=["adata"])
        self._adata = adata
        self.hparams["batch_size"] = self.batch_size

    @property
    def adata(self):
        if not self._adata:
            self._adata = read_h5ad(self.hparams["h5ad_path"], silent=True)
#             print(" - [ NOTE ] | Loading data from .h5ad path.")
        return self._adata

    @property
    def batch_size(self):
        if not self.hparams["batch_size"]:
            return int(self.n_cells / 10)
        return self.hparams["batch_size"]

    @property
    def n_cells(self):
        if hasattr(self, "adata"):
            return self.adata.shape[0]

    @property
    def n_features(self):
        if hasattr(self, "adata"):
            return self.adata.shape[1]

    @abstractmethod
    def prepare_data(self):
        """fetch the data. do any required preprocessing."""
        pass

    @abstractmethod
    def setup(self, stage=None):
        """Setup the data for feeding towards a specific stage"""
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.hparams["num_workers"],
            batch_size=self.hparams["batch_size"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.hparams["num_workers"],
            batch_size=self.hparams["batch_size"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=self.hparams["num_workers"],
            batch_size=self.hparams["batch_size"],
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            num_workers=self.hparams["num_workers"],
            batch_size=self.hparams["batch_size"],
        )

    def __repr__(self):
        return "⚡ {} ⚡".format(font_format("LightningAnnDataModule", ["PURPLE"]))