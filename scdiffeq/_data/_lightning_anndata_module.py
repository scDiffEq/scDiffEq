
# -- import packages: --------------------------------------------------------------------
from abc import abstractmethod
import os


from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from licorice_font import font_format
import torch_adata
import anndata
import torch
import os


# -- import local dependencies: ----------------------------------------------------------
from .._io._read_h5ad import read_h5ad
from .._core.utils import extract_func_kwargs



# -- main module: ------------------------------------------------------------------------
class LightningAnnDataModule(LightningDataModule):
    
    def __init__(
        self,
        Dataset: torch.utils.data.Dataset = None,
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
        silent: bool = True,
        percentages=[0.8, 0.2], # [0.8, 0.1, 0.1],
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
        super(LightningAnnDataModule, self).__init__()

        self.save_hyperparameters(ignore=["adata", "Dataset"])
        self._adata = adata
        self._Dataset = Dataset
        self.AnnDataset_kwargs = extract_func_kwargs(func=torch_adata.AnnDataset,
                                                     kwargs=locals(),
                                                     ignore=['adata']
                                                    )

    def _augment_obs_keys(
        self, weight_key="W", velocity_key="V", fate_key="F", other_keys=[]
    ):
        obs_cols = self._adata.obs.columns.tolist()
        obs_keys = [weight_key, velocity_key, fate_key] + other_keys

        for key in obs_keys:
            if not key in obs_cols:
                self._adata.obs[key] = 1
                
    def prepare_data(self):
        """fetch the data. do any required preprocessing."""
        self.dataset = self.Dataset
        
    def setup(self, stage=None):
        """Setup the data for feeding towards a specific stage"""
        if stage == "test":
            self.test_dataset = self.dataset
        elif stage == "predict":
            self.predict_dataset = self.dataset
        else:
            self.train_dataset, self.val_dataset = torch_adata.tl.split(
                self.dataset, percentages=self.hparams['percentages']
            )
            
    @property
    def Dataset(self):
        if not isinstance(self._Dataset, torch.utils.data.Dataset):
            self._augment_obs_keys()
            self._Dataset = torch_adata.AnnDataset(self.adata, **self.AnnDataset_kwargs)
        return self._Dataset

    @property
    def adata(self):
        if not self._adata:
            self._adata = read_h5ad(self.hparams["h5ad_path"], silent=True)
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
        
    @property
    def n_dims(self):
        if hasattr(self, "adata"):
            return self.adata.obsm[self.hparams['use_key']].shape[1]

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
