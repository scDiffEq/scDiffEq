
# -- import packages: --------------------------------------------------------------------
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from licorice_font import font_format
from torch_adata import AnnDataset
from typing import Union, List
import numpy as np
import anndata
import inspect
import torch
import os


# -- import local dependencies: ----------------------------------------------------------
from ..utils import function_kwargs, Base


NoneType = type(None)


# -- supporting classes and functions: ---------------------------------------------------
class SplitSize:
    def __init__(self, n_cells: int, n_groups: int):

        self._n_cells = n_cells
        self._n_groups = n_groups

    def _sum_norm(self, vals: Union[List, np.ndarray]) -> np.ndarray:
        return np.array(vals) / vals.sum()

    def uniformly(self):
        div, mod = divmod(self._n_cells, self._n_groups)
        return [div] * (self._n_groups - 1) + [div + mod]

    def proportioned(self, percentages=[0.8, 0.2], remainder_idx=-1):

        percentages = self._sum_norm(np.array(percentages))
        split_lengths = [int(self._n_cells * pct_i) for pct_i in percentages]
        remainder = self._n_cells - sum(split_lengths)
        split_lengths[remainder_idx] += remainder

        return split_lengths


class CellDataManager(Base):
    """Data Manager at the AnnData Level."""

    def __init__(
        self,
        adata,
        use_key="X_pca",
        time_key="Time point",
        t0_key = "t0",
        obs_keys=None,
        train_key="train",
        val_key="val",
        test_key="test",
        predict_key="predict",
        velocity_key="velo",
        n_groups=None,
        train_val_percentages=[0.8, 0.2],
        remainder_idx=-1,
        predict_all=True,
        attr_names={"obs": [], "aux": []},
        one_hot=False,
        aux_keys=None,
        silent=True,
        shuffle=True,
    ):

        self.__config__(locals())

    def __config__(self, kwargs, ignore=["self"]):
        
        if isinstance(kwargs['velocity_key'], NoneType):
            kwargs['aux_keys'] = [kwargs['velocity_key']]
        
        self.AnnDataset_kwargs = function_kwargs(
            func=AnnDataset, kwargs=kwargs, ignore=["adata"]
        )
        
        if not kwargs['time_key']:
            kwargs['time_key'] = "t"

        self.AnnDataset_kwargs['groupby'] = kwargs['time_key']
        self.AnnDataset_kwargs.pop("adata")
        self.__parse__(kwargs, ignore)
        self.df = self.adata.obs.copy()
        self.data_keys = self._get_keys(kwargs)
        
        if not self.n_groups:
            self.n_groups = len(self.train_val_percentages)

        self.split = SplitSize(self.n_cells, self.n_groups)

        # configure train-val split if it has train but not val
        if self.has_train_not_val:
            self.n_fit = self.train_adata.shape[0]
            self._allocate_validation()

    # -- supporting methods: -------------------------------------------------------------
    def _get_keys(self, kwargs):
        return {
            attr.strip("_key"): val
            for attr, val in kwargs.items()
            if attr.endswith("key")
        }

    def _subset_adata(self, key):
        access_key = self.data_keys[key]
        if not hasattr(self.df, access_key):
            if (key == "predict") and (self.predict_all):
                self.df[access_key] = True
            else:
                print("Key: Access Key pair: {}:{} not found".format(key, access_key))
        # else invoke split w/ requisite args
        return self.adata[self.df[access_key]].copy()

    def _set_new_idx(self, df, idx, key_added):

        tmp = np.zeros(len(df), dtype=bool)
        tmp[idx] = True
        df[key_added] = tmp.astype(bool)

    def _allocate_validation(self, remainder_idx=-1):
        """If validation key is not found, invoke this function. Takes the train subset
        adata and breaks it into non-overlapping train and validation adata subsets."""

        if not self.n_groups:
            self.n_groups = len(self.train_val_percentages)

        train_adata = self.train_adata
        n_cells = train_adata.shape[0]

        self.data_keys["train"] = "fit_train"
        self.data_keys["val"] = "fit_val"

        train_val_split = SplitSize(n_cells, self.n_groups)

        if not self.train_val_percentages:
            n_train, n_val = train_val_split.uniformly()
        else:
            n_train, n_val = train_val_split.proportioned(
                percentages=self.train_val_percentages, remainder_idx=remainder_idx
            )

        original_train_idx = train_adata.obs.index
        train_idx = np.random.choice(
            range(len(original_train_idx)), size=n_train, replace=False
        )
        train_cells = np.zeros(len(original_train_idx), dtype=bool)
        train_cells[train_idx] = True
        fit_train_idx = original_train_idx[train_cells]
        fit_val_idx = original_train_idx[~train_cells]

        self._set_new_idx(self.df, idx=fit_train_idx.astype(int), key_added="fit_train")
        self._set_new_idx(self.df, idx=fit_val_idx.astype(int), key_added="fit_val")

        self.adata.obs = self.df
    
    
    # -- key function that transform adata -> torch.utils.data.Dataset: -----------------
    def to_dataset(self, key: str)->torch.utils.data.Dataset:
        """adata -> torch.utils.data.Dataset"""
        adata = getattr(self, "{}_adata".format(key))
        return AnnDataset(adata=adata, **self.AnnDataset_kwargs)

    # -- Properties: ---------------------------------------------------------------------
    @property
    def cell_idx(self):
        return self.adata.obs.index

    @property
    def n_cells(self):
        return self.adata.shape[0]

    @property
    def n_features(self):
        return self.adata.shape[1]

    @property
    def uniform_split(self) -> list([int, ..., int]):
        return self.split.uniformly()

    @property
    def proportioned_split(self) -> list([int, ..., int]):
        return self.split.proportioned(percentages=self.train_val_percentages)

    @property
    def train_adata(self):
        return self._subset_adata("train")

    @property
    def val_adata(self):
        return self._subset_adata("val")

    @property
    def test_adata(self):
        return self._subset_adata("test")

    @property
    def predict_adata(self):
        return self._subset_adata("predict")

    @property
    def has_train_not_val(self):
        return (hasattr(self.df, self.data_keys["train"])) and (
            not hasattr(self.df, self.data_keys["val"])
        )

    @property
    def train_dataset(self):
        return self.to_dataset("train")

    @property
    def val_dataset(self):
        return self.to_dataset("val")

    @property
    def test_dataset(self):
        return self.to_dataset("test")

    @property
    def predict_dataset(self):
        return self.to_dataset("predict")
    
class LightningAnnDataModule(LightningDataModule):
    def __init__(
        self,
        adata: [anndata.AnnData] = None,
        batch_size=2000,
        N=False,
        num_workers=os.cpu_count(),
        use_key="X_pca",
        time_key="Time point",
        obs_keys=['W'],
        train_key="train",
        val_key="val",
        test_key="test",
        predict_key="predict",
        n_groups=None,
        train_val_percentages=[0.8, 0.2],
        remainder_idx=-1,
        predict_all=True,
        attr_names={"obs": [], "aux": []},
        one_hot=False,
        aux_keys=None,
        silent=True,
    ):
        super(LightningAnnDataModule, self).__init__()
        self.save_hyperparameters(ignore=["adata"])
        self.cell_data_manager_kwargs = function_kwargs(
            func=CellDataManager, kwargs=locals(), ignore=["adata"]
        )
        self.cell_data_manager_kwargs.pop("adata")
        # TODO: seems like the above "ignore" arg isn't working...
        self._adata = adata
        
        for key in self.cell_data_manager_kwargs['obs_keys']:
            if not hasattr(self._adata.obs, key):
                self._adata.obs[key] = 1        

    # -- Supporting methods --------------------------------------------------------------
    def _return_loader(self, dataset_key):
        if dataset_key == "test":
            if not hasattr(self, "n_test_cells"):
                self.setup(stage="test")
            batch_size = self.n_test_cells
        else:
            batch_size = self.hparams["batch_size"]
            
        return DataLoader(getattr(self, "{}_dataset".format(dataset_key)),
                          num_workers=self.hparams["num_workers"],
                          batch_size=batch_size,
                          shuffle=self.hparams["batch_size"],
        )

    # -- Properties: ---------------------------------------------------------------------
    @property
    def adata(self):

        if isinstance(self._adata, anndata.AnnData):
            return self._adata
        elif isinstance(self.hparams["h5ad_path"], str):
            return anndata.read_h5ad(self.hparams["h5ad_path"])
        print("Pass adata or h5ad_path")

    @property
    def n_cells(self):
        return self.adata.shape[0]

    @property
    def n_features(self):
        return self.adata.shape[1]

    @property
    def n_dims(self):
        return self.adata.obsm[self.hparams["use_key"]].shape[1]

    @property
    def batch_size(self):
        if not self.hparams["batch_size"]:
            return int(self.n_cells / 10)
        return self.hparams["batch_size"]

    @property
    def allocated_val(self):
        return self.hparams['train_val_percentages'][1] > 0
    
    # -- Standard methods: ---------------------------------------------------------------
    def prepare_data(self):
        self.data = CellDataManager(self.adata, **self.cell_data_manager_kwargs)

    def setup(self, stage=None):

        if stage in ["fit", "train", "val"]:
            if self.allocated_val:                
                self.val_dataset = self.data.val_dataset
            self.train_dataset = self.data.train_dataset

        elif stage == "test":
            self.test_dataset = self.data.test_dataset
            self.n_test_cells = len(self.test_dataset)
        elif stage in [None, "predict"]:
            self.predict_dataset = self.data.predict_dataset
        else:
            print(
                "CURRENT STAGE: {} - no suitable subset found during `LightningDataModule.setup()`".format(
                    stage
                )
            )

    # -- Required DataLoader methods: ----------------------------------------------------
    def train_dataloader(self):
        return self._return_loader("train")

    def val_dataloader(self):
        return self._return_loader("val")

    def test_dataloader(self):
        return self._return_loader("test")

    def predict_dataloader(self):
        return self._return_loader("predict")

    def __repr__(self):
        return "⚡ {} ⚡".format(font_format("LightningAnnDataModule", ["PURPLE"]))
    
class LightningDataModuleConfig:
    def __init__(
        self,
        adata: [anndata.AnnData] = None,
        batch_size=2000,
        N=False,
        num_workers=os.cpu_count(),
        use_key="X_pca",
        time_key="Time point",
        obs_keys=['W'],
        train_key="train",
        val_key="val",
        test_key="test",
        predict_key="predict",
        n_groups=None,
        train_val_percentages=[0.8, 0.2],
        remainder_idx=-1,
        predict_all=True,
        attr_names={"obs": [], "aux": []},
        one_hot=False,
        aux_keys=None,
        silent=True,
    ):
        kwargs = function_kwargs(LightningAnnDataModule, locals())
        self._LightningDataModule = LightningAnnDataModule(**kwargs)
        
    @property
    def LightningDataModule(self):
        return self._LightningDataModule
