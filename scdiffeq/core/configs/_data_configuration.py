
# -- import packages: --------------------------------------------------------------------
from torch_adata import LightningAnnDataModule
import pandas as pd
import numpy as np
import anndata
import torch
import os
import ABCParse


# -- import local dependencies: ----------------------------------------------------------
from .. import utils
from ... import tools


# -- set typing: -------------------------------------------------------------------------
from typing import Optional, Dict


class TimeKey:
    def __init__(
        self,
        auto_detected_time_cols=[
            "t",
            "Time point",
            "time",
            "time_pt",
            "t_info",
            "time_info",
        ],
    ):
        self._auto_detected_time_cols = auto_detected_time_cols

    @property
    def _OBS_COLS(self):
        return self._adata.obs.columns

    @property
    def _PASSED(self) -> bool:
        return not (self._time_key is None)

    @property
    def _PASSED_VALID(self) -> bool:
        return self._time_key in self._OBS_COLS

    @property
    def _DETECTED(self) -> bool:
        return self._OBS_COLS[
            [col in self._auto_detected_time_cols for col in self._OBS_COLS]
        ].tolist()

    @property
    def _DETECTED_VALID(self) -> bool:
        return len(self._DETECTED) == 1

    @property
    def _VALID(self):
        return any([self._PASSED_VALID, self._DETECTED_VALID])

    def __call__(self, adata: anndata.AnnData, time_key: Optional[str] = None) -> str:
        
        self._time_key = time_key
        self._adata = adata

        if self._PASSED:
            if self._PASSED_VALID:
                return self._time_key
            else:
                raise KeyError(
                    f"Passed `time_key`: {self._time_key} not found in adata.obs"
                )

        elif self._DETECTED:
            if self._DETECTED_VALID:
                return self._DETECTED[0]
            else:
                print(
                    f"More than one possible time column inferred: {found}\nPlease specify the desired time column in adata.obs."
                )
        return "t"

class TimeConfiguration(ABCParse.ABCParse):
    def __init__(
        self,
        adata: anndata.AnnData,
        time_key: Optional[str] = None,
        t_min: float = 0,
        t_max: float = 1,
        dt: float = 0.1,
        t0_idx: Optional[pd.Index] = None,
        t0_cluster: Optional[str] = None,
        time_cluster_key: Optional[str] = None,
    ):
        super().__init__()
        self.__parse__(kwargs=locals(), public=["adata"])
        
        self._time_key_config = TimeKey()
        self._time_key = self._time_key_config(adata = self.adata, time_key = self._time_key)
#         print(self._time_key_config._PASSED_VALID, self._time_key_config._DETECTED_VALID)
#         print(self._time_key)

    @property
    def t0_idx(self):

        """
        If t0_idx is provided, returns that.
        If t0_idx is not provided, but a cluster key and t0 cluster
        value-pair is provided, this can isolate the t0_idx.
        """

        if self.has_time_key:
            
            return self.adata.obs[
                self.adata.obs[self._time_key] == self.adata.obs[self._time_key].min()
            ].index

        if not self._t0_idx is None:
            return self._t0_idx

        if (not self._time_cluster_key is None) and (not self._t0_cluster is None):
            
            return self.adata.obs[
                self.adata.obs[self._time_cluster_key] == self._t0_cluster
            ].index

    def time_sampling(self):
        if not self.has_time_key:
            self.adata.obs["t0"] = self.adata.obs.index.isin(self.t0_idx)
#         else:
        tools.time_free_sampling(
            adata=self.adata,
            t0_idx=self.t0_idx,
            n_steps=self.n_steps,
            t0_key="t0",
            t_key="t",
        )

    @property
    def has_time_key(self):
        return self._time_key_config._VALID
        
#         return (not isinstance(self._time_key, NoneType)) and (
#             self._time_key in self.adata.obs_keys()
#         )

    @property
    def time_key(self):
        return self._time_key

#         time_key = time_key_id(adata, time_key=self._time_key)
#         if self.has_time_key:
#             return self._time_key
#         return "t"

    @property
    def t_min(self):
        if self.has_time_key:
            self._t_min = self.adata.obs[self._time_key].min()
        return self._t_min

    @property
    def t_max(self):
        if self.has_time_key:
            self._t_max = self.adata.obs[self._time_key].max()
        return self._t_max

    @property
    def t_diff(self):
        return self._t_max - self._t_min

    def _IS_CONTIGUOUS(self, n_steps):
        return n_steps % 1 == 0

    @property
    def n_steps(self):
        _n_steps = self.t_diff / self._dt + 1
        if self._IS_CONTIGUOUS(_n_steps):
            return int(_n_steps)
        raise ValueError("Time is non-contiguous")

    @property
    def dt(self):
        return self._dt

    def _t_from_time_key(self):
        return torch.Tensor(sorted(self.adata.obs[self._time_key].unique()))

    def _t_from_bounds(self):
        return torch.linspace(self._t_min, self._t_max, self.n_steps)

    def __call__(self):

        if self.has_time_key:
            return self._t_from_time_key()
        self.time_sampling()
        return self._t_from_bounds()

    @property
    def attributes(self):
        self._ATTR_KEYS = sorted(
            ["time_key", "t0_idx", "t_min", "t_max", "t_diff", "dt", "n_steps"]
        )
        return {attr: getattr(self, attr) for attr in self._ATTR_KEYS}

    def __repr__(self):

        header = "Time Attributes:\n\n  "

        _attrs = [
            "{:<9} :  {}".format(attr, val) for attr, val in self.attributes.items()
        ]
        return header + "\n  ".join(_attrs)


def configure_time(
    adata: anndata.AnnData,
    time_key: str = None,
    t_min: float = 0,
    t_max: float = 1,
    t0_idx: Optional[pd.Index] = None,
    dt: float = 0.1,
    t0_cluster: Optional[str] = None,
    time_cluster_key: Optional[str] = None,
) -> Dict:

    """Updates time for AnnData. Returns time_attributes dictionary."""

    time_config = TimeConfiguration(
        adata=adata,
        time_key=time_key,
        t_min=t_min,
        t_max=t_max,
        t0_idx=t0_idx,
        dt=dt,
        t0_cluster=t0_cluster,
        time_cluster_key=time_cluster_key,
    )
    
    t = time_config()
    
    return t, time_config


class LightningData(LightningAnnDataModule, ABCParse.ABCParse):
    def __init__(
        self,
        adata=None,
        h5ad_path=None,
        batch_size=2000,
        num_workers=os.cpu_count(),
        train_val_split=[0.8, 0.2],
        use_key="X_pca",
        obs_keys=[],
        weight_key='W',
        groupby: Optional[str] = None,
        train_key="train",
        val_key="val",
        test_key="test",
        predict_key="predict",
        silent=True,
        shuffle_time_labels = False,
        **kwargs,
    ):
        super(LightningData, self).__init__()
        
                    
        self.__parse__(locals(), public=[None])
        self._format_sinkhorn_weight_key()
        self._format_train_test_exposed_data()
        self.configure_train_val_split()
        

    @property
    def n_dim(self):
        if not hasattr(self, "_n_dim"):
            self._n_dim = self.train_dataset.X.shape[-1]
        return self._n_dim

    
    def _format_sinkhorn_weight_key(self):
        if not self._weight_key in self._adata.obs.columns:
            self._adata.obs[self._weight_key] = 1
        self._obs_keys.append(self._weight_key)
    
    def _format_train_test_exposed_data(self):
        
        if not self._train_key in self._adata.obs.columns:
            self._adata.obs[self._train_key] = True
        if not self._test_key in self._adata.obs.columns:
            self._adata.obs[self._test_key] = False
            
    def prepare_data(self):
        ...

    def setup(self, stage=None):
        ...


class DataConfiguration(ABCParse.ABCParse):
    """Should be called during config steps of scDiffEq model instantiation."""

    def __init__(self, *args, **kwargs):
        self.__parse__(locals(), public=[None])

    def _configure_anndata(self, adata):
        """
        Copy the object so as not to modify the input object, inplace.
        Stash input indices incase they are changed. Ensure that the obs
        index are str type.
        """

        # so as not to modify the input object, inplace.
        self.adata = adata.copy()

        # stash these in case they're needed later
        self._INPUT_OBS_IDX = self.adata.obs.index
        self._INPUT_VAR_IDX = self.adata.var.index
        # make sure index is str
        if not isinstance(adata.obs.index[0], str):
            self.adata = utils.idx_to_int_str(self.adata)
        self._PARAMS["adata"] = self.adata

    @property
    def _TIME_KWARGS(self) -> Dict:
        return utils.extract_func_kwargs(configure_time, self._PARAMS)

    def _configure_time(self) -> None:
        self.t, self.t_config = configure_time(**self._TIME_KWARGS)
        self._scDiffEq_kwargs["groupby"] = self.t_config.attributes["time_key"]
        self._scDiffEq_kwargs.update(self.t_config.attributes)

    @property
    def _LIT_DATA_KWARGS(self) -> Dict:
        self._PARAMS.update(self._scDiffEq_kwargs)
        return utils.extract_func_kwargs(func=LightningData, kwargs=self._PARAMS)

    def _configure_lightning_data(self):
        self.LitDataModule = LightningData(**self._LIT_DATA_KWARGS)
        self._scDiffEq_kwargs["data_dim"] = self.LitDataModule.n_dim

    def _update_scDiffEq(self, scDiffEq):
        scDiffEq._PARAMS.update(self._scDiffEq_kwargs)

        scDiffEq.t = self.t
        scDiffEq.t_config = self.t_config
        scDiffEq.LitDataModule = self.LitDataModule
        scDiffEq.adata = self.adata
        scDiffEq._data_dim = self._scDiffEq_kwargs["data_dim"]

    def __call__(
        self,
        adata: anndata.AnnData,
        scDiffEq,
        scDiffEq_kwargs: Dict,
        *args,
        **kwargs,
    ) -> None:
        
        """
        Data configuration occurs in four steps:
        (1) input adata is configured
        (2) Time is configured
        (3) Lightning Data is configured
        (4) scDiffEq model is updated with new attributes.
        
        Parameters
        ----------
        adata: anndata.AnnData
        
        scDiffEq
        
        scDiffEq_kwargs: Dict
        
        Returns
        -------
        None, updates scDiffEq model.
        """
        
        self.__update__(locals(), ignore=["adata", "scDiffEq"], public=[None])

        self._PARAMS.update(scDiffEq_kwargs)

        self._configure_anndata(adata=adata)
        self._configure_time()
        self._configure_lightning_data()
        self._update_scDiffEq(scDiffEq)
