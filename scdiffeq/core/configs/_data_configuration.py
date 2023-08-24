
# -- import packages: --------------------------------------------------------------------
from torch_adata import LightningAnnDataModule
import pandas as pd
import numpy as np
import anndata
import torch
import os


# -- import local dependencies: ----------------------------------------------------------
from .. import utils
from ... import tools


# -- set typing: -------------------------------------------------------------------------
from typing import Optional, Dict


class TimeConfiguration(utils.ABCParse):
    def __init__(
        self,
        adata: anndata.AnnData,
        t_min: float = 0,
        t_max: float = 1,
        dt: float = 0.1,
        time_key: Optional[str] = None,
        t0_idx: Optional[pd.Index] = None,
        t0_cluster: Optional[str] = None,
        time_cluster_key: Optional[str] = None,
    ):
        super().__init__()
        self.__parse__(kwargs=locals(), public=["adata"])
        
## WE DONT NEED THIS B/C WE CAN GENERATE T0_IDX FROM OTHER KEY:VAL PAIRS
#         self._check_passed_time_args(self._t0_idx, self._time_key)
        
#     def _check_passed_time_args(self, t0_idx, time_key):
#         """If time_key is passed"""
        
#         if sum([utils.not_none(time_key), utils.not_none(t0_idx)]) < 1:
#             raise ValueError("Must provide `t0_idx` and/or `time_key`.")

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

        tools.time_free_sampling(
            adata=self.adata,
            t0_idx=self.t0_idx,
            n_steps=self.n_steps,
            t0_key="t0",
            t_key="t",
        )

    @property
    def has_time_key(self):
        return (not self._time_key is None) and (
            self._time_key in self.adata.obs_keys()
        )

    @property
    def time_key(self):
        if self.has_time_key:
            return self._time_key
        return "t"

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
    t_min: float = 0,
    t_max: float = 1,
    dt: float = 0.1,
    t0_idx: Optional[pd.Index] = None,
    time_key: Optional[str] = None,
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

# # -- main class: -------------------------------------------------------------------------
# class TimeConfiguration(utils.AutoParseBase):
#     def __init__(
#         self,
#         adata: AnnData,
#         time_key: str = None,
#         t0_idx: pd.Index = None,
#         t: torch.Tensor = None,
#         dt: float = None,
#         n_steps: int = 40,
#         t_min: float = 0,
#         t_max: float = 1,
#     ):
#         """
#         Must provide t0_idx or time_key. if both are provided,
#         time_key overrules t0_idx.
#         """
#         self._check_passed_time_args(t0_idx, time_key)
#         self.__parse__(locals(), public=[None])

#     # -- utility funcs: ------------------------------------------------------------------
#     def _check_passed_time_args(self, t0_idx, time_key):
#         """If time_key is passed"""
#         if utils.not_none(t0_idx):
#             time_key = None
#         elif sum([utils.not_none(time_key), utils.not_none(t0_idx)]) < 1:
#             "Must provide t0_idx and/or time_key. If both are provided, t0_idx overrules time_key"
            
    
# #     def _value_pass_check(self, time_key, t0_idx):
# #         assert any(
# #             [not isinstance(time_key, NoneType), not isinstance(t0_idx, NoneType)]
# #         ), "Must provide t0_idx or time_key. If both are provided, time_key overrules t0_idx"

#     def _time_from_adata(self):
#         return torch.Tensor(sorted(self._adata.obs[self.time_key].unique()))

#     # -- time_key: -----------------------------------------------------------------------
#     def _configure_time_key(self):
#         if isinstance(self._time_key, NoneType):
#             try:

#                 time_free_sampling(
#                     self._adata,
#                     self.t0_idx,
#                     n_steps=self.n_steps,
#                     t0_key="t0",
#                     t_key="t",
#                 )
#                 utils.normalize_time(
#                     self._adata,
#                     time_key="t",
#                     t_min=self._t_min,
#                     t_max=self._t_max,
#                 )
#                 self._time_key = "t"

#             except:
#                 raise ValueError("Pass `t0_idx` or `time_key`")

#     @property
#     def time_key(self):
#         self._configure_time_key()
#         return self._time_key

#     # -- t: ------------------------------------------------------------------------------
#     def _configure_t(self):
#         """
#         we are here because we don't have t
#         we can use 0-1 in place of a real time and throw in n_steps
#         """

#         if isinstance(self._t, NoneType):
#             self._t = self._time_from_adata()

#     @property
#     def t(self):
#         self._configure_t()
#         return self._t

#     # -- t0_idx: -------------------------------------------------------------------------
#     def _configure_t0_idx(self):
#         if isinstance(self._t0_idx, NoneType):
#             raise ValueError("Must provide t0_idx")

#     @property
#     def t0_idx(self):
#         self._configure_t0_idx()
#         return self._t0_idx

#     # -- t_min: --------------------------------------------------------------------------
#     def _configure_t_min(self):
#         if not isinstance(self.t, NoneType):
#             self._t_min = min(self.t).item()

#     @property
#     def t_min(self) -> float:
#         self._configure_t_min()
#         return self._t_min

#     # -- t_max: --------------------------------------------------------------------------
#     def _configure_t_max(self):
#         if not isinstance(self.t, NoneType):
#             self._t_max = max(self.t).item()

#     @property
#     def t_max(self) -> float:
#         self._configure_t_max()
#         return self._t_max

#     # -- t_span: -------------------------------------------------------------------------
#     def _configure_t_span(self):
#         if not hasattr(self, "_t_span"):
#             self._t_span = abs(self.t_max - self.t_min)

#     @property
#     def t_span(self):
#         self._configure_t_span()
#         return self._t_span

#     # -- dt: -----------------------------------------------------------------------------
#     def _configure_dt(self):
#         """
#         we are here because we don't have dt.
#         to get dt, de novo, we need t and n_steps.
#         """
#         if isinstance(self._dt, NoneType):
#             self._dt = self.t_span / self._n_steps

#     @property
#     def dt(self):
#         self._configure_dt()
#         return self._dt

#     # -- n_steps: ------------------------------------------------------------------------
#     @property
#     def n_steps(self):
#         return int(self._n_steps + 1)

#     # -- collect attributes: -------------------------------------------------------------

#     def __attributes__(self):

#         self._attrs = {}

#         for attr in self.__dir__():
#             if (not attr == "attributes") and (not attr.startswith("_")):
#                 if attr == "t0_idx":
#                     if self._t0_idx is None:
#                         continue
#                 self._attrs[attr] = getattr(self, attr)

#         return self._attrs

#     @property
#     def attributes(self):
#         return self.__attributes__()
    
# def configure_time(
#     adata: AnnData,
#     time_key: str = None,
#     t0_idx: pd.Index = None,
#     t: torch.Tensor = None,
#     dt: float = None,
#     n_steps: int = 40,
#     t_min: float = 0,
#     t_max: float = 1,
# ) -> dict:

#     """Updates time for AnnData. Returns time_attributes dictionary."""

#     time_config = TimeConfiguration(
#         adata=adata,
#         time_key=time_key,
#         t0_idx=t0_idx,
#         t=t,
#         dt=dt,
#         n_steps=n_steps,
#         t_min=t_min,
#         t_max=t_max,
#     )
#     return time_config.attributes





class LightningData(LightningAnnDataModule, utils.AutoParseBase):
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
        groupby="Time point",  # TODO: make optional
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
    
#     def shuffle_t(self):
        
#         df = self._adata.obs.copy()
#         non_t0 = df.loc[df['t'] != 0]['t']
        
#         shuffled_t = np.zeros(len(df))
#         shuffled_t[non_t0.index.astype(int)] = np.random.choice(non_t0.values, len(non_t0))
#         self._adata.obs["t"] = shuffled_t
    
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



class DataConfiguration(utils.ABCParse):
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
    def _TIME_KWARGS(self):
        return utils.extract_func_kwargs(configure_time, self._PARAMS)

    def _configure_time(self):
        self.t, self.t_config = configure_time(**self._TIME_KWARGS)
        self._scDiffEq_kwargs["groupby"] = self.t_config.attributes["time_key"]
        self._scDiffEq_kwargs.update(self.t_config.attributes)

    @property
    def _LIT_DATA_KWARGS(self):
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

    def __call__(self, adata, scDiffEq, scDiffEq_kwargs, *args, **kwargs):
        
        self.__update__(locals(), ignore=["adata", "scDiffEq"], public=[None])

        self._PARAMS.update(scDiffEq_kwargs)

        self._configure_anndata(adata=adata)
        self._configure_time()
        self._configure_lightning_data()
        self._update_scDiffEq(scDiffEq)
