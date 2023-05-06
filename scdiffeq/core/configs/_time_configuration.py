
# -- import packages: --------------------------------------------------------------------
import anndata
import pandas as pd
import numpy as np
import torch


# -- import local dependencies: ----------------------------------------------------------
from .. import utils
from ... import tools


# -- define types: -----------------------------------------------------------------------
NoneType = type(None)


class TimeConfiguration(utils.ABCParse):
    def __init__(
        self,
        adata: anndata.AnnData,
        time_key: str = None,
        t_min=0,
        t_max=1,
        dt=0.1,
        t0_idx=None,
        t0_cluster=None,
        cluster_key=None,
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

        if not isinstance(self._t0_idx, NoneType):
            return self._t0_idx

        if (not isinstance(self._cluster_key, NoneType)) and (
            not isinstance(self._t0_cluster, NoneType)
        ):
            return self.adata.obs[
                self.adata.obs[self._cluster_key] == self._t0_cluster
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
        return (not isinstance(self._time_key, NoneType)) and (
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
    time_key: str = None,
    t_min: float = 0,
    t_max: float = 1,
    t0_idx: pd.Index = None,
    dt: float = 0.1,
    t0_cluster=None,
    cluster_key=None,
) -> dict:

    """Updates time for AnnData. Returns time_attributes dictionary."""

    time_config = TimeConfiguration(
        adata=adata,
        time_key=time_key,
        t_min=t_min,
        t_max=t_max,
        t0_idx=t0_idx,
        dt=dt,
        t0_cluster=t0_cluster,
        cluster_key=cluster_key,
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
