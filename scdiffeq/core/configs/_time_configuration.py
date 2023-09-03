
# -- import packages: --------------------------------------------------------------------
import anndata
import pandas as pd
import numpy as np
import torch
import ABCParse


# -- import local dependencies: ----------------------------------------------------------
from .. import utils
from ... import tools


# -- define types: -----------------------------------------------------------------------
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
        print(self._time_key_config._PASSED_VALID, self._time_key_config._DETECTED_VALID)
        print(self._time_key)

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

        if (not isinstance(self._time_cluster_key, NoneType)) and (
            not isinstance(self._t0_cluster, NoneType)
        ):
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

