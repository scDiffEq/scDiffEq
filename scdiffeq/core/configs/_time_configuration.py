
# -- import packages: --------------------------------------------------------------------
from anndata import AnnData
import pandas as pd
import torch


# -- import local dependencies: ----------------------------------------------------------
from .. import utils
from ...tools import time_free_sampling


# -- define types: -----------------------------------------------------------------------
NoneType = type(None)


# -- main class: -------------------------------------------------------------------------
class TimeConfiguration(utils.AutoParseBase):
    def __init__(
        self,
        adata: AnnData,
        time_key: str = None,
        t0_idx: pd.Index = None,
        t: torch.Tensor = None,
        dt: float = None,
        n_steps: int = 40,
        t_min: float = 0,
        t_max: float = 1,
    ):
        """
        Must provide t0_idx or time_key. if both are provided,
        time_key overrules t0_idx.
        """
        self._check_passed_time_args(t0_idx, time_key)
        self.__parse__(locals(), public=[None])

    # -- utility funcs: ------------------------------------------------------------------
    def _check_passed_time_args(self, t0_idx, time_key):
        """If time_key is passed"""
        if utils.not_none(t0_idx):
            time_key = None
        elif sum([utils.not_none(time_key), utils.not_none(t0_idx)]) < 1:
            "Must provide t0_idx or time_key. If both are provided, t0_idx overrules time_key"
            
    
#     def _value_pass_check(self, time_key, t0_idx):
#         assert any(
#             [not isinstance(time_key, NoneType), not isinstance(t0_idx, NoneType)]
#         ), "Must provide t0_idx or time_key. If both are provided, time_key overrules t0_idx"

    def _time_from_adata(self):
        return torch.Tensor(sorted(self._adata.obs[self.time_key].unique()))

    # -- time_key: -----------------------------------------------------------------------
    def _configure_time_key(self):
        if isinstance(self._time_key, NoneType):
            try:

                time_free_sampling(
                    self._adata,
                    self.t0_idx,
                    n_steps=self.n_steps,
                    t0_key="t0",
                    t_key="t",
                )
                utils.normalize_time(
                    self._adata,
                    time_key="t",
                    t_min=self._t_min,
                    t_max=self._t_max,
                )
                self._time_key = "t"

            except:
                raise ValueError("Pass `t0_idx` or `time_key`")

    @property
    def time_key(self):
        self._configure_time_key()
        return self._time_key

    # -- t: ------------------------------------------------------------------------------
    def _configure_t(self):
        """
        we are here because we don't have t
        we can use 0-1 in place of a real time and throw in n_steps
        """

        if isinstance(self._t, NoneType):
            self._t = self._time_from_adata()

    @property
    def t(self):
        self._configure_t()
        return self._t

    # -- t0_idx: -------------------------------------------------------------------------
    def _configure_t0_idx(self):
        if isinstance(self._t0_idx, NoneType):
            raise ValueError("Must provide t0_idx")

    @property
    def t0_idx(self):
        self._configure_t0_idx()
        return self._t0_idx

    # -- t_min: --------------------------------------------------------------------------
    def _configure_t_min(self):
        if not isinstance(self.t, NoneType):
            self._t_min = min(self.t).item()

    @property
    def t_min(self) -> float:
        self._configure_t_min()
        return self._t_min

    # -- t_max: --------------------------------------------------------------------------
    def _configure_t_max(self):
        if not isinstance(self.t, NoneType):
            self._t_max = max(self.t).item()

    @property
    def t_max(self) -> float:
        self._configure_t_max()
        return self._t_max

    # -- t_span: -------------------------------------------------------------------------
    def _configure_t_span(self):
        if not hasattr(self, "_t_span"):
            self._t_span = abs(self.t_max - self.t_min)

    @property
    def t_span(self):
        self._configure_t_span()
        return self._t_span

    # -- dt: -----------------------------------------------------------------------------
    def _configure_dt(self):
        """
        we are here because we don't have dt.
        to get dt, de novo, we need t and n_steps.
        """
        if isinstance(self._dt, NoneType):
            self._dt = self.t_span / self._n_steps

    @property
    def dt(self):
        self._configure_dt()
        return self._dt

    # -- n_steps: ------------------------------------------------------------------------
    @property
    def n_steps(self):
        return int(self._n_steps + 1)

    # -- collect attributes: -------------------------------------------------------------

    def __attributes__(self):

        self._attrs = {}

        for attr in self.__dir__():
            if (not attr == "attributes") and (not attr.startswith("_")):
                if attr == "t0_idx":
                    if self._t0_idx is None:
                        continue
                self._attrs[attr] = getattr(self, attr)

        return self._attrs

    @property
    def attributes(self):
        return self.__attributes__()
    
def configure_time(
    adata: AnnData,
    time_key: str = None,
    t0_idx: pd.Index = None,
    t: torch.Tensor = None,
    dt: float = None,
    n_steps: int = 40,
    t_min: float = 0,
    t_max: float = 1,
) -> dict:

    """Updates time for AnnData. Returns time_attributes dictionary."""

    time_config = TimeConfiguration(
        adata=adata,
        time_key=time_key,
        t0_idx=t0_idx,
        t=t,
        dt=dt,
        n_steps=n_steps,
        t_min=t_min,
        t_max=t_max,
    )
    return time_config.attributes
