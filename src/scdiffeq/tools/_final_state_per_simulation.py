

# import scdiffeq_plots as sdq_pl
import numpy as np
import ABCParse

from ..core import utils
        
class FinalStatePerSimulation(ABCParse.ABCParse):
    """"""

    def __init__(
        self,
        adata,
        obs_key="Cell type annotation",
        N_sim=2000,
        time_key="t",
        sim_key="simulation",
    ):
        self.__parse__(locals(), public=["adata"])
        self._configure_simulation_key()

    @property
    def _MAX_TIME(self):
        return self.adata.obs[self._time_key].max()

    @property
    def _N_TIMEPOINTS(self):
        return self.adata.obs[self._time_key].nunique()

    @property
    def _SIM_COL(self):
        if not hasattr(self, "_sim_col"):
            self._sim_col = np.tile(np.arange(self._N_sim), self._N_TIMEPOINTS)
        return self._sim_col

    @property
    def _SIMULATION_GROUPED(self):
        if not hasattr(self, "_sim_grouped"):
            self._sim_grouped = self.adata.obs.groupby(self._sim_key)
        return self._sim_grouped

    def _configure_simulation_key(self):
        if not self._sim_key in self.adata.obs:
            self.adata.obs[self._sim_key] = self._SIM_COL

    def _final_annot_single_sim(self, df):
        return df.loc[df[self._time_key] == self._MAX_TIME][self._obs_key].values[0]

    @property
    def GROUPED_BY_FINAL_STATE(self):
        return self.adata.obs.groupby(self.key_added)

    def annotate_final_state(self, key_added="final_state"):
        self.key_added = key_added
        STATE = self._SIMULATION_GROUPED.apply(self._final_annot_single_sim)
        self.adata.obs[key_added] = self.adata.obs[self._sim_key].map(STATE)

        return self.GROUPED_BY_FINAL_STATE
