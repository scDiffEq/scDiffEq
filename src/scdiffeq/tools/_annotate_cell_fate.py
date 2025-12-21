# -- import packages: ---------------------------------------------------------
import anndata
import ABCParse
import logging
import numpy as np
import pandas as pd
import logging

# -- configure logging: --------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -- controlling class: -------------------------------------------------------
class CellFateAnnotation(ABCParse.ABCParse):
    """ """

    def __init__(
        self,
        time_key: str = "t",
        sim_key: str = "sim",
        silent: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self.__parse__(locals())

    @property
    def _TIME(self) -> pd.Series:
        return self._adata_sim.obs[self._time_key]

    @property
    def _T_MAX(self) -> float:
        """Max time observed."""
        return self._TIME.max()

    @property
    def _FATE_VALUES(self) -> np.ndarray:
        return self._adata_sim.obs.loc[self._TIME == self._T_MAX][
            [self._sim_key, self._state_key]
        ].values

    def forward(self) -> None:
        mappable_fate_dict = {row[0]: row[1] for row in self._FATE_VALUES}
        self._adata_sim.obs[self._key_added] = self._adata_sim.obs[self._sim_key].map(
            mappable_fate_dict
        )
        if not self._silent:
            logger.info(f"Added fate annotation: adata_sim.obs['{self._key_added}']")
        self._count_fates()

    def _count_fates(self) -> None:
        """ """
        fate_obs = self._adata_sim.obs.loc[self._TIME == self._T_MAX]
        self._adata_sim.uns["fate_counts"] = (
            fate_obs[self._key_added].value_counts().to_dict()
        )

        if not self._silent:
            logger.info("Added fate counts: adata_sim.uns['fate_counts']")

    def __call__(
        self,
        adata_sim: anndata.AnnData,
        state_key: str,
        key_added: str = "fate",
        *args,
        **kwargs,
    ) -> None:
        self.__update__(locals())

        self.forward()


# -- API-facing function: -----------------------------------------------------
def annotate_cell_fate(
    adata_sim: anndata.AnnData,
    state_key: str = "state",
    key_added: str = "fate",
    time_key: str = "t",
    sim_key: str = "sim",
    silent: bool = False,
    *args,
    **kwargs,
) -> None:
    """Annotate the fate of a simulated cell trajectory, given annotated states.

    Parameters
    ----------
    adata_sim : anndata.AnnData
        Simulated AnnData object.

        state_key (str)
            Key in ``adata_sim.obs`` corresponding to annotated states.

        key_added (str)
            Key added to ``adata_sim.obs`` to indicate the fate of each trajectory.
            **Default**: "fate"

        time_key (str)
            Key in ``adata_sim.obs`` corresponding to time.

        sim_key (str)
            Key in ``adata_sim.obs`` corresponding to each individual simulation.

        silent (bool)
            If ``True``, suppresses informational messages.

    Returns
    -------
    None
    """

    fate_annot = CellFateAnnotation(time_key=time_key, sim_key=sim_key, silent=silent)
    fate_annot(adata_sim=adata_sim, state_key=state_key, key_added=key_added)
