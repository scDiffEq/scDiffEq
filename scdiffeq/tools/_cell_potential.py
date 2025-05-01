
# -- import packages: ---------------------------------------------------------
import sklearn.preprocessing
import numpy as np
import lightning
import logging
import anndata
import torch
import adata_query
import logging

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -- import local dependencies: -----------------------------------------------
from ..core import utils
from ._knn_smoothing import kNNSmoothing

# from ._fetch import fetch
from ._knn import kNN
import ABCParse

# -- set typing: --------------------------------------------------------------
from typing import Union

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -- operator classes: --------------------------------------------------------
class CellPotentialNormalization(ABCParse.ABCParse):
    """
    Procedure occurs in 5 steps:
    1. Sign flip
    2. Clip outliers via quantile cutoff
    3. kNN Smoothing
    4. Log-transformation
    5. MinMaxScaling

    Notes:
    ------
    1.  Log-transform results in slightly better per-cell
        correlatation w/ CytoTRACE

    """

    def __init__(
        self,
        q: float = 0.05,
        raw_psi_key: str = "_psi",
        kNN_use_key: str = "X_pca",
        knn_smoothing_iters: int = 5,
        use_tqdm: bool = True,
    ) -> None:

        """
        Parameters:
        -----------
        q: quantile cutoff
        """

        self.__parse__(locals(), public=[None])

    @property
    def kNN(self) -> kNN:
        if not hasattr(self, "_graph"):
            self._graph = kNN(self.adata, use_key=self._kNN_use_key)
        return self._graph

    @property
    def _RAW_PSI(self):
        return self.adata.obs[self._raw_psi_key].values

    @property
    def _POSITIVE_RAW_PSI(self):
        """STEP 1: Sign flip"""
        return self._RAW_PSI * -1

    @property
    def _Q_MIN(self):
        return self._q

    @property
    def _Q_MAX(self):
        return 1 - self._q

    @property
    def _Q_MIN_CUTOFF(self):
        return np.quantile(self._POSITIVE_RAW_PSI, self._Q_MIN)

    @property
    def _Q_MAX_CUTOFF(self):
        return np.quantile(self._POSITIVE_RAW_PSI, self._Q_MAX)

    def _clip_psi(self):
        """STEP 2: Clip outlier values using a quantile"""
        logger.info("Quantile-clipping outliers")
        _CLIPPED_PSI = self._POSITIVE_RAW_PSI.copy()
        _CLIPPED_PSI[self._POSITIVE_RAW_PSI < self._Q_MIN_CUTOFF] = self._Q_MIN_CUTOFF
        _CLIPPED_PSI[self._POSITIVE_RAW_PSI > self._Q_MAX_CUTOFF] = self._Q_MAX_CUTOFF
        return _CLIPPED_PSI

    @property
    def _CLIPPED_PSI(self):
        """STEP 2: Clip outlier values using a quantile"""
        if not hasattr(self, "_clipped_psi"):
            self._clipped_psi = self._clip_psi()
        return self._clipped_psi

    def _knn_smoothing(self):
        """STEP 3: kNN smoothing"""
        self.adata.obs["_CLIPPED_PSI"] = self._CLIPPED_PSI
        smoothing = kNNSmoothing(
            self.adata,
            kNN=self.kNN,
            use_key=self._kNN_use_key,
            n_iters=self._knn_smoothing_iters,
            use_tqdm=self._use_tqdm,
        )
        logger.info("kNN smoothing")
        return smoothing(key="_CLIPPED_PSI", add_to_adata=False)

    @property
    def _SMOOTHED_PSI(self):
        """STEP 3: kNN smoothing"""
        if not hasattr(self, "_smoothed_psi"):
            self._smoothed_psi = self._knn_smoothing()
        return self._smoothed_psi

    @property
    def _LOG_PSI(self):
        """STEP 4: Log-transform"""
        logger.info("Log-transforming")
        if not hasattr(self, "_log_psi"):
            self._log_psi = np.log10(self._SMOOTHED_PSI)
        return self._log_psi

    def _min_max_scaling(self):
        """STEP 5: min-max scaling"""
        logger.info("Scaling")
        scaler = sklearn.preprocessing.MinMaxScaler()
        return scaler.fit_transform(self._LOG_PSI.reshape(-1, 1))

    @property
    def _SCALED_PSI(self):
        """STEP 5: min-max scaling"""
        if not hasattr(self, "_scaled_psi"):
            self._scaled_psi = self._min_max_scaling()
        return self._scaled_psi

    def _clean_up_adata(self) -> None:
        self.adata.obs.drop("_CLIPPED_PSI", axis=1, inplace=True)

    def __call__(self, adata, key_added="psi") -> None:
        
        self.__update__(locals(), public = ['adata'])
        adata.obs[key_added] = self._SCALED_PSI
        self._clean_up_adata()


class CellPotential(ABCParse.ABCParse):
    """Calculate [raw] potential values, given cells and a model."""

    def __init__(
        self,
        use_key: str = "X_pca",
        device: Union[str, torch.device] = torch.device("cuda:0"),
        seed: int = 0,
        gpu = True,
    ) -> None:
        
        """ """

        self.__parse__(locals(), public=[None])
        lightning.seed_everything(0)

    @property
    def Z_input(self):
        if not hasattr(self, "_Z_input"):
            self._Z_input = adata_query.fetch(
                adata=self.adata,
                key=self._use_key,
                torch=self._gpu,
                device=self._device,
            )
        return self._Z_input

    def forward(self, model):
        logger.info("Computing")
        return model.DiffEq.DiffEq.mu(self.Z_input).flatten().detach().cpu().numpy()

    def __call__(
        self, adata: anndata.AnnData, model, key_added: str = "_psi"
    ) -> None:

        self.__update__(locals(), public=["adata"])
        self.Z_psi = self.forward(model)
        adata.obs[key_added] = self.Z_psi


# -- API-facing functions: -----------------------------------------------------
def normalize_cell_potential(
    adata,
    use_key="_psi",
    key_added="psi",
    q=0.05,
    kNN_use_key="X_pca",
    knn_smoothing_iters=5,
    use_tqdm=True,
) -> None:
    """Can be AnnData from a simulation or the original AnnData object containing observed cells"""

    cell_potential_norm = CellPotentialNormalization(
        q=q,
        raw_psi_key=use_key,
        kNN_use_key=kNN_use_key,
        knn_smoothing_iters=knn_smoothing_iters,
        use_tqdm=use_tqdm,
    )

    cell_potential_norm(adata, key_added=key_added)


def cell_potential(
    adata: anndata.AnnData,
    model,
    use_key: str = "X_pca",
    raw_key_added: str = "_psi",
    norm_key_added: str = "psi",
    device: Union[str, torch.device] = torch.device("cuda:0"),
    seed: int = 0,
    normalize: bool = True,
    return_raw_array: bool = False,
    q: float = 0.05,
    knn_smoothing_iters: int = 5,
    use_tqdm: bool = True,
    gpu: bool = True,
):
    """ """
    cell_potential = CellPotential(use_key=use_key, device=device, seed=seed, gpu=gpu)
    cell_potential(adata=adata, model=model, key_added=raw_key_added)

    if normalize:
        normalize_cell_potential(
            adata=adata,
            use_key=raw_key_added,
            key_added=norm_key_added,
            q=q,
            kNN_use_key=use_key,
            knn_smoothing_iters=knn_smoothing_iters,
            use_tqdm=use_tqdm,
        )
    if return_raw_array:
        return cell_potential.Z_psi
