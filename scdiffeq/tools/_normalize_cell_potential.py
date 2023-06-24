
from ..core import utils
from ._knn_smoothing import kNNSmoothing
from ._knn import kNN

import numpy as np
import sklearn.preprocessing

class CellPotentialNormalization(utils.ABCParse):

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
    ):

        """
        Parameters:
        -----------
        q: quantile cutoff
        """

        self.__parse__(locals(), public=[None])
        self._INFO = utils.InfoMessage()

    @property
    def kNN(self):
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
            n_iters=self._knn_smoothing_iters,
            use_tqdm=self._use_tqdm,
        )
        self._INFO("kNN smoothing...")
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
        if not hasattr(self, "_log_psi"):
            self._log_psi = np.log10(self._SMOOTHED_PSI)
        return self._log_psi

    def _min_max_scaling(self):
        """STEP 5: min-max scaling"""
        scaler = sklearn.preprocessing.MinMaxScaler()
        return scaler.fit_transform(self._LOG_PSI.reshape(-1, 1))

    @property
    def _SCALED_PSI(self):
        """STEP 5: min-max scaling"""
        if not hasattr(self, "_scaled_psi"):
            self._scaled_psi = self._min_max_scaling()
        return self._scaled_psi

    def _clean_up_adata(self):
        self.adata.obs.drop("_CLIPPED_PSI", axis=1, inplace=True)

    def __call__(self, adata, key_added="psi"):
        self.__update__(locals())

        adata.obs[key_added] = self._SCALED_PSI
        self._clean_up_adata()


def normalize_cell_potential(
    adata,
    use_key="_psi",
    key_added="psi",
    q=0.05,
    kNN_use_key="X_pca",
    knn_smoothing_iters=5,
    use_tqdm=True,
):
    """Can be AnnData from a simulation or the original AnnData object containing observed cells"""
    
    cell_potential_norm = CellPotentialNormalization(
        q=q,
        raw_psi_key=use_key,
        kNN_use_key=kNN_use_key,
        knn_smoothing_iters=knn_smoothing_iters,
        use_tqdm=use_tqdm,
    )

    cell_potential_norm(adata, key_added=key_added)
