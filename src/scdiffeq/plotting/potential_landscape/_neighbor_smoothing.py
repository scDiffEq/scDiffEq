# -- import packages: ----------------------------------------------------------
import sklearn
import pandas as pd
import numpy as np
import ABCParse


from ...core import utils

# -- set typing: ---------------------------------------------------------------
NoneType = type(None)



# -- Operational class: --------------------------------------------------------
class NeighborSmoothing(ABCParse.ABCParse):
    _FIT = False

    def __init__(
        self, mode: [NoneType, str] = "radius", radius: int = 1, neighbors: int = 15
    ):
        """
        Parameters:
        -----------
        mode
            type: [NoneType, str]
            default: "radius"

        radius
            type: int
            default: 1

        neighbors
            type: int
            default: 15

        Returns:
        --------
        None

        Notes:
        ------

        """

        self.__parse__(locals(), public=[None])
        self._AVAILABLE_MODELS = {
            "radius": sklearn.neighbors.RadiusNeighborsRegressor,
            "neighbors": sklearn.neighbors.KNeighborsRegressor,
        }

        self._AVAILABLE_MODEL_ARGS = {
            "radius": self._radius,
            "neighbors": self._neighbors,
        }

    def _configure_neighbor_model(self):
        """
        Sets the model and the supplied argument(s).
        """
        arg = self._AVAILABLE_MODEL_ARGS[self._mode]
        neighbor_model = self._AVAILABLE_MODELS[self._mode]
        return neighbor_model(arg)

    @property
    def NEIGHBOR_MODEL(self):
        if not hasattr(self, "_neighbor_model"):
            self._neighbor_model = self._configure_neighbor_model()
        return self._neighbor_model

    @property
    def _DO_SMOOTHING(self):
        return not isinstance(self._mode, NoneType)

    def fit(self, X_umap: np.ndarray, value: np.ndarray) -> None:
        self.NEIGHBOR_MODEL.fit(X_umap, value)
        self._FIT = True

    def predict(self, X_umap: np.ndarray) -> np.ndarray:
        return self.NEIGHBOR_MODEL.predict(X_umap)