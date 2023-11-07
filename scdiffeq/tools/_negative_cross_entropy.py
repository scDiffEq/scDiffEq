

from ..core import utils
import ABCParse
import numpy as np
import pandas as pd


class NegativeCrossEntropy(ABCParse.ABCParse):
    def __init__(self, epsilon=1e-7):

        self.__parse__(locals(), public=[None])

    def _clip_predictions(self, pred):
        """Given a defined epsilon parameter, clip predictions"""
        return np.clip(pred, self._epsilon, 1 - self._epsilon)

    def _log_transformed_predictions(self, obs, pred_clipped):
        return obs * np.log(pred_clipped)

    def _summing(self, pred_logged):
        return -np.sum(np.sum(pred_logged, axis=1))

    def __call__(self, obs: pd.DataFrame, pred: pd.DataFrame):
        pred_clipped = self._clip_predictions(pred)
        pred_logged = self._log_transformed_predictions(obs, pred_clipped)
        return self._summing(pred_logged)

    def __repr__(self):
        return """NegativeCrossEntropy()"""