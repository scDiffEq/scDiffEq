
# -- import packages: ---------------------------------------------------------
import ABCParse
import anndata


# -- set typing: --------------------------------------------------------------
from typing import Optional


# -- operational class: -------------------------------------------------------
class TimeKeyConfiguration(ABCParse.ABCParse):
    def __init__(self, *args, **kwargs):
        self.__parse__(locals())

    @property
    def _AUTO_DETECTED_TIME_COLS(self):
        return [
            "t",
            "Time point",
            "time",
            "time_pt",
            "t_info",
            "time_info",
        ]

    @property
    def _OBS_COLS(self):
        return self._adata.obs.columns

    @property
    def _PASSED(self) -> bool:
        return hasattr(self, "_time_key")

    @property
    def _PASSED_VALID(self) -> bool:
        if self._PASSED:
            return self._time_key in self._OBS_COLS

    @property
    def _DETECTED(self) -> bool:
        return self._OBS_COLS[
            [col in self._AUTO_DETECTED_TIME_COLS for col in self._OBS_COLS]
        ].tolist()

    @property
    def _DETECTED_VALID(self) -> bool:
        return len(self._DETECTED) == 1

    @property
    def _VALID(self):
        return any([self._PASSED_VALID, self._DETECTED_VALID])

    @property
    def _KeyNotFoundError(self):
        raise KeyError(f"Passed `time_key`: {self._time_key} not found in adata.obs")

    @property
    def _MultipleKeysFoundError(self):
        msg = f"More than one time column inferred: {self._DETECTED}. Specify the desired time column in adata.obs."
        raise KeyError(msg)

    def __call__(
        self,
        adata: anndata.AnnData,
        time_key: Optional[str] = None,
        default_time_key: str = "t",
        *args,
        **kwargs,
    ) -> str:

        self.__update__(locals())

        if self._PASSED:
            if self._PASSED_VALID:
                return self._time_key
            raise self._KeyNotFoundError()

        if self._DETECTED:
            if self._DETECTED_VALID:
                return self._DETECTED[0]
            raise self._MultipleKeysFoundError()

        return self._default_time_key


# -- model-facing function: ---------------------------------------------------
def configure_time_key(
    adata: anndata.AnnData,
    time_key: Optional[str] = None,
) -> str:
    """
    Configure time key

    Parameters
    ----------
    adata: anndata.AnnData

    time_key: Optional[str], default = None

    Returns
    -------
    time_key: str
    """
    time_key_config = TimeKeyConfiguration()
    return time_key_config(adata=adata, time_key=time_key)
