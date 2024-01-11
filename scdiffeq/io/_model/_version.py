
# -- import packages: ---------------------------------------------------------
import ABCParse
import pathlib
import pandas as pd


# -- import local dependencies: -----------------------------------------------
from ._hparams import HParams
from ._checkpoint import Checkpoint
from ._grouped_metrics import GroupedMetrics


# -- set typing: --------------------------------------------------------------
from typing import Dict, List, Union


# -- operational class: -------------------------------------------------------
class Version(ABCParse.ABCParse):
    """scDiffEq Version object container"""

    def __init__(self, path: Union[pathlib.Path, str] = None, groupby: str = "epoch", *args, **kwargs):
        """Instantiate Version by providing a path

        Args:
            path (Union[pathlib.Path]): path to the version within an
            scDiffEq project. **Default** = None.


        Returns:
            None
        """
        self.__parse__(locals())

    @property
    def _PATH(self) -> pathlib.Path:
        """check and format the provided path"""
        if isinstance(self._path, pathlib.Path):
            return self._path
        if isinstance(self._path, str):
            self._path = pathlib.Path(self._path)
        return self._path

    @property
    def _NAME(self) -> str:
        """version name from provided path"""
        return self._PATH.name

    @property
    def _CONTENTS(self) -> List[pathlib.Path]:
        """return one-level glob of the provided path"""
        return list(self._PATH.glob("*"))

    @property
    def hparams(self):
        """check if the .yaml exists and instantiate the HParams cls each time"""
        hparams_path = self._PATH.joinpath("hparams.yaml")
        if hparams_path.exists():
            return HParams(hparams_path)

    @property
    def metrics_df(self) -> pd.DataFrame:
        """check if metrics.csv path exists as given and read. reads new
        each time"""
        metrics_path = self._PATH.joinpath("metrics.csv")
        if metrics_path.exists():
            return pd.read_csv(metrics_path)
        
    @property
    def per_epoch_metrics(self):
        self._GROUPED_METRICS = GroupedMetrics(groupby = self._groupby)
        return self._GROUPED_METRICS(self.metrics_df)

    @property
    def _CKPT_PATHS(self) -> List[pathlib.Path]:
        """formatted ckpt paths"""
        _ckpt_paths = list(self._PATH.joinpath("checkpoints").glob("*"))
        return [pathlib.Path(path) for path in _ckpt_paths]

    @property
    def _SORTED_CKPT_KEYS(self) -> List:
        """sorting for organization's sake"""
        epochs = list(self.ckpts.keys())
        _epochs = sorted([epoch for epoch in epochs if epoch != "last"])
        if "last" in epochs:
            _epochs.append("last")
        return _epochs

    @property
    def ckpts(self) -> Dict:
        """format and update available checkpoints for the version"""
        if not hasattr(self, "_CHECKPOINTS"):
            self._CHECKPOINTS = {}
            for ckpt_path in self._CKPT_PATHS:
                ckpt = Checkpoint(ckpt_path)
                self._CHECKPOINTS[ckpt.epoch] = ckpt
        return self._CHECKPOINTS

    def __repr__(self) -> str:
        """return the name of the obj"""
        return self._NAME
