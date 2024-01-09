
import ABCParse
import pathlib
import pandas as pd

from ._hparams import HParams
from ._checkpoint import Checkpoint

from typing import Dict, List, Union

class Version(ABCParse.ABCParse):
    def __init__(self, path: Union[pathlib.Path, str] = None, *args, **kwargs):

        self.__parse__(locals())

    @property
    def _PATH(self):
        if isinstance(self._path, pathlib.Path):
            return self._path
        if isinstance(self._path, str):
            self._path = pathlib.Path(self._path)
        return self._path
    
    @property
    def _NAME(self) -> str:
        return self._PATH.name

    @property
    def _CONTENTS(self):
        return list(self._PATH.glob("*"))

    @property
    def hparams(self):
        
        hparams_path = self._PATH.joinpath("hparams.yaml")
        if hparams_path.exists():
            return HParams(hparams_path)

    @property
    def metrics_df(self) -> pd.DataFrame:
        """ """
        metrics_path = self._PATH.joinpath("metrics.csv")
        if metrics_path.exists():
            return pd.read_csv(metrics_path)

    @property
    def _CKPT_PATHS(self) -> List[pathlib.Path]:
        """ """
        _ckpt_paths = list(self._PATH.joinpath("checkpoints").glob("*"))
        return [pathlib.Path(path) for path in _ckpt_paths]

    @property
    def _SORTED_CKPT_KEYS(self) -> List:
        """ """
        epochs = list(self.ckpts.keys())
        _epochs = sorted([epoch for epoch in epochs if epoch != "last"])
        if "last" in epochs:
            _epochs.append("last")
        return _epochs

    @property
    def ckpts(self) -> Dict:
        """ """
        if not hasattr(self, "_CHECKPOINTS"):
            self._CHECKPOINTS = {}
            for ckpt_path in self._CKPT_PATHS:
                ckpt = Checkpoint(ckpt_path)
                self._CHECKPOINTS[ckpt.epoch] = ckpt
        return self._CHECKPOINTS

    def __repr__(self) -> str:
        return self._NAME