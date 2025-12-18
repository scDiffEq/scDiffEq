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
    """scDiffEq Version object container

    Attributes
    ----------
    _path : Union[pathlib.Path, str]
        Path to the version within an scDiffEq project.
    _groupby : str
        Grouping method for metrics, default is "epoch".
    """

    def __init__(
        self,
        path: Union[pathlib.Path, str] = None,
        groupby: str = "epoch",
        *args,
        **kwargs
    ):
        """Instantiate Version by providing a path

        Parameters
        ----------
        path : Union[pathlib.Path, str], optional
            Path to the version within an scDiffEq project, by default None.
        groupby : str, optional
            Grouping method for metrics, by default "epoch"

        Returns
        -------
        None
        """
        self.__parse__(locals())

    @property
    def _PATH(self) -> pathlib.Path:
        """Check and format the provided path

        Returns
        -------
        pathlib.Path
            Formatted path

        Raises
        ------
        TypeError
            If the path is not of type pathlib.Path or str
        """
        if isinstance(self._path, pathlib.Path):
            return self._path
        if isinstance(self._path, str):
            self._path = pathlib.Path(self._path)
        return self._path

    @property
    def _NAME(self) -> str:
        """Version name from provided path

        Returns
        -------
        str
            Name of the version
        """
        return self._PATH.name

    @property
    def _CONTENTS(self) -> List[pathlib.Path]:
        """Return one-level glob of the provided path

        Returns
        -------
        List[pathlib.Path]
            List of contents in the provided path
        """
        return list(self._PATH.glob("*"))

    @property
    def hparams(self):
        """Check if the .yaml exists and instantiate the HParams class each time

        Returns
        -------
        HParams
            Instance of HParams class if hparams.yaml exists
        """
        hparams_path = self._PATH.joinpath("hparams.yaml")
        if hparams_path.exists():
            return HParams(hparams_path)

    @property
    def metrics_df(self) -> pd.DataFrame:
        """Check if metrics.csv path exists and read it

        Returns
        -------
        pd.DataFrame
            DataFrame containing the metrics if metrics.csv exists
        """
        metrics_path = self._PATH.joinpath("metrics.csv")
        if metrics_path.exists():
            return pd.read_csv(metrics_path)

    @property
    def per_epoch_metrics(self):
        """Group metrics by the specified groupby attribute

        Returns
        -------
        GroupedMetrics
            Instance of GroupedMetrics class
        """
        self._GROUPED_METRICS = GroupedMetrics(groupby=self._groupby)
        return self._GROUPED_METRICS(self.metrics_df)

    @property
    def _CKPT_PATHS(self) -> List[pathlib.Path]:
        """Formatted checkpoint paths

        Returns
        -------
        List[pathlib.Path]
            List of checkpoint paths
        """
        _ckpt_paths = list(self._PATH.joinpath("checkpoints").glob("*"))
        return [pathlib.Path(path) for path in _ckpt_paths]

    @property
    def _SORTED_CKPT_KEYS(self) -> List:
        """Sorting for organization's sake

        Returns
        -------
        List
            Sorted list of checkpoint keys
        """
        epochs = list(self.ckpts.keys())
        _epochs = sorted([epoch for epoch in epochs if epoch != "last"])
        if "last" in epochs:
            _epochs.append("last")
        return _epochs

    @property
    def ckpts(self) -> Dict:
        """Format and update available checkpoints for the version

        Returns
        -------
        Dict
            Dictionary of checkpoints
        """
        if not hasattr(self, "_CHECKPOINTS"):
            self._CHECKPOINTS = {}
            for ckpt_path in self._CKPT_PATHS:
                ckpt = Checkpoint(ckpt_path)
                self._CHECKPOINTS[ckpt.epoch] = ckpt
        return self._CHECKPOINTS

    def __repr__(self) -> str:
        """Return the name of the object

        Returns
        -------
        str
            Name of the object
        """
        return self._NAME
