# -- import packages: ---------------------------------------------------------
import ABCParse
import logging
import pandas as pd
import pathlib
import torch


# -- set typing: --------------------------------------------------------------
from typing import Union, Dict


# -- set up logging: ----------------------------------------------------------
logger = logging.getLogger(__name__)


# -- operational class: -------------------------------------------------------
class Checkpoint(ABCParse.ABCParse):
    def __init__(self, path: Union[pathlib.Path, str], *args, **kwargs) -> None:
        """Instantiates checkpoint object.

        Args:
            path (Union[pathlib.Path, str]): Path to saved checkpoint.
        """
        self.__parse__(locals())

    @property
    def path(self) -> pathlib.Path:
        """
        Returns:
            pathlib.Path
            The path to the checkpoint.
        """
        return pathlib.Path(self._path)

    @property
    def _fname(self) -> str:
        """
        Returns:
            str
            Filename without extension.
        """
        return self.path.name.split(".")[0]

    @property
    def version(self):
        """
        Returns:
            str
            Version of the checkpoint.
        """
        if not hasattr(self, "_version"):
            v, n = self.path.parent.parent.name.split("_")
            self._version = " ".join([v.capitalize(), n])
        return self._version

    @property
    def _PATH_F_HAT_RAW(self):
        """
        Returns:
            pathlib.Path
            Path to the raw F_hat file.
        """
        if not hasattr(self, "_FATE_PREDICTION_METRICS_PATH"):
            base_path = self.path.parent.parent.joinpath("fate_prediction_metrics")
            converted_name = (
                self.path.name.replace("=", "_").replace("-", ".").split(".ckpt")[0]
            )
            self._FATE_PREDICTION_METRICS_PATH = base_path.joinpath(
                f"{converted_name}/F_hat.unfiltered.csv"
            )
        return self._FATE_PREDICTION_METRICS_PATH

    @property
    def F_hat(self):
        """
        Returns:
            pd.DataFrame or None
            DataFrame containing F_hat data if the path exists, otherwise None.
        """
        if not hasattr(self, "_F_hat"):
            if self._PATH_F_HAT_RAW.exists():
                self._F_hat = pd.read_csv(self._PATH_F_HAT_RAW, index_col=0)
                self._F_hat.index = self._F_hat.index.astype(str)
            else:
                logger.warning(f"F_hat path does not exist.")
                self._F_hat = None
        return self._F_hat

    @property
    def epoch(self) -> Union[int, str]:
        """
        Returns:
            Union[int, str]
            Epoch number if not 'last', otherwise 'last'.
        """
        if self._fname != "last":
            return int(self._fname.split("=")[1].split("-")[0])
        return self._fname

    @property
    def state_dict(self) -> Dict[str, "LightningCheckpoint"]:
        """
        Returns:
            Dict[str, "LightningCheckpoint"]
            State dictionary created by PyTorch Lightning.
        """
        if not hasattr(self, "_ckpt"):
            self._state_dict = torch.load(self.path)  # ["state_dict"]
        return self._state_dict

    def __repr__(self) -> str:
        """
        Returns:
            str
            Object description of checkpoint at epoch.
        """
        return f"ckpt epoch: {self.epoch} [{self.version}]"
