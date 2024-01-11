
__module_name__ = "_checkpoint.py"
__doc__ = """Module containing the ``Checkpoint`` container object."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["mvinyard.ai@gmail.com"])


# -- import packages: ---------------------------------------------------------
import ABCParse
import pathlib
import torch


# -- set typing: --------------------------------------------------------------
from typing import Union, Dict


# -- operational class: -------------------------------------------------------
class Checkpoint(ABCParse.ABCParse):
    def __init__(self, path: Union[pathlib.Path, str], *args, **kwargs) -> None:
        """Instantiates checkpoint
        
        Args:
            path (Union[pathlib.Path, str]): path to saved checkpoint.
            
        Returns:
            None
        """
        self.__parse__(locals())
        
    @property
    def path(self) -> pathlib.Path:
        """"""
        return pathlib.Path(self._path)

    @property
    def _fname(self) -> str:
        """filename"""
        return self.path.name.split(".")[0]

    @property
    def epoch(self) -> Union[int, str]:
        if self._fname != "last":
            return int(self._fname.split("=")[1].split("-")[0])
        return self._fname

    @property
    def state_dict(self) -> Dict[str, 'LightningCheckpoint']:
        """loads state_dict created by PyTorch Lightning"""
        if not hasattr(self, "_ckpt"):
            self._state_dict = torch.load(self.path)  # ["state_dict"]
        return self._state_dict

    def __repr__(self) -> str:
        """obj description of checkpoint at epoch"""
        return f"ckpt epoch: {self.epoch}"
