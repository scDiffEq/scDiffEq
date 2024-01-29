
# -- import packages: ---------------------------------------------------------
import ABCParse
import pathlib
import yaml


# -- set types: ---------------------------------------------------------------
from typing import Any, Dict, Union


# -- operational class: -------------------------------------------------------
class HParams(ABCParse.ABCParse):
    """scDiffEq container for HyperParams"""

    def __init__(self, yaml_path: Union[pathlib.Path, str]) -> None:
        """Initialize the HParams object by providing a path to the
        corresponding yaml file (created by Lightning)

        Args:
            yaml_path (Union[pathlib.Path, str]): path to the hparams file created
            by Lightning.

        Returns:
            None
        """
        self.__configure__(locals())

    def _read(self) -> None:
        """read path to yaml file and set as cls attribute"""
        if not hasattr(self, "_file"):
            self._yaml_file = yaml.load(open(self._yaml_path), Loader=yaml.Loader)

    def __configure__(self, kwargs, private=["yaml_path"]) -> None:
        """set hparams as cls attributes"""
        self.__parse__(kwargs, private=private)
        self._read()
        for key, val in self._yaml_file.items():
            setattr(self, key, val)

    @property
    def _ATTRS(self) -> Dict[str, Any]:
        """formatted attribute dictionary from hparams.yaml"""
        self._attrs = {
            attr: getattr(self, attr)
            for attr in self.__dir__()
            if not attr[0] in ["_", "a"]
        }
        return self._attrs

    def __getitem__(self, attr: str) -> Any:
        """format version key and return path"""
        return self._ATTRS[attr]

    def __repr__(self):
        """Return a readable representation of the discovered hyperparameters"""

        string = "HyperParameters\n"
        for attr, val in self._ATTRS.items():
            string += "\n  {:<34}: {}".format(attr, val)

        return string

    def __call__(self) -> Dict[str, Any]:
        """Return formatted dictionary of attributes from the hparams.yaml

        Args:
            None

        Returns:
            Dict[str, Any]
        """
        return self._ATTRS
