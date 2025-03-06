# -- import packages: ---------------------------------------------------------
import ABCParse
import pathlib
import yaml

# -- set types: ---------------------------------------------------------------
from typing import Any, Dict, Union


# -- operational class: -------------------------------------------------------
class HParams(ABCParse.ABCParse):
    """scDiffEq container for HyperParams

    Attributes:
        _yaml_path (Union[pathlib.Path, str]): Path to the hparams file created by Lightning.
        _yaml_file (dict): Dictionary containing the contents of the yaml file.
        _attrs (Dict[str, Any]): Formatted attribute dictionary from hparams.yaml.
    """

    def __init__(self, yaml_path: Union[pathlib.Path, str]) -> None:
        """Initialize the HParams object by providing a path to thecorresponding yaml file (created by Lightning)

        Args:
            yaml_path (Union[pathlib.Path, str]): Path to the hparams file created by Lightning.

        Returns:
            None
        """
        self.__configure__(locals())

    def _read(self) -> None:
        """Read path to yaml file and set as class attribute

        Returns:
            None
        """
        if not hasattr(self, "_file"):
            self._yaml_file = yaml.load(open(self._yaml_path), Loader=yaml.Loader)

    def __configure__(self, kwargs, private=["yaml_path"]) -> None:
        """Set hparams as class attributes

        Args:
            kwargs (dict): Dictionary of keyword arguments.
            private (list, optional): List of private attributes, by default ["yaml_path"]

        Returns:
            None
        """
        self.__parse__(kwargs, private=private)
        self._read()
        for key, val in self._yaml_file.items():
            setattr(self, key, val)

    @property
    def _ATTRS(self) -> Dict[str, Any]:
        """Formatted attribute dictionary from hparams.yaml

        Returns:
            Dict[str, Any]
            Dictionary of attributes.
        """
        self._attrs = {
            attr: getattr(self, attr)
            for attr in self.__dir__()
            if not attr[0] in ["_", "a"]
        }
        return self._attrs

    def __getitem__(self, attr: str) -> Any:
        """Format version key and return path

        Args:
            attr (str): Attribute name.

        Returns:
            Any
            Attribute value.
        """
        return self._ATTRS[attr]

    def __repr__(self) -> str:
        """Return a readable representation of the discovered hyperparameters

        Returns:
            str
            Readable representation of the hyperparameters.
        """
        string = "HyperParameters\n"
        for attr, val in self._ATTRS.items():
            string += "\n  {:<34}: {}".format(attr, val)

        return string

    def __call__(self) -> Dict[str, Any]:
        """Return formatted dictionary of attributes from the hparams.yaml

        Returns:
            Dict[str, Any]
            Dictionary of attributes.
        """
        return self._ATTRS
