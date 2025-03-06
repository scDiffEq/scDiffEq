# -- import packages: ---------------------------------------------------------
import ABCParse
import pathlib


# -- import local dependencies: -----------------------------------------------
from ._version import Version


# -- set typing: --------------------------------------------------------------
from typing import Dict, Union


# -- operational class: -------------------------------------------------------
class Project(ABCParse.ABCParse):
    """Object container for an scDiffEq project

    Attributes:
        path (Union[str, pathlib.Path]): Path to the project, created by scDiffeq.
        metrics_groupby (str): Grouping method for metrics, default is "epoch".
    """

    def __init__(
        self,
        path: Union[str, pathlib.Path] = pathlib.Path("./").absolute(),
        metrics_groupby: str = "epoch",
        *args,
        **kwargs,
    ) -> None:
        """Initialize the project object by providing a path.

        Args:
            path (Union[str, pathlib.Path], optional): Path to the project, created by scDiffeq, by default pathlib.Path("./").absolute()
            metrics_groupby (str, optional): Grouping method for metrics, by default "epoch"

        Returns:
            None
        """
        self.__parse__(locals())

        self._set_version_path_attributes()

    @property
    def _PROJECT_PATH(self) -> pathlib.Path:
        """Check and format the provided project path

        Returns:
            pathlib.Path
            Formatted project path

        Raises:
            TypeError: If the path is not of type pathlib.Path or str
        """
        if isinstance(self._path, pathlib.Path):
            return self._path
        elif isinstance(self._path, str):
            return pathlib.Path(self._path)
        else:
            raise TypeError("arg: `path` must be of type: [pathlib.Path, str]")

    @property
    def _VERSION_PATHS(self) -> Dict[str, pathlib.Path]:
        """Assemble available version paths based on the provided project path

        Returns:
            Dict[str, pathlib.Path]
            Dictionary of version names and their corresponding paths
        """
        version_paths = sorted(list(self._PROJECT_PATH.glob("version_*")))
        return {path.name: path for path in version_paths}

    def _set_version_path_attributes(self) -> None:
        """Sets the name of each version as a class attribute, pointing to the
        path of the version.
        """
        for v_name, v_path in self._VERSION_PATHS.items():
            version = Version(path=v_path, groupby=self._metrics_groupby)
            setattr(self, v_name, version)

    def __getitem__(self, version: int) -> pathlib.Path:
        """Format version key and return version

        Args:
            version (int): Version number

        Returns:
            pathlib.Path
            Path to the specified version
        """
        return getattr(self, f"version_{version}")

    def __repr__(self) -> str:
        return """scDiffEq Project"""
