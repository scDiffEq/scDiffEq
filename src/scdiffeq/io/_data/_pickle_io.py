# -- import packages: ---------------------------------------------------------
import pathlib
import pickle


# -- set typing: --------------------------------------------------------------
from typing import Any, Dict, Optional, Union


# -- operational class: -------------------------------------------------------
class PickleIO:
    def __init__(self, *args, **kwargs):
        """ """
        pass

    def read(self, path, mode="rb"):
        return pickle.load(self.__path__(path, mode))

    def write(self, obj, path, mode="wb", protocol=pickle.HIGHEST_PROTOCOL):
        """If writing for use in colab, use protocol=4"""

        pickle.dump(obj=obj, file=self.__path__(path, mode), protocol=protocol)

    def __path__(self, path, mode):
        return open(path, mode)


# -- API-facing functions: ----------------------------------------------------
def read_pickle(path: Union[str, pathlib.Path], mode: Optional[str] = "rb"):
    """Read the contents of a pickle file into memory.

    Args:
        path (Union[str, pathlib.Path]): pickle file path.

        mode (Optional[str]): read mode. **Default** = "rb"

    Returns:
        obj (Dict[Any,Any]): Object, usually a dictionary contained in pickle file.
    """

    pickle_io = PickleIO()
    return pickle_io.read(path, mode=mode)


def write_pickle(
    obj: Dict[Any, Any],
    path: Union[str, pathlib.Path],
    mode: Optional[str] = "wb",
    protocol: Optional[int] = pickle.HIGHEST_PROTOCOL,
) -> None:
    """Save an object to a pickle file.

    Args:
        obj (Dict[Any, Any]): Object, usually a dictionary to write to pickle file.

        path (Union[str, pathlib.Path]): Path to which pickle file should be written.

        mode (Optional[str]): Write mode. **Default** = "wb"

        protocol (Optional[int]): Pickling protocol. **Default** = pickle.HIGHEST_PROTOCOL

    Returns:
        None
    """

    pickle_io = PickleIO()
    pickle_io.write(obj=obj, path=path, mode=mode, protocol=protocol)
