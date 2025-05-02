# -- configure logging: -------------------------------------------------------
from ._backend_utilities._configure_logging import configure_logging

logger = configure_logging()

# -- handle dependency-related warnings: --------------------------------------
import os as _os

_os.environ["KEOPS_VERBOSE"] = "0"


# -- import model API: --------------------------------------------------------
from .core._scdiffeq import scDiffEq


# -- import sub-packages: -----------------------------------------------------
from . import core
from . import io
from . import plotting as pl
from . import tools as tl
from . import datasets
from . import _backend_utilities as utils


# -- version: -----------------------------------------------------------------
from .__version__ import __version__


__all__ = [
    "scDiffEq",
    "core",
    "io",
    "pl",
    "tl",
    "datasets",
    "utils",
    "__version__",
    "logger",
]
