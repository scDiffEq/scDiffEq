__module_name__ = "__init__.py"
__doc__ = """Top-level __init__ for the scDiffEq package."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["mvinyard.ai@gmail.com"])


# -- setup logging: -----------------------------------------------------------
# import py_pkg_logging as _ppl
# _LOGGING = _ppl.PackageLogging(name = __name__, file = __file__)


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
