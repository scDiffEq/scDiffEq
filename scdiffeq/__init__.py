
__module_name__ = "__init__.py"
__doc__ = """Top-level __init__ for the scDiffEq package."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])


import os as _os


from ._version import _PackageVersion

_package_version = _PackageVersion()
__version__ = __VERSION__ = __Version__ = _package_version()

_os.environ["KEOPS_VERBOSE"] = "0"


# -- import model API: -------------------------------------------------------------------
from .core._scdiffeq import scDiffEq


# -- import sub-packages: ----------------------------------------------------------------
from . import core
from . import io
# from . import plotting as pl
from . import tools as tl
from . import datasets

from . import _backend_utilities as utils
