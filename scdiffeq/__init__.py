
__module_name__ = "__init__.py"
__doc__ = """Top-level __init__ for the scDiffEq package."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])


import os as _os

class _PackageVersion:
    
    def __init__(self):
        ...
        
    @property
    def PACKAGE_PATH(self):
        return _os.path.abspath(_os.path.dirname(_os.path.dirname(__file__)))
    
    @property
    def SETUP_FPATH(self):
        return _os.path.join(self.PACKAGE_PATH, "setup.py")
    
    def _read_setup_dot_py(self):
        f = open(self.SETUP_FPATH)
        file = f.readlines()
        f.close()
        return file
    
    @property
    def VERSION(self):
        file = self._read_setup_dot_py()
        return [line for line in file if "version" in line][0].split("version=")[1].split('"')[1]
    
    def __call__(self):
        return self.VERSION
    
_package_version = _PackageVersion()
__version__ = __VERSION__ = __Version__ = _package_version()

_os.environ["KEOPS_VERBOSE"] = "0"


# -- import model API: -------------------------------------------------------------------
from .core._scdiffeq import scDiffEq


# -- import sub-packages: ----------------------------------------------------------------
from . import core
from . import io
from . import plotting as pl
from . import tools as tl
from . import datasets

from . import _backend_utilities as utils