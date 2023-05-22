
__module_name__ = "__init__.py"
__version__ = __Version__ = "0.0.47rc1"
__doc__ = """Top-level __init__ for the scdiffeq package."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
    ]
)


# -- import model API: -------------------------------------------------------------------
from .core._scdiffeq import scDiffEq


# -- import sub-packages: ----------------------------------------------------------------
from . import core
from . import io
from . import plotting as pl
from . import tools as tl
