
__module_name__ = "__init__.py"
__version__ = "0.0.45"
__doc__ = """Top-level __init__ for the scdiffeq package."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- import model API: -------------------------------------------------------------------
from ._core._scdiffeq import scDiffEq


# -- import sub-packages: ----------------------------------------------------------------
from . import data
from . import io
from . import plotting
from . import tools


# -- developer import: -------------------------------------------------------------------
from . import _core
