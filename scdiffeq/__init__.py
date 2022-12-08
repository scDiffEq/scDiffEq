
__module_name__ = "__init__.py"
__version__ = "0.0.44"
__doc__ = """Top-level __init__ for the scdiffeq package."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- sub-package imports: ----------------------------------------------------------------
from . import _io as io
from . import _models as models


# -- developer imports: ------------------------------------------------------------------
from ._models import _base