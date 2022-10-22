
__module_name__ = "__init__.py"
__doc__ = """To-do."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# specify version: -----------------------------------------------------------------------
__version__ = "0.0.44"


# -- import base module groups: ----------------------------------------------------------
from ._lightning_callbacks import *
from ._core import *
