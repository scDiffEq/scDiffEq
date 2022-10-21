
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


# version: -------------------------------------------------------------------------------
__version__ = "0.0.44"


# import modules / functions to be accessed as sdq.models.<MODEL>: -----------------------
from ._scdiffeq import scDiffEq
from ._base._core._base_model import BaseModel
from . import _base as base

<<<<<<< HEAD
from ._base._core._prepare_data import prepare_data

=======
>>>>>>> 2b690a534e502cdc43b2e976220475b8f72fdf16
# to-do: ---------------------------------------------------------------------------------
# from ._prescient import PRESCIENT