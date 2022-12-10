
__module_name__ = "__init__.py"
__version__ = "0.0.45"
__doc__ = """Top-level __init__ for accessing the scDiffEq model."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- import models accessed as sdq.models.<MODEL>: ---------------------------------------
from ._scdiffeq import scDiffEq
from .loss import Loss

# -- developer-facing modules: -----------------------------------------------------------
from . import configs
from . import loss
from . import utils
