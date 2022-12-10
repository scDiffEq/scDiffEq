
__module_name__ = "__init__.py"
__version__ = "0.0.45"
__doc__ = """TODO"""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- import modules: ---------------------------------------------------------------------
from ._scdiffeq_configuration import scDiffEqConfiguration

from ._fetch_from_torch import fetch_optimizer, fetch_lr_scheduler
from ._extract_func_kwargs import func_params, extract_func_kwargs