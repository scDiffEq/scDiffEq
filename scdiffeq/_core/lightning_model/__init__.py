
__module_name__ = "__init__.py"
__doc__ = """LightningModels __init__.py"""
__version__ = "0.0.45"
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)

from ._lightning_diffeq import LightningDiffEq
from ._default_neural_sde import default_NeuralSDE