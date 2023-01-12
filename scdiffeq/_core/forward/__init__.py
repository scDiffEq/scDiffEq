
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


# -- import base and derived module groups: ----------------------------------------------
# from ._forward import UniversalForwardIntegrator, forward
from ._batch import Batch
from ._loss_log import LossLog
from ._function_credentials import Credentials
from ._potential_regularizer import PotentialRegularizer
from ._loss import Loss
from ._sinkhorn_divergence import SinkhornDivergence

from ._loss_manager import LossManager
from ._forward_manager import ForwardManager