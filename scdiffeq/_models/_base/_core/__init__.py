
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


# -- specify version: --------------------------------------------------------------------
__version__ = "0.0.44"

from ._integrators import *

from ._sinkhorn_divergence import SinkhornDivergence
from ._prepare_lightning_data_module import prepare_LightningDataModule
from ._configure_lightning_trainer import configure_lightning_trainer
from ._configure_inputs import InputConfiguration
from ._batch_forward import BatchForward
# from ._base_model import BaseLightningModel, BaseModel

from ._base_utility_functions import (
    autodevice,
    func_params,
    extract_func_kwargs,
    local_arg_parser,
)