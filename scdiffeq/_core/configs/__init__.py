
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

from ._lightning_model_configuration import LightningModelConfig
from ._lightning_trainer_configuration import LightningTrainerConfig
from ._lightning_data_module_configuration import LightningDataModuleConfig
from ._configure_time import TimeConfig