
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

# from neural_diffeqs import NeuralODE, NeuralSDE

# from ._integrators import *

# from ._sinkhorn_divergence import SinkhornDivergence
# from ._configure import configure_lightning_trainer, prepare_LightningDataModule, InputConfiguration
# from ._batch_forward import BatchForward
# from ._scdiffeq_datamodule import scDiffEqDataModule

# from ._base_utility_functions import (
# #     autodevice,
#     func_params,
#     extract_func_kwargs,
#     local_arg_parser,
# )


from ._lightning_model import LightningModel

# from ._batch_forward import BatchForward