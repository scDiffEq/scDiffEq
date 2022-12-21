
__module_name__ = "__init__.py"
__version__ = "0.0.45"
__doc__ = """data __init__ module. Sub-package of the main scdiffeq API."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# import functions accessed as sdq.data.<func>: ------------------------------------------
from ._lightning_anndata_module import LightningAnnDataModule
from ._larry_lightning_data_module import LARRY_LightningDataModule as LARRY