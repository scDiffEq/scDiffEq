
__module_name__ = "__init__.py"
__version__ = "0.0.45"
__doc__ = """I/O __init__ module. Sub-package of the main scdiffeq API."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# import functions accessed as sdq.io.<func>: --------------------------------------------
from ._read_h5ad import read_h5ad
from ._pickle_io import read_pickle, write_pickle