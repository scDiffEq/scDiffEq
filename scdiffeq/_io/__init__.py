
__module_name__ = "__init__.py"
__doc__ = """IO __init__ module. Sub-package of the main scdiffeq API."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(["mvinyard@broadinstitute.org", "arasmuss@broadinstitute.org", "ruitong@broadinstitute.org"])


# specify version: -----------------------------------------------------------------------
__version__ = "0.0.44"


# import functions to be accessed as sdq.io.<func>: --------------------------------------
from ._read_h5ad import read_h5ad