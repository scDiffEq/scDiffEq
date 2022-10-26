
__module_name__ = "__init__.py"
__doc__ = """Main __init__ module - most user-visible API."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(["mvinyard@broadinstitute.org", "arasmuss@broadinstitute.org", "ruitong@broadinstitute.org"])


# version: -------------------------------------------------------------------------------
__version__ = "0.0.44"


from . import _io as io
from . import _models as models
from ._models._base import _core as core