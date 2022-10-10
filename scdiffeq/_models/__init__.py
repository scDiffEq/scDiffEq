
__module_name__ = "__init__.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])



# specify version: -----------------------------------------------------------------------
__version__ = "0.0.43"

from ._core._BaseModel import BaseModel
from ._scDiffEq import scDiffEq
from ._PRESCIENT import PRESCIENT


from ._core import _base_ancilliary as base
from ._core import _integrators as integrators