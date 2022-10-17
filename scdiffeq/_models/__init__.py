
__module_name__ = """__init__.py"""
__doc__ = """To-Do"""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])



# specify version: -----------------------------------------------------------------------
__version__ = "0.0.44"


from ._core._base_model import BaseModel
from ._core import *
from ._scdiffeq import scDiffEq
from ._prescient import PRESCIENT