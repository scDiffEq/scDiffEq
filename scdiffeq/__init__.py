# scdiffeq __init__.py
__module_name__ = "__init__.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import model #
# ------------ #
from ._model._scDiffEq_model import _scDiffEq as scDiffEq


# import sub-packages #
# ------------------- #
from . import _data as data
from . import _tools as tl
from . import _plotting as pl
from . import _utilities as ut
from . import _study as study
from . import _io as io

from ._model._supporting_functions._training._OptimalTransportLoss import _OptimalTransportLoss as OTLoss

# pandas defaults #
# --------------- #
import pandas as pd

pd.set_option("display.max_columns", None)


# matplotlib defaults #
# ------------------- #
import matplotlib
import matplotlib.font_manager

font = {"size": 12}
matplotlib.rc("font", **font)
matplotlib.rcParams["font.sans-serif"] = "Arial"
matplotlib.rcParams["font.family"] = "sans-serif"
