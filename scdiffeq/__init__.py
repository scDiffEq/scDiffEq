# scdiffeq __init__.py
__module_name__ = "__init__.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import model #
# ------------ #
# from ._model._scDiffEq_model import _scDiffEq as scDiffEq

# import sub-packages #
# ------------------- #
# from . import _clonal as clonal
from . import _data as data
# from . import _tools as tl
# from . import _plotting as pl
from . import _utilities as ut
# from . import _study as study
# from . import _io as io

# from ._model._supporting_functions._training._OptimalTransportLoss import _OptimalTransportLoss as OTLoss

# pandas defaults #
# --------------- #
import pandas as _pd

_pd.set_option("display.max_columns", None)


# matplotlib defaults #
# ------------------- #
import matplotlib as _mpl
# import matplotlib.font_manager as _mpl_fm

font = {"size": 12}
_mpl.rc("font", **font)
_mpl.rcParams["font.sans-serif"] = "Arial"
_mpl.rcParams["font.family"] = "sans-serif"
