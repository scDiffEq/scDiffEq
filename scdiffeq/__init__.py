
# scdiffeq __init__.py
__module_name__ = "__init__.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])

### ------------------------------------------------- IMPORTS -------------------------------------------------- ###

# import sub-packages #
# ------------------- #
from . import _data as data
from . import _io as io
from . import _metrics as metrics
from . import _model as model
from . import _plotting as pl
<<<<<<< HEAD
from . import _preprocessing as pp
from . import _tools as tl
from . import _utilities as ut
=======
from . import _tools as tl
from . import _utilities as ut


# from . import _clonal as clonal
# from . import _study as study
>>>>>>> bd2e6c546905b2372e0e4d9443e0957793613823

### ------------------------------------------------- DEFAULTS ------------------------------------------------- ###


### ------------------------------------------------- DEFAULTS ------------------------------------------------- ###

# pandas defaults #
# --------------- #
import pandas as _pd

_pd.set_option("display.max_columns", None)


# matplotlib defaults #
# ------------------- #
import matplotlib as _mpl

_mpl.rc("font", **{"size": 12})
_mpl.rcParams["font.sans-serif"] = "Arial"
_mpl.rcParams["font.family"] = "sans-serif"
