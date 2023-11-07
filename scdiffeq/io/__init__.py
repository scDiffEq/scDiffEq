
__module_name__ = "__init__.py"
__doc__ = """I/O __init__ module. Sub-package of the main scdiffeq API."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["mvinyard.ai@gmail.com"])


# import functions accessed as sdq.io.<func>: --------------------------------------------
from ._read_h5ad import read_h5ad
from ._pickle_io import read_pickle, write_pickle
from ._model_loader import load_diffeq, load_model