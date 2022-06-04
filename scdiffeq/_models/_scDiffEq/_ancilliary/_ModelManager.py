
__module_name__ = "_ModelManager.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import local dependencies #
# ------------------------- #
# from . import _model_functions as funcs
from ._utility_functions._utilities import _transfer_attributes
from ._utility_functions._count_model_params import _count_model_params

class _ModelManager:
    def __init__(
        self,
        model,
    ):
        
        _transfer_attributes(model, self)
        self._ParamCount = _count_model_params(model)