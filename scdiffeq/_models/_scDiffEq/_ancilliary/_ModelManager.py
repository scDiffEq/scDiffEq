
__module_name__ = "_ModelManager.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import local dependencies #
# ------------------------- #
from . import _model_functions as funcs


class _ModelManager:
    def __init__(
        self,
        model,
    ):
        
        funcs.transfer_attributes(model, self)
        self._ParamCount = funcs.count_model_params(model)