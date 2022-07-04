
__module_name__ = "_ModelManager.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import local dependencies #
# ------------------------- #
from .._model_utils._count_model_params import _count_model_params
from .._model_utils._transfer_attributes import _transfer_attributes


class _ModelManager:
    def __init__(
        self,
        model,
    ):
        
#         _transfer_attributes(model, self)
        self._ParamCount = _count_model_params(model)