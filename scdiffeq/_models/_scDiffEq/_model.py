
__module_name__ = "_model.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #


# import local dependencies #
# ------------------------- #
from ._ancilliary._ModelManager import _ModelManager
from ._ancilliary._Learner import _Learner
from ._ancilliary import _model_functions as funcs

class _scDiffEq:
    def __init__(self, adata=None):

        """ """
        
        self._adata = adata
        
        self._Model = _DefineModel()
        self._ModelManager = _ModelManager(self._Model)
        self._Learner = _Learner(self._Model)

    def train(self, training_args):

        _training_program(self._Model,
                          self._ModelManager,
                          self._Learner,
                          training_args,
                         )

    def evaluate(self):
        
        _evaluate(self._Learner)

        
    def load_run(self, path):
        
        self._path = path
        self._Model, self._ModelManager, self._Learner = _load_run(self._path)
    
    def load_model(self, path):
        
        self._path = path
        self._Model = _load_model(self._path)


    def save(self, path):

        self._ModelManager.save()
        