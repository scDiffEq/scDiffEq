
# -- import local dependencies: --------------------------------------------------------
from ..utils import sum_normalize


# -- Model-facing Batch: ---------------------------------------------------------------
class Batch:
    """Catch batch and make sure it's in the right format."""
    
    def __parse__(self, kwargs, ignore=['self']):
        for key, val in kwargs.items():
            setattr(self, "_{}".format(key), val)
    
    def __init__(self, batch, stage, func_type):
        self.__parse__(locals())
        
    @property
    def stage(self):
        return self._stage
    
    @property
    def t(self):
        _t = self._batch[0].unique()
        if self._func_type == "NeuralSDE":
            return {"ts": _t}
        return {"t": _t}

    @property
    def X(self):
        return self._batch[1].transpose(1,0)

    @property
    def X0(self):
        return self.X[0]

    @property
    def W(self):
        return sum_normalize(self._batch[2].transpose(1,0))