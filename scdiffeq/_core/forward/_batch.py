
from ..utils import sum_normalize

class Batch:
    """Catch batch and make sure it's in the right format."""
    def __init__(self, batch, func_type):
        self._passed_batch = batch
        self._func_type = func_type

    @property
    def t(self):
        _t = self._passed_batch[0].unique()
        if self._func_type == "NeuralSDE":
            return {"ts": _t}
        return {"t": _t}

    @property
    def X(self):
        return self._passed_batch[1].transpose(1,0)

    @property
    def X0(self):
        return self.X[0]

    @property
    def W(self):
        return sum_normalize(self._passed_batch[2].transpose(1,0))