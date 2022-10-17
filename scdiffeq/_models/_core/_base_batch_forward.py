

from abc import ABC, abstractmethod
from ._integrators import credential_handoff


class BaseBatchForward(ABC):
    def __init__(self, func, loss_function):
        """To-do: add docs."""
        self.integrator, self.func_type = credential_handoff(func)
        self.loss_function = loss_function
        self.func = func

    @abstractmethod
    def __parse__(self):
        pass

    @abstractmethod
    def __inference__(self):
        pass

    @abstractmethod
    def __loss__(self):
        pass
    
    @abstractmethod
    def __call__(self, model, batch, stage, **kwargs):
        pass