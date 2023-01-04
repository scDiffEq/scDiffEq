import neural_diffeqs
import brownian_diffuser
import torchdiffeq
import torchsde


from ..utils import extract_func_kwargs


class Credentials:
    def __init__(self, func):
        self.func = func

    @property
    def is_NeuralODE(self):
        return isinstance(self.func, neural_diffeqs.NeuralODE)

    @property
    def is_NeuralSDE(self):
        return isinstance(self.func, neural_diffeqs.NeuralSDE)

    @property
    def is_TorchNet(self):
        return (not self.is_NeuralODE) and (not self.is_NeuralSDE)

    @property
    def integrator(self):
        return self._integrator

    def __call__(self):

        if self.is_NeuralODE:
            self._integrator = torchdiffeq.odeint
            return "NeuralODE"

        if self.is_NeuralSDE:
            self._integrator = torchsde.sdeint
            return "NeuralSDE"

        if self.is_TorchNet:
            self._integrator = brownian_diffuser.nn_int
            return "TorchNet"
        
        
class UniversalForwardIntegrator:
    def __parse__(self, kwargs, ignore=["self", "X0", "t"]):

        self.KWARGS = {}
        for key, val in kwargs.items():
            if not key in ignore:
                self.KWARGS[key] = val
                setattr(self, key, val)

    def __init__(self, func):

        self.func = func
        creds = Credentials(self.func)
        self.func_type = creds()
        self.integrator = creds.integrator

    def __call__(self, X0, t, dt=0.1):

        self.__parse__(locals())
        int_kwargs = extract_func_kwargs(func=self.integrator, kwargs=self.KWARGS)
        return self.integrator(self.func, X0, t, **int_kwargs)
