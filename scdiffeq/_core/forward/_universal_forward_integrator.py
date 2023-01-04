

from ..utils import extract_func_kwargs
from ._function_credentials import Credentials
        
        
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
        self.func_type, self.mu_is_potential, self.sigma_is_potential = creds()
        self.integrator = creds.integrator

    def __call__(self, X0, t, dt=0.1):

        self.__parse__(locals())
        int_kwargs = extract_func_kwargs(func=self.integrator, kwargs=self.KWARGS)
        return self.integrator(self.func, X0, t, **int_kwargs)
