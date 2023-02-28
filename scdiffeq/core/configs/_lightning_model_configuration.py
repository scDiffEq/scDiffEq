
# -- import packages: ------
import brownian_diffuser
import neural_diffeqs
import torchdiffeq
import torchsde


# -- import local dependencies: ------
from .. import lightning_models


# -- main operator class: ------------
class LightningModelConfiguration:
    def __init__(self, func, adjoint=False):
        self.func = func
        self._adjoint = adjoint

        self.func_type, self.mu_is_potential, self.sigma_is_potential = self.function_credentials

    @property
    def function_credentials(self):
        if isinstance(self.func, neural_diffeqs.NeuralODE):
            if self._adjoint:
                integrator = torchdiffeq.odeint_adjoint
            else:
                integrator = torchdiffeq.odeint
            return "NeuralODE", self._is_mu_potential(), self._is_sigma_potential()

        if isinstance(self.func, neural_diffeqs.NeuralSDE):
            if self._adjoint:
                integrator = torchsde.sdeint_adjoint
            else:
                integrator = torchsde.sdeint
            return "NeuralSDE", self._is_mu_potential(), self._is_sigma_potential()

        integrator = brownian_diffuser.nn_int
        self._integrator = brownian_diffuser.nn_int
        return "DriftNet", self._is_mu_potential(), self._is_sigma_potential()
    
    def _is_mu_potential(self):
        """Assumes potential is 1-D"""
        if not isinstance(self.func, neural_diffeqs.NeuralODE):
            return False
        return list(self.func.mu.parameters())[-1].shape[0] == 1

    def _is_sigma_potential(self):
        """Assumes potential is 1-D"""
        if not isinstance(self.func, neural_diffeqs.NeuralSDE):
            return False
        return list(self.func.sigma.parameters())[-1].shape[0] == 1

    def _non_import(self, attr):
        return (
            attr.startswith("_")
            or attr.startswith("Base")
            or (attr in ["SinkhornDivergence"])
        )

    @property
    def available_models(self):
        """Available, implemented models"""
        return {
            attr: getattr(lightning_models, attr)
            for attr in lightning_models.__dir__()
            if not self._non_import(attr)
        }

    @property
    def FunctionModuleName(self):
        func_type = self.func_type.strip("Neural")
        if self.mu_is_potential:
            return "".join(["LightningPotential"] + [func_type])
        return "".join(["Lightning"] + [func_type])

    def __call__(self):
        """Return the LightningModel"""
        return self.available_models[self.FunctionModuleName](self.func)
