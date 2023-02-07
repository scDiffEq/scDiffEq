
import brownian_diffuser
import neural_diffeqs
import torchdiffeq
import torchsde


class Credentials:
    def __init__(self, func, adjoint=False):
        self.func = func
        self._adjoint = adjoint

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

    @property
    def mu_is_potential_func(self):
        """Assumes potential is 1-D"""
        if not self.is_TorchNet:
            return list(self.func.mu.parameters())[-1].shape[0] == 1
        return list(self.func.parameters())[-1].shape[0] == 1
    
    @property
    def sigma_is_potential_func(self):
        """Assumes potential is 1-D"""
        if not self.is_NeuralSDE:
            return False
        return list(self.func.sigma.parameters())[-1].shape[0] == 1

    def __call__(self):

        if self.is_NeuralODE:
            if self._adjoint:
                self._integrator = torchdiffeq.odeint_adjoint
            else:
                self._integrator = torchdiffeq.odeint
            return "NeuralODE", self.mu_is_potential_func, self.sigma_is_potential_func

        if self.is_NeuralSDE:
            if self._adjoint:
                self._integrator = torchsde.sdeint_adjoint
            else:
                self._integrator = torchsde.sdeint
            return "NeuralSDE", self.mu_is_potential_func, self.sigma_is_potential_func

        if self.is_TorchNet:
            self._integrator = brownian_diffuser.nn_int
            return "TorchNet", self.mu_is_potential_func, self.sigma_is_potential_func

def function_credentials(func, adjoint=False):
    
    creds = Credentials(func, adjoint=adjoint)
    func_type, mu_is_potential, sigma_is_potential = creds()
    
    return {
        "func_type": func_type,
        "mu_is_potential": mu_is_potential,
        "sigma_is_potential": sigma_is_potential,
        "use_adjoint": adjoint,
    }