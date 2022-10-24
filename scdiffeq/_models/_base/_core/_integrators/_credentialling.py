
__module_name__ = "_credentialling.py"
__doc__ = """To-do."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- import packages: --------------------------------------------------------------------
from neural_diffeqs import NeuralODE, NeuralSDE
from torch_composer import TorchNet
import torch


# -- import local dependencies: ----------------------------------------------------------
from ._integrators import SDEIntegrator, ODEIntegrator, TorchNNIntegrator


# -- controller class: -------------------------------------------------------------------
class FunctionCredentials:
    def __init__(self):
        pass

    def _is_neural_ode(self, func):
        return isinstance(func, NeuralODE)

    def _is_neural_sde(self, func):
        return isinstance(func, NeuralSDE)

    def _is_torch_net(self, func):
        return isinstance(func, torch.nn.Module)

    def _is_potential_net(self, func):
        pass

    def _has_f(self, func):
        return hasattr(func, "f")

    def _has_g(self, func):
        return hasattr(func, "g")

    def _has_forward(self, func):
        return hasattr(func, "forward")

    def CREDENTIALS_HANDOFF(self, CRED):
        """
        Notes:
        ------
        Could update with hierarchical func type (as is now)
        then potential/forward net.
        """
        if all([CRED["IS"]["SDE"], CRED["HAS"]["f"], CRED["HAS"]["g"]]):
            return SDEIntegrator(), "neural_SDE"
        if all([CRED["IS"]["ODE"], CRED["HAS"]["forward"]]):
            return ODEIntegrator(), "neural_ODE"
        if all([CRED["IS"]["TORCH_NET"]]):
            return TorchNNIntegrator(), "neural_net"

    def __call__(self, func):
        """returns integrator, function_type"""

        CREDENTIALS = {"IS": {}, "HAS": {}}
        CREDENTIALS["IS"]["ODE"] = self._is_neural_ode(func)
        CREDENTIALS["IS"]["SDE"] = self._is_neural_sde(func)
        CREDENTIALS["IS"]["TORCH_NET"] = self._is_torch_net(func)
        CREDENTIALS["IS"]["POTENTIAL_NET"] = self._is_potential_net(func)
        CREDENTIALS["HAS"]["f"] = self._has_f(func)
        CREDENTIALS["HAS"]["g"] = self._has_g(func)
        CREDENTIALS["HAS"]["forward"] = self._has_forward(func)

        return self.CREDENTIALS_HANDOFF(CREDENTIALS)


# -- model-facing function: --------------------------------------------------------------
def credential_handoff(func):
    CREDENTIALER = FunctionCredentials()
    return CREDENTIALER(func)
