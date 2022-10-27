

__module_name__ = "_BrownianDiffuser.py"
__doc__ = """BrownianDiffuser module"""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)



# -- import packages: --------------------------------------------------------------------
import torch
import numpy as np


# from .._base_utility_functions import autodevice


# -- general setup function: -------------------------------------------------------------
def timespan(t: (torch.Tensor or np.ndarray)):
    return (t.max() - t.min()).item()

def state_shape(X_state, n_steps):
    return [n_steps] + list(X_state.shape)


def brownian_motion(X_state, stdev, n_steps):
    return torch.randn(state_shape(X_state, n_steps), requires_grad=True) * stdev


def is_potential_net(net):
    return list(net.parameters())[-1].data.numel() == 1


def do_steps(max_steps, n_steps):
    if max_steps:
        return max_steps
    return n_steps


# -- main BrownianDiffuser class: --------------------------------------------------------
class BrownianDiffuser:
    """Class for manual drift functions"""

    # -- one-time init calculations: -----------------------------------------------------
    def __init__(self, X0, t, device, dt=0.1, stdev=0.5, max_steps=None): # =autodevice()
        """
        Set up all elements that are not necessary to repeat, here.
        """

        self.__dict__.update(locals())
        self.sqrt_dt = np.sqrt(self.dt)
        self.n_steps = int(timespan(self.t) / self.dt)
        self.do_steps = do_steps(self.max_steps, self.n_steps)
        self.Z = brownian_motion(self.X0, stdev, n_steps=self.n_steps).to(self.device)

    # -- lowest-level operators: ---------------------------------------------------------
    def _potential(self, net, x):
        """
        Pass through potential net
        Note:
        -----
        (1) I'd need to look deeper, but if memory serves, there is a good
            reason for dedicating an entire line to x = x.requires_grad_()
            rather than simply returning net(x.requires_grad_()).
        """
        x = x.requires_grad_()
        return net(x)

    def potential_drift(self, net, x):
        """
        Return the drift position as the gradient of the potential.
        """
        potential = self._potential(net, x)
        return torch.autograd.grad(
            potential, x, torch.ones_like(potential), create_graph=True
        )[0]

    def forward_drift(self, net, x):
        """
        If not a potential_net, simply pass the torch.nn.Module
        through the network.
        """
        return net(x)

    # -- setup mu: -----------------------------------------------------------------------
    def set_mu(self, net):
        """
        Determine which function should be called for calculating drift.
        Ideally, only called once.
        """
        if not hasattr(self, "mu"):
            if is_potential_net(net):
                setattr(self, "mu", getattr(self, "potential_drift"))
            setattr(self, "mu", getattr(self, "forward_drift"))

    # -- a single brownian step: ---------------------------------------------------------
    def brownian_step(self, net, X_state, Z_step):
        """
        Take a singular brownian step.
        """
        return X_state + self.mu(net, X_state) * self.dt + Z_step * self.sqrt_dt

    # -- flat-memory stepper: ------------------------------------------------------------
    def run_generator(self, net):
        """Generator to control / execute brownian stepping."""
        self.set_mu(net)
        current_step = 0
        X_state = self.X0
        while current_step < self.do_steps:
            X_state = self.brownian_step(net, X_state, self.Z[current_step])
            yield X_state
            current_step += 1

    # -- call the generator and format the outputs: --------------------------------------
    def __call__(self, net):
        """The main API function."""
        return torch.stack(list(self.run_generator(net)))