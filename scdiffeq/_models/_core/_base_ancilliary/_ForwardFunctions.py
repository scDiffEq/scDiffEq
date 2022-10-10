
__module_name__ = "_ForwardFunctions.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages: -----------------------------------------------------------------------
import numpy as np
import pydk
import torch
import torchsde
import torchdiffeq


# Module Table of Contents: --------------------------------------------------------------
# (1) Manual forward-stepping functions
# (2) Currently implemented credential-handling flags
# (3) Credentialing functions
# (4) Register integrator kwargs
# (5) Main module class: ForwardFunctions


# Manual forward-stepping functions (1): -------------------------------------------------
def _potential(net, x):
    x = x.requires_grad_()
    return net(x)


def _drift(net, x):
    x_ = x.requires_grad_()
    pot = _potential(net, x_)
    return torch.autograd.grad(pot, x_, torch.ones_like(pot), create_graph=True)[0]


def _forward_step(net, x, dt, z):
    sqrtdt = np.sqrt(dt)
    return x + _drift(net, x) * dt + z * sqrtdt


def _brownian_motion(x, stdev, n_steps=None):

    """
    gaussian-sampled brownian motion

    if n_steps are supplied, the brownian motion vector is generated for the
    entire set of forward steps. Otherwise, it is only prepared for the step
    given.
    """

    if n_steps:
        return torch.randn(n_steps, x.shape[0], x.shape[1], requires_grad=True) * stdev
    else:
        return torch.randn(x.shape[0], x.shape[1], requires_grad=True) * stdev


def _timespan(t):
    if (type(t) == torch.Tensor) or (type(t) == np.ndarray):
        return (t.max() - t.min()).item()
    elif type(t) == list:
        return max(t) - min(t)


def _manual_forward_step(net, x, dt, stdev, tspan, device):

    n_steps = int(tspan / dt)
    z = _brownian_motion(x, stdev, n_steps=n_steps)
    x_hat = x
    for step in range(n_steps):
        x_hat = _forward_step(net, x_hat, dt, z[step].to(device))
    return x_hat


def _manual_forward(func, x0, t, dt, stdev, device):

    tspan = _timespan(t)
    n_timesteps = int(tspan / dt)

    x_step, x_forward = x0, [x0]
    for step in range(n_timesteps):
        x_step = _manual_forward_step(func, x_step, dt, stdev, tspan, device)
        x_forward.append(x_step)

    return torch.stack(x_forward)

# ----------------------------------------------------------------------------------------


# Currently implemented credentialing flags (2): -----------------------------------------
NEURAL_SDE_FLAGS = {
    "using_neural_diffeq": True,
    "integrator": torchsde.sdeint,
    "require_dt": True,
    "use_time_scalar": False,
    "use_stdev": False,
}
NEURAL_ODE_FLAGS = {
    "using_neural_diffeq": True,
    "integrator": torchdiffeq.odeint,
    "require_dt": False,
    "use_time_scalar": True,
    "use_stdev": False,
}
MANUAL_FORWARD_FLAGS = {
    "using_neural_diffeq": False,
    "integrator": _manual_forward,
    "require_dt": False,
    "use_time_scalar": False,
    "use_stdev": True,
}

# ----------------------------------------------------------------------------------------


# Credentialing functions (3): -----------------------------------------------------------
def _SDE_credentials(func):

    """Returns boolean indicator for SDE-passing credentials."""
    return hasattr(func, "mu") and hasattr(func, "sigma")


def _ODE_credentials(func):
    return hasattr(func, "mu")


def _manual_forward_credentials(func):
    return True


def _set_attributes(CLASS, FLAGS):
    for key, value in FLAGS.items():
        CLASS.__setattr__(key, value)


def _drift_net_credentials(net):

    """Validates μ(x)/f(x) credentials only"""

    POTENTIAL_NET = list(net.parameters())[-1].data.numel() == 1
    if POTENTIAL_NET:
        FORWARD_NET = False
    else:
        FORWARD_NET = True
    return {"using_potential_net": POTENTIAL_NET, "using_forward_net": FORWARD_NET}


def _register_credentials(ForwardFunctionsClass):

    self = ForwardFunctionsClass

    if _SDE_credentials(self.func):
        _set_attributes(self, NEURAL_SDE_FLAGS)
    elif _ODE_credentials(self.func):
        _set_attributes(self, NEURAL_ODE_FLAGS)
    elif _manual_forward_credentials(self.func):
        _set_attributes(self, MANUAL_FORWARD_FLAGS)

    if not self.using_neural_diffeq:
        mu = self.func
    else:
        mu = self.func.mu

    _set_attributes(self, _drift_net_credentials(mu))

# ----------------------------------------------------------------------------------------


# Register integrator **kwargs (4): ------------------------------------------------------
def _register_integrator_kwargs(integrator, t, require_dt, **kwargs):

    """primarily used to toggle dt but gives an interface to add other sdeint/odeint kwargs"""

    if integrator.__name__ == "sdeint":
        kwarg_dict = {"ts": t}
    else:
        kwarg_dict = {"t": t}

    for key, value in kwargs.items():
        if value:
            if key == "dt":
                kwarg_dict[key] = value
            elif key == "device":
                if not integrator.__name__ in ["sdeint", "odeint"]:
                    kwarg_dict[key] = kwargs[key]
            else:
                kwarg_dict[key] = value

    return kwarg_dict

# ----------------------------------------------------------------------------------------


# Main module class: ForwardFunctions (5): -----------------------------------------------
class ForwardFunctions:
    def __init__(self, func, time_scale=None, stdev=None, dt=None, device=None):
        
        
        """
        Parameters:
        -----------
        func [torch.nn.Module, neural_diffeq]
        
        time_scale
        
        stdev
        
        dt
        
        device
        """

        # format func credentials
        self.func = func
        _register_credentials(self)

        if self.use_time_scalar:
            self._time_scale = time_scale

        if self.use_stdev:
            self._stdev = stdev
        else:
            self._stdev = None

        self._dt = dt
        self._device = device

    def __call__(self, func, X0, t):

        # format t and kwargs
        if self.use_time_scalar:
            t = pydk.min_max_normalize(t) * self._time_scale
        tspan = _timespan(t)

        kwargs = _register_integrator_kwargs(
            self.integrator,
            t,
            self.require_dt,
            **{"dt": self._dt, "stdev": self._stdev, "device": self._device}
        )

        # do integration
        return self.integrator(func, X0, **kwargs)

# ----------------------------------------------------------------------------------------