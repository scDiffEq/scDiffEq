
import numpy as np
import torch

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

    if not type(stdev) == float:
        stdev = stdev.item()

    if n_steps:
        return torch.randn(n_steps, x.shape[0], x.shape[1], requires_grad=False)*stdev
    else:
        return torch.randn(x.shape[0], x.shape[1], requires_grad=True) * stdev
    
def _manual_forward_step(net, x, dt, stdev, tspan, device):
    
    n_steps = int(tspan / dt)
    z = _brownian_motion(x, stdev, n_steps=n_steps).to(device)
    
    x_hat = x
    for step in range(n_steps):
        x_hat = _forward_step(net, x_hat, dt, z[step])
    return x_hat