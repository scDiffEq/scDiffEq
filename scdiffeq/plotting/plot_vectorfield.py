from .plotting_presets import single_fig_presets as presets
from ..utilities.torch_device import set_device

import torch
import numpy as np
from torchdiffeq import odeint
import matplotlib.pyplot as plt

def get_mgrid_dydt(odefunc):

    """Pass a mesh grid through the ODEFunc to obtain a vectorfield representing the transformation of the space once passed through the learned ODE."""

    device = set_device()

    y, x = np.mgrid[-2:2:21j, -2:2:21j]
    dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device))
    dydt_cpu = dydt.cpu().detach().numpy()
        
        
    mag = np.sqrt(dydt_cpu[:, 0] ** 2 + dydt_cpu[:, 1] ** 2).reshape(-1, 1)
    dydt_cpu = dydt_cpu / mag
    dydt_cpu = dydt_cpu.reshape(21, 21, 2)

    return x, y, dydt_cpu


def plot_vectorfield(adata, savename):
    
    presets(title="Learned Vector Field", x_lab="$X$", y_lab="$Y$")
    x, y, dydt = get_mgrid_dydt(adata.uns["odefunc"])
    plt.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="midnightblue")
    
    if savename != None:
        plt.savefig(savename + ".png")
    plt.show()