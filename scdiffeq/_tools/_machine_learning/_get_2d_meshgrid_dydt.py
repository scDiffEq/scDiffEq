import numpy as np
import torch
import vintools as v

def _make_2d_meshgrid(data, bins=25, dim_1="x", dim_2="y"):

    """
    Parameters:
    -----------


    """

    bins_complex = complex(0, bins)

    DataBounds = v.ut.get_data_bounds(data_2d=data, dim_1=dim_1, dim_2=dim_2)

    x, y = np.mgrid[
        DataBounds[dim_1]["min"] : DataBounds[dim_1]["max"] : bins_complex,
        DataBounds[dim_2]["min"] : DataBounds[dim_2]["max"] : bins_complex,
    ]

    mgrid = torch.Tensor(np.stack([x, y], -1).reshape(bins * bins, 2))

    return mgrid, x, y

def _get_2d_meshgrid_dydt(data, ODE, bins=25):

    """Pass a mesh grid through the ODEFunc to obtain a vectorfield representing the transformation of the space once passed through the learned ODE."""

    mgrid, x, y = _make_2d_meshgrid(data, bins=bins)
    dydt_mgrid = ODE(0, mgrid).detach().numpy()
    velocity_mag = np.sqrt(dydt_mgrid[:, 0] ** 2 + dydt_mgrid[:, 1] ** 2).reshape(-1, 1)
    dydt_mgrid_mag_adjusted = (dydt_mgrid / velocity_mag).reshape(bins, bins, 2)

    return x, y, dydt_mgrid_mag_adjusted, velocity_mag.reshape(bins, bins)