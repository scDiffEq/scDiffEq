import numpy as np
import pandas as pd
import scipy
from .EMT_funcs import EMT_dydt, parameterize
from .utils import data_to_anndata, record_trajectory, remove_nan

def EMT(kmrgd, t, parameters):

    # EMT functions are supplied in a supplementary script
    miR200, mZ, miR34, mS, i2 = kmrgd[0], kmrgd[1], kmrgd[2], kmrgd[3], kmrgd[4]
    dydt = EMT_dydt(kmrgd, t, parameters)

    return dydt

def simulate_emt(y0, t, variance):
    
    "time and column names are predefined for this simulation. just provide y0 and variance, if chosen."
    
#     t = np.arange(0, 7200, 1)
    colnames = ["time", "miR200", "mZEB", "miR34", "mSNAIL", "i2"]
    parameters = parameterize(variance)

    df = pd.DataFrame(
        np.hstack([t[:, np.newaxis], scipy.integrate.odeint(EMT, y0, t, (parameters,))]),
        columns=colnames,
    )

    return df

def simulate_iteratively(
    initial_conditions, cols, parameter_variance=None, time_length=7200, time_scale=7200
):

    """
    time is generated randomly within a set of time constraints for each trajectory such that each cell within each trajectory has it's own unique timestamp, however runs along the same timeline.
    """

    trajectories = np.array([])
    y_trajectories = np.array([])
    time_stamps = np.array([])

    for i, y0 in enumerate(initial_conditions):

        t = np.sort(
            np.random.rand(time_length) * 7200
        )  # 7200 is the default time scale required to properly simulate the trajectory
        time_stamps = np.append(time_stamps, t)

        y = simulate_emt(y0, t, parameter_variance)

        trajectories = record_trajectory(
            trajectories, time_length=time_length, trajectory_number=i
        )
        y_trajectories = np.append(y_trajectories, y)

    data_ = y_trajectories.reshape(
        (time_length * initial_conditions.shape[0]), (initial_conditions.shape[1] + 1),
    )

    data_, nan_rows = remove_nan(data_)

    trajectories = np.delete(trajectories, nan_rows)

    data = pd.DataFrame(data_, columns=cols,)

    adata = data_to_anndata(data, trajectories, cols, time_length)

    return adata