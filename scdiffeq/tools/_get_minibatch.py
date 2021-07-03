import numpy as np
import pandas as pd
from ..utilities._torch_device import _torch_device as device


def _get_obs(adata, item):

    obs = adata.obs.loc[adata.obs.trajectory == item]

    return obs


def _format_trajectory(data_object, traj_num):

    obs = get_obs(data_object, traj_num)
    idx = obs.index.values.astype(int)

    t = obs.time.unique()
    y = data_object.data[idx]
    y0 = y[0]
    emb = data_object.emb[idx]

    class make_formatted_traj:
        def __init__(self, y0, y, t, emb):

            self.y0 = device(y0)
            self.y = device(y)
            self.t = device(pd.Series(np.sort(t))) # unique()
            self.emb = device(emb)

    formatted_trajectory = make_formatted_traj(y0, y, t, emb)

    return formatted_trajectory


def _get_minibatch(data_object):

    """"""
    
    batch_size = data_object.obs.trajectory.nunique()    
    
    unique_trajectories = data_object.obs.trajectory.unique()
    batch_trajectories = np.sort(
        np.random.choice(unique_trajectories, batch_size, replace=False)
    )

    batch = []

    for i in batch_trajectories:

        formatted_trajectory = format_trajectory(data_object, i)
        batch.append(formatted_trajectory)

    return batch