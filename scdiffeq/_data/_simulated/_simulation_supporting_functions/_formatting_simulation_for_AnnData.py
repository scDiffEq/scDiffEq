# package imports #
# --------------- #
import numpy as np
import anndata


def _format_simulation_data(simulated_trajectories):

    """
    Prepares simulation data for both plotting as well as formatting with AnnData.
    """

    n_traj = len(simulated_trajectories)
    n_obs_one_traj = simulated_trajectories[0].shape[0]
    n_var = simulated_trajectories[0].shape[1]

    X = np.zeros([n_traj, n_obs_one_traj, n_var])

    for n, trajectory in enumerate(simulated_trajectories):
        X[n] = trajectory

    return X


def _simulation_to_AnnData(self, silent=False):

    """"""

    self.X = _format_simulation_data(self.simulated_trajectories)

    X = self.X.reshape(-1, 2)
    adata = anndata.AnnData(X)
    adata.obs["time"] = np.tile(self.time_vector, self.n_traj)
    adata.obs["trajectory"] = np.repeat(range(self.n_traj), len(self.time_vector))

    if not silent:
        print("\n", adata)

    return adata
