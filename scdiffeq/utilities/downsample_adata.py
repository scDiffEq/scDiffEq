
from . import general_utility_functions as util
from ..plotting.plotting_presets import presets_for_plotting_multiple_trajectories 
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def plot_downsampled(adata, title="Simulated Data", figsize=(10, 8)):

    """
    Plots downsampled data (assuming 2-D). Higher-D downsample visualization has yet to be implemented.
    """

    figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(1, 1, 1)

    trajs = adata.obs.trajectory.unique()

    for i in trajs:

        traj = adata.obs.loc[adata.obs.trajectory == i].index.values.astype(int)
        data = adata.X[traj]

        presets_for_plotting_multiple_trajectories(
            ax, title, data[:, 0], data[:, 1], "x", "y",
        )

def downsample_adata(adata, percentage_of_data=1, random=False, plot=False):
    
    """
    Returns a fraction of the original adata object.
    
    Parameters:
    ----------
    adata
        AnnData object
        
    percentage_of_data
        Percentage of data to be retained in downsampling
        
    random
        (boolean) If True, random points within trajectories are chosen. Otherwise, evenly sampled. Default False.
        
    plot
        (boolean) If true, plots the downsampled data (only 2-D supporting at this time).
        default: False

    Returns:
    --------
    
    adata
    """
    
    # make sure data is not in a sparse matrix 
    util.ensure_array(adata)
    
    array = adata.obs.index.values
    data_size = adata.shape[0]
    
    if random==True:
        downsample = np.random.choice(
            array, size=int(round(data_size * percentage_of_data)), replace=False
        )
        # perform the downsampling
        adata = adata[downsample]
    
    else:
        # samples from within the traj. essentially thinning time points across a traj. 
        downsample_size = int(round(data_size * percentage_of_data))
        interval = int(round(data_size / downsample_size))
        downsampled = adata.obs.sort_values("trajectory")[::interval].index.astype(int)
        adata = adata[downsampled]
        
    adata.obs.reset_index(
        inplace=True, drop=True,
    )
    adata.obs.index = adata.obs.index.astype("str")
    adata.obs = adata.obs.sort_values("time")
    print(adata)
    
    if plot==True:
        plot_downsampled(adata)
    
    return adata