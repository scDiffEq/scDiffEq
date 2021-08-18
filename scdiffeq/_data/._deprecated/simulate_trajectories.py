import numpy as np
import pandas as pd
import anndata as a
from scipy.integrate import odeint
from .supporting_functions._get_initial_conditions import get_initial_conditions
from .supporting_functions._construct_time_vector import construct_time_vector
from .supporting_functions.plot_simulated_trajectories import (
    plot_simulated_trajectories,
)


def simulate_trajectories(
    state_eqn,
    ic_type=None,
    distributions=["gaussian", "gaussian"],
    time_span=10,
    number_time_samples=50,
    number_trajectories=20,
    variance=0.15,
):

    """
    Simulate single-cell data given a set of state equations. 
    
    Parameters
    ----------
    state_eqn
        a set of ODEs (requiured)
        
    distributions
        define the distribution type for each simulated axis/gene/dimension. one must be chose for every dimension. Currently supported options are: gaussian, random, axis-locked-negative, axis-locked-positive, and axis-zeroed. 
    
    time_span
        the number of time units over which one wishes to simulate. 
        
    number_time_samples
        the number of samples to be taken within the defined time_span.
    
    number_trajectories
        integer. number of cell trajectories/paths to be simulated.
        
    variance
        variance specified, if the gaussian function should be specified. 

    returns:
    --------
    AnnData object of the simulation results.   
    """

    if ic_type == "meshgrid":
        x = np.arange(-1, 1, 0.1)
        y = np.arange(-1, 1, 0.1)
        initial_conditions = np.array(np.meshgrid(x, y)).T.reshape(400, 2)
        number_dimensions = 2
        number_trajectories = 400
    else:
        initial_conditions = get_initial_conditions(distributions, number_trajectories)
        number_dimensions = len(distributions)

    # make empty adata.X
    all_points = np.array([])

    # make empty adata.obs
    time_all_cells = np.array([])
    trajectory_count = np.array([])
    all_timepoints = np.array([])

    for count, i in enumerate(range(len(initial_conditions))):

        time = construct_time_vector(1, time_span, number_time_samples)
        state0 = initial_conditions[i]
        trajectory = odeint(state_eqn, state0, time)
        all_points = np.append(all_points, trajectory)
        time_all_cells = np.append(time_all_cells, time)
        all_timepoints = np.append(all_timepoints, np.arange(0, number_time_samples))
        trajectory_count = np.append(
            trajectory_count, np.full((number_time_samples), int(count))
        )

    simulated_data = all_points.reshape(
        initial_conditions.shape[0], number_time_samples, initial_conditions.shape[1]
    )

    adata = a.AnnData(
        simulated_data.reshape(simulated_data.shape[0] * simulated_data.shape[1], 2)
    )

    adata.uns["number_of_timepoints"] = number_time_samples
    adata.uns["number_of_trajectories"] = number_trajectories
    adata.obs["time"] = time_all_cells
    adata.obs["trajectory"] = trajectory_count
    adata.obs["timepoint"] = pd.Categorical(all_timepoints)

    if simulated_data.shape[2] <= 2:
        plot_simulated_trajectories(adata)

    print(adata)

    return adata
