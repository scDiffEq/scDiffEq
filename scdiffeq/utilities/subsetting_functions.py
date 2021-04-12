
import numpy as np
import pandas as pd

"""
The functions in this module conveniently subset other data objects used by this package. 
They are useful internally but also available to the user.
"""

def subset_adata(
    adata,
    subset_specification,
    include_time=True,
    time_name="latent_time",
    in_place=False,
):

    """
    AnnData.obs[subset_specification] uses a boolean indicator of [True / False] if the data exists within a certain subset of interest.
    Parameters:
    -----------
    adata
      AnnData object
    subset_specification
        string. adata.obs_key. key value for the adata.obs dataframe.

    include_time
        boolean. if True, includes the time component

    in_place
        boolean. if True, adds a key to adata.uns for the data subset. default=False.
    Returns:
    --------
    subset
        object class containing:
            subset.obs
            subset.index
            subset.data
            subset.t (optional, included by default)
            subset.obsm
    adata (optional, default=False)
        adata.uns[subset_specfication]
    """

    try:
        subset_obs = adata.obs.loc[adata.obs[subset_specification] == True]
    except:
        subset_obs = None

    import pandas as pd

    if type(subset_obs) != pd.core.frame.DataFrame:

        subset_data = adata.uns[subset_specification]
        subset_index = subset_data.index.to_numpy(dtype=int)
        time = adata.uns[subset_specification.replace("data", "time")]

    else:
        subset_index = subset_obs.index.to_numpy(dtype=int)
        subset_data = adata.X[subset_index]

        if include_time == True:
            time = subset_obs[time_name]
        if in_place == True:
            adata.uns[subset_specification] = subset_data

    try:
        subset_emb = adata.obsm["X_pca"][subset_index]
    except:
        subset_emb = None

    class _subset:
        def __init__(self, subset_obs, subset_index, subset_data, time, subset_emb):

            self.obs = subset_obs
            self.index = subset_index
            self.data = subset_data
            self.t = time
            self.emb = subset_emb

    subset = _subset(subset_obs, subset_index, subset_data, time, subset_emb)

    return subset


def randomly_subset_trajectories(adata, set_of_trajectories, subset):

    """
    Parameters:
    -----------
    adata
        AnnData object

    set_of_trajectories
        The set from which the subset is drawn at random.

    subset
        Proportion of the original set to be drawn.

    Returns:
    --------
    random_subset_of_trajectories
        Sampled from the original set of trajectories

    """

    size = int(round(subset * len(set_of_trajectories)))
    random_subset_of_trajectories = np.sort(
        np.random.choice(set_of_trajectories, size=size, replace=False)
    )

    return random_subset_of_trajectories