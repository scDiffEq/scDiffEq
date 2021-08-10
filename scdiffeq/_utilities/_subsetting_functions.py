
import numpy as np
import pandas as pd

"""
The functions in this module conveniently subset other data objects used by this package. 
They are useful internally but also available to the user.
"""

def _group_adata_subset(
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


def _randomly_subset_trajectories(adata, set_of_trajectories, subset):

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
    
    set_of_trajectories = adata.obs.trajectory.unique()

    size = int(round(subset * len(set_of_trajectories)))
    random_subset_of_trajectories = np.sort(
        np.random.choice(set_of_trajectories, size=size, replace=False)
    )

    return random_subset_of_trajectories

def _isolate_trajectory(data_object, trajectory_number):

    """
    Get the entire obs df for a given trajectory.
    
    Parameters:
    -----------
    data_object
        assumes the the object contains data_object.obs
    
    trajectory_number
        number that is assumed to be present within data_object.obs.trajectory
        
    Returns:
    --------
    df
        data_object.obs dataframe subset for only the trajectory queried
    """

    df = data_object.obs.loc[data_object.obs.trajectory == trajectory_number]

    return df


def _check_df(df, column, subset):

    assert column in df.columns, "Column not found in df."
#     assert subset in df[column].unique().astype(type(subset)), "Subset value not found in selected column."

def _get_subset_idx(df, column, subset=True, return_inverse=False):

    """
    Parameters:
    -----------
    df
        pandas DataFrame

    column
        Column / key of pandas DataFrame

    subset [optional, default: True]
        Data contained within a selected column of a pandas DataFrame. Criteria by which
        subset is determined. If undefined by user, a True / False boolean mask selection
        will be used where subset == True.

    return_inverse [optional, default: False]
        if True, inverse df is returned along with df in a tuple.

    Returns:
    --------
    [subset_idx] or [subset_idx, inverse_idx]
    """

    _check_df(df, column, subset)

    subset_idx = df.loc[df[column].astype(type(subset)) == subset].index.astype(int)
    try:
        inverse_idx = df.loc[dfs[column].astype(type(subset)) == subset].index.astype(int)
    except:
        inverse_idx = df.loc[df[column].astype(type(subset)) != subset].index.astype(int)

    if return_inverse == True:
        return [subset_idx, inverse_idx]
    return subset_idx


def _subset_df(df, column, subset=None, return_inverse=False, return_idx=False):

    """
    Parameters:
    -----------
    df
        pandas DataFrame

    column
        Column / key of pandas DataFrame

    subset [optional, default: True]
        Data contained within a selected column of a pandas DataFrame. Criteria by which
        subset is determined. If undefined by user, a True / False boolean mask selection
        will be used where subset == True.


    return_inverse [optional, default: False]
        if True, inverse df is returned along with df in a tuple.

    return_idx [optional, default: False]
        if True, index or indices are returned along with df(s)

    Returns:
    --------
    [df_subset] or [df_subset, df_inverse] or [df_subset, df_inverse, idx*]

    *idx may be a tuple of indices
    """

    _check_df(df, column, subset)

    try:
        df_subset = df.loc[df[column] == subset]
        df_inverse = df.loc[~df[column] == subset]
    except:
        df_inverse = df.loc[df[column] != subset]

    if return_idx == True:
        idx = _get_subset_idx(df, column, subset=subset, return_inverse=return_inverse)
        if return_inverse == True:
            return [df_subset, df_inverse, idx]
        return [df_subset, idx]
    if return_inverse == True:
        return [df_subset, df_inverse]
    return df_subset


def _subset_adata(
    adata,
    obs_key,
    subset=True,
    return_inverse=False,
    return_obs=False,
    return_idx=False,
    print_statement=False,
):

    """
    Parameters:
    --------
    adata
        AnnData object
        
    obs_key
        Column / key of pandas DataFrame stored as adata.obs[obs_key]

    subset [optional, default: True]
        Data contained within a selected column of the adata.obs[obs_key] pandas 
        DataFrame. Criteria by which subset is determined. If undefined by user,
        a True / False boolean mask selection will be used where subset == True.

    return_inverse [optional, default: False]
        if True, inverse adata, [df, index] is returned.
        
    return_obs [optional, default: False]
         if True, adata.obs pandas DataFrame is returned.
         
    return_idx [optional, default: False]
        if True, index or indices are returned along with adata and/or adata.obs DataFrame. 

    Returns:
    --------
    Any combination / selection of the following list (output if list is len(list) > 1):

        [adata_subset, df_subset, idx_subset,
         adata_inverse, df_inverse, idx_inverse]

        type: AnnData, pandas DataFrame, Index, or list
    """

    _subset_obs = _subset_df(
        df=adata.obs,
        column=obs_key,
        subset=subset,
        return_inverse=True,
        return_idx=True,
    )
    # return_index set to True here because we need it to subset adata

    # subset_obs[0] = adata.obs_subset
    # subset_obs[1] = adata.obs_inverse
    # subset_obs[2][0] = subset_idx
    # subset_obs[2][1] = inverse_idx

    # perform subset of AnnData and separate obs and inverse DataFrames as well as obs and inverse indices
    subset_adata, inverse_adata = adata[_subset_obs[2][0]], adata[_subset_obs[2][1]]
    if print_statement == True:
        print("Subset AnnData:")
        print(subset_adata)
    subset_obs, inverse_obs = _subset_obs[0], _subset_obs[1]
    subset_idx, inverse_idx = _subset_obs[2][0], _subset_obs[2][1]

    all_true = [
        subset_adata,
        subset_obs,
        subset_idx,
        inverse_adata,
        inverse_obs,
        inverse_idx,
    ]
    
#     # if it's just one item (i.e., only returning adata), no need to return as a list
#     if np.all([return_inverse, return_obs, return_idx]) == False:
#         return subset_adata

    if return_inverse:
        return [
            all_true[i]
            for i in np.where(
                np.tile([return_inverse, return_obs, return_idx], 2) == True
            )[0]
        ]

    return [
        all_true[i]
        for i in np.where(
            np.array([True, return_obs, return_idx, False, False, False]) == True
        )[0]
    ]