
import numpy as np
from ..._utilities._subsetting_functions import _group_adata_subset
from ..._utilities._subsetting_functions import _randomly_subset_trajectories
from ..._utilities._AnnData_handlers._format_AnnData import _format_AnnData_mtx_as_numpy_array
from ._forward_integration_functions._format_trajectories_into_batches import _check_overlap_bewteen_data_subsets

"""
This module contains the forward-facing function for splitting a dataset into test, train, and validation
by trajectory as well as all fuctions supporting it. The functions imported above are more universal and 
thereby contained in a seperate module. 
"""

def _calculate_train_validation_percentages(proportion_train, proportion_validation):

    """
    Calculates the relative amount of the dataset reserved for training and validation.

    Parameters:
    -----------
    proportion_training
        Relative amount of trajectories dedicated to training
        default: 0.60

    proportion_validation
        Relative amount of trajectories dedicated to testing
        default: 0.20

    Returns:
    --------
    train_validation_percentage


    proportion_validation

    """

    train_validation_percentage = proportion_train + proportion_validation
    proportion_validation = proportion_validation / train_validation_percentage

    return train_validation_percentage, proportion_validation


def _test_train_split_by_trajectory(
    adata,
    proportion_train,
    proportion_validation,
    return_data_subsets,
    trajectory_column,
    time_column,
):

    """
    This is the key function to assigning data to test, train, and validation sets.

    Parameters:
    -----------

    adata
        AnnData object

    proportion_train
        Relative amount of trajectories dedicated to training
        default: 0.60

    proportion_validation
        Relative amount of trajectories dedicated to validation
        default: 0.20

    return_data_subsets
        boolean.
        default: True

    trajectory_column
        Column name for trajectories in adata.obs
        default:"trajectory"

    time_column
        Column name for time in adata.obs
        default:"time"

    Returns:
    --------
    training, validation, test
        groups to be passed to the user-facing function and encapsulated in a single data_object class.

    """

    # get all unique trajectories in the dataset
    all_trajectories = adata.obs[trajectory_column].unique()

    (
        train_validation_percentage,
        proportion_validation,
    ) = _calculate_train_validation_percentages(
        proportion_train, proportion_validation
    )

    # employ the randomly_subset_trajectories function twice to get trajectories for
    # first, training and validation, and then subset randomly from within that set
    # to get just the validation trajectories.
    training_and_validation_trajectories = _randomly_subset_trajectories(
        adata, all_trajectories, train_validation_percentage
    )
#     validation_trajectories = _randomly_subset_trajectories(
#         adata, training_and_validation_trajectories, proportion_validation
#     )
    
    training_and_validation_trajectories = _randomly_subset_trajectories(
        adata, all_trajectories, train_validation_percentage
    )
    size_validation = int(round(proportion_validation * len(training_and_validation_trajectories)))
    validation_trajectories = np.sort(
        np.random.choice(training_and_validation_trajectories, size=size_validation, replace=False)
    )


    # get training trajectories from the first subset by removing trajectories in the
    # validation subset
    training_trajectories = training_and_validation_trajectories[
        ~np.isin(training_and_validation_trajectories, validation_trajectories)
    ]

    # get test trajectories from the whole set of trajectories by removing trajectories
    # in the training and validation subset
    test_trajectories = all_trajectories[
        ~np.isin(all_trajectories, training_and_validation_trajectories)
    ]

    # add boolean columns to adata.obs to indicate which group (test, training, or validation)
    # each trajectory and point within belongs to
    adata.obs["train"] = adata.obs[trajectory_column].isin(training_trajectories)
    adata.obs["validation"] = adata.obs[trajectory_column].isin(validation_trajectories)
    adata.obs["test"] = adata.obs[trajectory_column].isin(test_trajectories)

    adata.obs.reset_index(drop=True, inplace=True)

    if return_data_subsets:

        train = _group_adata_subset(adata, "train", time_name=time_column)
        validation = _group_adata_subset(adata, "validation", time_name=time_column)
        test = _group_adata_subset(adata, "test", time_name=time_column)

        return train, validation, test


class _data_splitting:

    """
    Class for organizing the split data into a single object.
    """

    def __init__(self, train, validation, test):

        train.obs.reset_index(drop=True, inplace=True)
        validation.obs.reset_index(drop=True, inplace=True)
        test.obs.reset_index(drop=True, inplace=True)

        self.train = train
        self.validation = validation
        self.test = test

def _split_test_train(
    adata,
    trajectory_column="trajectory",
    proportion_train=0.60,
    proportion_validation=0.20,
    return_data_subsets=True,
    time_column="time",
    silent=False,
    return_split_data=False,):

    """
    This is the user-facing function to split data into testing, training, and validation sets.

    Parameters:
    -----------

    adata
        AnnData object

    trajectory_column
        Column name for trajectories in adata.obs
        default:"trajectory"

    proportion_training
        Relative amount of trajectories dedicated to training
        default: 0.60

    proportion_validation
        Relative amount of trajectories dedicated to validation
        default: 0.20

    return_data_subsets
        boolean.
        default: True

    time_column
        Column name for time in adata.obs
        default:"time"

    Returns:
    --------

    data_object
        data class with three subsets of data: train, test, and validation

    """
    
    try:
        adata.obs["trajectory"]

    except:
        adata.obs["trajectory"] = 0

    # ensure the data is not in sparse format
    _format_AnnData_mtx_as_numpy_array(adata, silent=silent)

    #
    train, validation, test = _test_train_split_by_trajectory(
        adata,
        proportion_train=proportion_train,
        proportion_validation=proportion_validation,
        return_data_subsets=True,
        trajectory_column=trajectory_column,
        time_column=time_column,
    )

    split_data = _data_splitting(train, validation, test)
    adata.uns["data_split_keys"] = {
        "test": test,
        "train": train,
        "validation": validation,
    }
    print("\nChecking for overlap between test, train, and validation subsets...\n")
    _check_overlap_bewteen_data_subsets(adata)
    adata.uns['split_data'] = split_data
    
    if return_split_data:
        return split_data