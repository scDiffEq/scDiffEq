

# _split_adata.py
__module_name__ = "_split_adata.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# package imports #
# --------------- #
import numpy as np


from ...._utilities._Messages_Module import _Messages_
messages = _Messages_()


def _shuffle_assignment_indices(DataSplitDict, idx):

    """
    SplitData
        Dictionary defined by parent class. contains proportions of test, train, validation to be applied to the
        dataset in this function.

    idx
        This parameter may be anything by which you wish the data to be split (trajectories, cells, etc.).
        Determination of this input is established upstream of this function (e.g., getting all unique trajectories.)

    """

    shuffled_indices = np.random.choice(idx, size=len(idx), replace=False)

    index_counter = 0
    for data_group in DataSplitDict["proportions"].keys():
        n_samples_data_group = int(len(idx) * DataSplitDict["proportions"][data_group])
        DataSplitDict["{}_idx".format(data_group)] = shuffled_indices[
            index_counter : index_counter + n_samples_data_group
        ]
        index_counter += n_samples_data_group

    return DataSplitDict


def _annotate_adata_by_DataSplit(adata, DataSplitDict, splitby_key):

    """"""

    for key in DataSplitDict["proportions"].keys():
        if key is "proportions":
            continue
        else:
            adata.obs[key] = adata.obs[splitby_key].isin(
                DataSplitDict["{}_idx".format(key)]
            )
    return adata


class SplitData:
    def __init__(
        self, adata, train_proportion, valid_proportion, test_proportion
    ):
        """This class encapsulates two functions for splitting adata."""

        self.DataSplitDict = {}
        self.DataSplitDict["proportions"] = {
            "train": train_proportion,
            "valid": valid_proportion,
            "test": test_proportion,
        }

        self.adata = adata
        self.n_samples = adata.shape[0]

    def by_trajectories(self, splitby_key="trajectory"):

        """"""
        unique_indexer = self.adata.obs[splitby_key].unique()

        self.DataSplitDict = _shuffle_assignment_indices(
            self.DataSplitDict, unique_indexer
        )

        self.adata = _annotate_adata_by_DataSplit(
            self.adata, self.DataSplitDict, splitby_key
        )
        
def _split_adata(
    adata,
    hyper_parameters,
    preferences,
    overfit,
):
    
    """

    Notes:
    ------
    (1) This function takes the strategy of annotating test/train/validation group assignments within
        the `adata.obs` table. 
    """
    
    single_trajectory = adata.obs["trajectory"].nunique() == 1
    
    if single_trajectory or overfit:
        hyper_parameters.train_proportion = 1
        perform_validation = False
        if not single_trajectory:
            messages.overfit()
        else:
            messages.single_trajectory()
    else:
        perform_validation = True
    
    split = SplitData(adata, 
                      hyper_parameters.train_proportion, 
                      hyper_parameters.valid_proportion, 
                      hyper_parameters.test_proportion)
    
    if hyper_parameters.learn_by == "trajectory":
        split.by_trajectories(splitby_key="trajectory")
        
    if not preferences.silent:
        print(adata)
    
    return split.adata, perform_validation