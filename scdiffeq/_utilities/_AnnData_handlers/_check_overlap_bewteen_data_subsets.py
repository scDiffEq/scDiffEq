# package imports #
# --------------- #
import licorice
import numpy as np

def _check_overlap_bewteen_data_subsets(adata, subsets=["train", "validation", "test"]):

    """
    This function checks between data subsets within AnnData to see if there is any overlap between groups.
    
    Parameters:
    -----------
    adata
        AnnData
        
    subsets
        default: ['train', 'validation', 'test']
        type: list of str
        
    Returns:
    --------
    prints n_overlap. Desired: 0
    """

    df = adata.obs

    overlaps = []
    print("Checking between...\n")
    for subset in subsets:
        for subset_check_against in subsets:
            if subset != subset_check_against:
                print("\t{:<11} <-- --> {:>11}".format(subset, subset_check_against))
                overlaps.append(
                    df.loc[df[subset] == True]
                    .loc[df[subset_check_against] == True]
                    .shape[0]
                )
    n_overlap = np.array(overlaps).sum()
    print(
        "\nIdentified {} overlapping (leaking) datapoints.".format(
            licorice.font_format(str(n_overlap), ["BOLD", "RED"])
        )
    )
