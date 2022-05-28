
__module_name__ = "_progress_bar.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
from tqdm.notebook import tqdm as tqdm_nb
from tqdm import tqdm as tqdm_cli


def _tqdm(notebook):

    if notebook:
        return tqdm_nb
    else:
        return tqdm_cli


def _progress_bar(TrainingProgram, epoch_count):
    
    """
    Return an epoch-ranged progress bar wrapped by tqdm (notebook or cli version).
    
    Parameters:
    -----------
    TrainingProgram [ required ]
        Dictionary of hyper-parameters dictating the programmed training plan. The keys important to this
        function are "epochs" and "notebook".
        type: dict
    
    epoch_count [ required ]
        Eclipsed epochs
        type: int
    
    Returns:
    --------
    progress_bar
        tqdm progress bar for the specified epoch range
        type: tqdm.notebook.tqdm_notebook
    
    Notes:
    ------
    """

    tqdm = _tqdm(TrainingProgram["notebook"])

    """add the epoch count to the epoch training range"""

    start_epoch = epoch_count + 1
    final_epoch = TrainingProgram["epochs"] + 1

    return tqdm(range(start_epoch, final_epoch))