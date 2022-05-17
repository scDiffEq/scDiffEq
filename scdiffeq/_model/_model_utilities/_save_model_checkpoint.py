

import torch


def _save_model_checkpoint(func, epoch, outdir, silent=True):

    """
    Save model during training.

    Parameters:
    -----------
    func
        type: Neural_Differential_Equation

    epoch
        type: int

    outdir
        type: str

    silent
        default: True
        type: bool

    Returns:
    --------
    None, saves model and [ optionally ] prints a report of this.

    """

    state = func.state_dict()
    save_path = "{}/model/{}_epochs.model".format(outdir, epoch)

    torch.save(state, save_path)

    if not silent:
        print("Saved model to: {}".format(save_path))