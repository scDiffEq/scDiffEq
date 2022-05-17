__module_name__ = "_scDiffEq_Model.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import torch


def _add_augmentation_dimensions(X, augment_dim, device):

    """Assumes shape of t x cells x dim"""

    augmentation = torch.zeros(X.shape[0], X.shape[1], augment_dim).to(device)
    return torch.cat([X, augmentation], -1)


def _augment_NDE(X, augment_dim=0, device="cpu"):

    """
    Add additional null input dimensions to your data.
    Parameters:
    -----------
    X
        Unmodified input data.
    augment_dim
        default: 0
    device
        default: "cpu"
    Returns:
    --------
    X_aug
        Augmented data of shape: [N,M,D+a]
        type: torch.Tensor
    Notes:
    ------
    (1) This implementation ignores the case of convolutional NNs.
    Source:
    -------
    https://github.com/EmilienDupont/augmented-neural-odes/blob/88d9865fa863069fa65e538ee11eebea934dbbe4/anode/models.py#L135-L149
    """

    if augment_dim is 0:
        return X
    else:
        return _add_augmentation_dimensions(X, augment_dim, device)
