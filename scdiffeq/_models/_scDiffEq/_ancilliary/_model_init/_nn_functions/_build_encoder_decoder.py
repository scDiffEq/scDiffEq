
__module_name__ = "_build_encoder_decoder.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import licorice_font
import numpy as np
import torch


# import local dependencies #
# ------------------------- #
from ._compose_nn_sequential import _compose_nn_sequential
from ._no_transform import _no_transform

def _power_space(start, stop, n, power):

    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))

    return np.power(np.linspace(start, stop, num=n), power)


def _get_sequenced_VAE_layer_n(data_dim, latent_dim, hidden_layers, power):
    hidden_layers += 3  # we don't want to count the input/output layers as "hidden"

    return _power_space(
        start=latent_dim, stop=data_dim, n=hidden_layers, power=power
    ).astype(int)

def _build_encoder_decoder(
    data_dim,
    latent_dim,
    hidden_layers,
    power=2,
    dropout=0.1,
    activation_function_dict={"LeakyReLU": torch.nn.LeakyReLU()},
    device="cuda:0",
):

    """
    Construct a simple encoder / decoder network.
    Parameters:
    -----------
    data_dim
        Input data dimension
        type: int
    latent_dim
        Latent space dimensions
        type: int
    layers
        Number of layers in both the encoder and decoder.
        type: int
    power
        Exponential magnitude of change in layer size.
        default: 2
        type: float or int
    dropout
        Fraction of pre-programmed dropout nodes to be included in each layer.
        default: 0.1
        type: float
    silent
        If true, no message is returned describing the encoder / decoder network composition.
        default: False
        type: bool
    Returns:
    --------
    encoder, decoder
    Notes:
    ------
    """
    
    encoder_nodes_by_layer = _get_sequenced_VAE_layer_n(
        data_dim, latent_dim, hidden_layers, power
    ).astype(int)[::-1]
    encoder_nodes_by_layer[-1] = encoder_nodes_by_layer[-1] * 2

    decoder_nodes_by_layer = _get_sequenced_VAE_layer_n(
        data_dim, latent_dim, hidden_layers, power
    ).astype(int)

    encoder = _compose_nn_sequential(
        encoder_nodes_by_layer, activation_function_dict, dropout
    )
    decoder = _compose_nn_sequential(
        decoder_nodes_by_layer, activation_function_dict, dropout
    )

    return encoder.to(device), decoder.to(device)