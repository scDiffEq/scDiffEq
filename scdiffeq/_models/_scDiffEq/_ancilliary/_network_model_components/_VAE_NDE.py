
__module_name__ = "_VAE_SDE.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import torch
import torchsde


# import local dependencies #
# ------------------------- #
from ._VAE_utilities import _no_transform
from ._build_encoder_decoder import _build_encoder_decoder
from ._reparameterize import _reparameterize


class _VAE_SDE(torch.nn.Module):
    def __init__(
        self,
        X_dim=None,
        latent_dim=10,
        hidden_layers=1,
        power=2,
        dropout=0.1,
        activation_function_dict={"LeakyReLU": torch.nn.LeakyReLU()},
        device=0,
    ):
        super(_LinearVAE, self).__init__()
        
        self._X_dim = X_dim
        self._latent_dim = latent_dim
        self._device = device
        self._hidden_layers = hidden_layers
        self._power = power
        self._dropout = dropout
        self._activation_function_dict = activation_function_dict

        self._encoder, self._decoder = _build_encoder_decoder(
            data_dim=self._X_dim,
            latent_dim=self._latent_dim,
            hidden_layers=self._hidden_layers,
            power=self._power,
            dropout=self._dropout,
            activation_function_dict=self._activation_function_dict,
            device=self._device,
        )

    def encode(self, X, transform=_no_transform):

        self._X_latent = transform(self._encoder(X))

    def reparameterize(self):

        self._X_latent, self._mu, self._log_var = _reparameterize(self._X_latent)

    def forward_integrate(self, func, time):
        
        """X_latent is replaced with what is really X_pred_latent"""
        
        self._X_latent = torchsde.sdeint(func, self._X_latent, time)
    
    def decode(self, transform=_no_transform, return_reconstructed=True):

        self._X_reconstructed = transform(self._decoder(self._X_latent))
        if return_reconstructed:
            return self._X_reconstructed