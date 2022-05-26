
__module_name__ = "_model.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import torch

# import local dependencies #
# ------------------------- #
from ._ancilliary._ModelManager import _ModelManager
from ._ancilliary._Learner import _Learner

from ._ancilliary import _model_functions as funcs
from ._ancilliary import _loss_functions as loss_funcs


class _scDiffEq:
    def __init__(
        self,
        adata,
        device=0,
        silent=False,
        use_key="X",
        use_layer=None,
        VAE=True,
        VAE_latent_dim=10,
        VAE_hidden_layers=2,
        VAE_power=2,
        VAE_dropout=0.1,
        VAE_activation_function_dict={
            "LeakyReLU": torch.nn.LeakyReLU(negative_slope=0.01)
        },
        drift_hidden_architecture={1: [500, 500], 2: [500, 500]},
        drift_activation_function=torch.nn.Tanh(),
        diffusion_hidden_architecture={1: [500, 500], 2: [500, 500]},
        diffusion=True,
        diffusion_activation_function=torch.nn.Tanh(),
        drift_dropout=0.1,
        diffusion_dropout=0.1,
        batch_size=1,
        brownian_size=1,
    ):

        """ """

        self._adata = adata
        self._use_key = use_key
        self._use_layer = use_layer
        self._device = funcs.get_device(device)
        self._silent = silent
        self._X_data = funcs.determine_input_data(
            self._adata, use_key=use_key, layer=use_layer
        )

        (
            self._VAE,
            self._NeuralDiffEq,
            self._parameters,
            self._Model_HyperParams,
        ) = funcs.define_model(
            X_data=self._X_data,
            device=self._device,
            silent=self._silent,
            use_key=use_key,
            use_layer=use_layer,
            VAE=VAE,
            VAE_latent_dim=VAE_latent_dim,
            VAE_hidden_layers=VAE_hidden_layers,
            VAE_power=VAE_power,
            VAE_dropout=VAE_dropout,
            VAE_activation_function_dict=VAE_activation_function_dict,
            drift_hidden_architecture=drift_hidden_architecture,
            drift_activation_function=drift_activation_function,
            diffusion_hidden_architecture=diffusion_hidden_architecture,
            diffusion=diffusion,
            diffusion_activation_function=diffusion_activation_function,
            drift_dropout=drift_dropout,
            diffusion_dropout=diffusion_dropout,
            batch_size=batch_size,
            brownian_size=brownian_size,
        )
        
        self._ModelManager = _ModelManager(self._Model)
        self._Learner = _Learner(self._Model)

    def train(self, training_args):

        _training_program(self._Model,
                          self._ModelManager,
                          self._Learner,
                          training_args,
                         )

    def evaluate(self):
        
        _evaluate(self._Learner)

        
    def load_run(self, path):
        
        self._path = path
        self._Model, self._ModelManager, self._Learner = _load_run(self._path)
    
    def load_model(self, path):
        
        self._path = path
        self._Model = _load_model(self._path)


    def save(self, path):

        self._ModelManager.save()
        