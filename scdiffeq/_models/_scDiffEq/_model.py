
__module_name__ = "_model.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import torch

# import local dependencies #
# ------------------------- #
# from ._ancilliary._ModelManager import _ModelManager
# from ._ancilliary._Learner import _Learner

from . import _ancilliary as funcs
# from ._ancilliary import _loss_functions as loss_funcs


class _scDiffEq:
    def __init__(
        self,
        adata,
        device=0,
        lr=1e-3,
        batch_size=200,
        silent=False,
        use_key="X",
        time_key=None,
        TimeAugmentDict={2: 0, 4: 0.01, 6: 0.02},
        lineage_key=None,
        use_layer=None,
        optimization_func=torch.optim.RMSprop,
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
        brownian_size=1,
        reconstruction_loss_function=torch.nn.MSELoss(),
        reparameterization_loss_function=torch.nn.KLDivLoss(),
        save=False,
        save_path="X_train.pt",
    ):

        """ """
        
        self._reconstruction_loss_function = reconstruction_loss_function
        self._reparameterization_loss_function = reparameterization_loss_function
        
        self._adata = adata
        self._use_key = use_key
        self._batch_size = batch_size
        self._use_layer = use_layer
        self._device = funcs.ut.get_device(device)
        self._silent = silent
        self._X_data = funcs.data.determine_input_data(
            self._adata, use_key=use_key, layer=use_layer
        )
        self._lr = lr
        self._time_key = time_key
        self._save = save
        self._save_path = save_path
        self._silent = silent
        
        if time_key:
            self._time_key = time_key
            self._TimeAugmentDict = TimeAugmentDict
            funcs.data.augment_time(self._adata, self._time_key, "t_augmented", self._TimeAugmentDict)
        
        self._Model = funcs.init.define_model(
            X_data=self._X_data,
            device=self._device,
            silent=self._silent,
            use_key=use_key,
            use_layer=use_layer,
            optimization_func=optimization_func,
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
            reconstruction_loss_function=self._reconstruction_loss_function,
            reparameterization_loss_function=self._reparameterization_loss_function,
        )
        
#         self._ModelManager = _ModelManager(self._Model)
#         self._Learner = _Learner(self._Model,
#                                  lr=self._lr,
#                                  batch_size=self._batch_size,
#                                  device=self._device,
#                                 )
        
#         # pass attributes from the Learner and the ModelManager back to the main model class for convenient access
#         funcs.transfer_attributes(self._ModelManager, self)
#         funcs.transfer_attributes(self._Learner, self)

    def train(self,
              t=torch.Tensor([0, 0.01, 0.02]),
              epochs=5,
              learning_rate=1e-3,
              validation_frequency=5,
              checkpoint_frequency=20,
              notebook=True,
             ):
        
        print("Train...")
        
        
#         self._TrainingProgram = funcs.define_training_program(epochs,
#                                                         learning_rate,
#                                                         validation_frequency,
#                                                         checkpoint_frequency,
#                                                         notebook,
#                                                        )
        
#         self._X_data, self._t =  funcs.prepare_data_no_lineages(self._adata,
#                                   self._time_key,
#                                   self._use_key,
#                                   self._save,
#                                   self._save_path,
#                                   self._silent,
#                                  )
#         funcs.training_procedure(self, self._X_data, self._t)

#     def evaluate(self):
        
#         _evaluate(self._Learner)

        
#     def load_run(self, path):
        
#         self._path = path
#         self._Model, self._ModelManager, self._Learner = _load_run(self._path)
    
#     def load_model(self, path):
        
#         self._path = path
#         self._Model = _load_model(self._path)


#     def save(self, path):

#         self._ModelManager.save()
        