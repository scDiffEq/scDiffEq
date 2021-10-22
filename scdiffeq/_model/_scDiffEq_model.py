
# _scDiffEq_model.py
__module_name__ = "_scDiffEq_model.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])

import vintools as v

# package imports #
# --------------- #
# import torch


# local imports (5 steps / module groups) #
# --------------------------------------- #

from .._utilities._reset_scDiffEq import _reset_scDiffEq

# 1. initialization
from ._supporting_functions._initialization._neural_differential_equation import _formulate_network_model

# 2. training preflight
from ._supporting_functions._preflight._training_preflight import _preflight_parameters
from ._supporting_functions._preflight._split_adata import _split_adata

# 3. training
from ._supporting_functions._training._training_monitor import _TrainingMonitor
from ._supporting_functions._training._learn_diffeq import _learn_diffeq

# 4. evaluation
from ._supporting_functions._evaluation._evaluate_diffeq import _evaluate_diffeq

# 5. analysis

# --------------------------------------- #

class _scDiffEq:
    def __init__(self, diffusion=True, **kwargs):
        
        """
        Main class for representation of partial differential equation akin to a Fokker-Planck / Drift-Diffusion
        equation. 
        
        Parameters:
        -----------
        diffusion
            default: False
            type: bool
        
        in_dim
            default: 2
            type: int
        
        out_dim
            default: 2
            type: int
        
        layers
            default: 2
            type: int
        
        nodes
            default: 5
            type: int
        
        activation_function
            default: torch.nn.Tanh()
            type: torch.nn.modules.activation.<func>
            
        batch_size
            default: 10
            type: int
        
        brownian_size
            default: 1
            type: int
            
        
        Returns:
        --------
        scDiffEq
            type: class
        
        Notes:
        ------
        (1) Preflight is launched here as a "soft" run so that dependencies on these HPs can be instantiated in this 
            step. I should find a more efficient way to do this eventually. It is run again in the next step to update 
            any parameters.
        """
        
        self.diffusion = diffusion
        self.network_model, self.integration_function = _formulate_network_model(diffusion, **kwargs)
        self.hyper_parameters, self.ux_preferences, self._InvalidKwargs =  _preflight_parameters(self.network_model)
        self.TrainingMonitor = _TrainingMonitor(self.hyper_parameters.smoothing_momentum)
        model_instantiation_msg = "Neural DiffEq defined as:"
        print("{}\n\n {}".format(v.ut.format_pystring(model_instantiation_msg, ["BOLD", "CYAN"]), self.network_model))
        
    def preflight(self, adata, overfit=False, **kwargs):
        
        """
        Ensure that all data is formatted properly for training and all necessary parameters 
        are present and valid. 
        
        Parameters:
        -----------
        adata
            anndata._core.anndata.AnnData
        
        Returns:
        --------
        
        Notes:
        ------
        """
        
        self.overfit = overfit
        self.hyper_parameters, self.ux_preferences, self._InvalidKwargs =  _preflight_parameters(self.network_model, **kwargs)
        self.adata, self.validation_status = _split_adata(adata, self.hyper_parameters, self.ux_preferences, self.overfit)       
        
    def learn(self, plot=True, valid_plot_savepath=False):
        
#         try:
#             self.TrainingMonitor
#         except:
#              self.TrainingMonitor = _TrainingMonitor(self.hyper_parameters.smoothing_momentum)
            
        self.Learner = _learn_diffeq(self.adata, 
                      self.network_model, 
                      self.diffusion,
                      self.integration_function,
                      self.hyper_parameters, 
                      self.TrainingMonitor,
                      self.validation_status,
                      plot,
                      valid_plot_savepath,
                     )
        
    def evaluate(self, plot=True, save_path=False):
        
        if not self.validation_status:
            self.adata.obs["test"] = True
        
        self.Evaluator = _evaluate_diffeq(self, plot, save_path)
        
    def reset(self):

        """

        Wipe memory of model parameters and training statistics.

        Parameters:
        -----------
        None
        
        Returns:
        --------
        None
        
        Notes:
        ------
        (1) Resets the network parameters.
        (2) Clears the Training Monitor through resinstantiation of the class. 
        """
        
        self.network_model, self.TrainingMonitor = _reset_scDiffEq(self.network_model,
                                                                   self.TrainingMonitor, 
                                                                   self.hyper_parameters,
                                                                   self.ux_preferences.silent)
        
    def compute_quasi_potential(self,):
        
        print("")
        
    def save(self,):
        
        print("")
        
        