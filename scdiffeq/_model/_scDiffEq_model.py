
# _scDiffEq_model.py
__module_name__ = "_scDiffEq_model.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])

import licorice

# local imports (5 steps / module groups) #
# --------------------------------------- #

from .._utilities._reset_scDiffEq import _reset_scDiffEq

# 1. initialization
from ._supporting_functions._initialization._neural_differential_equation import _formulate_network_model
from ._supporting_functions._model_output._ModelOutput import _ModelOutput

# 2. training preflight
from ._supporting_functions._preflight._training_preflight import _preflight_parameters
from ._supporting_functions._preflight._split_adata import _split_adata

# 3. training
from ._supporting_functions._training._training_monitor import _TrainingMonitor
from ._supporting_functions._training._learn_diffeq import _learn_diffeq

# 4. evaluation
from ._supporting_functions._evaluation._evaluate_diffeq import _evaluate_diffeq

# 5. analysis



from .._utilities._torch_device import _set_device

# --------------------------------------- #

class _scDiffEq:
    def __init__(self, diffusion=True, run_name=False, device=False, outpath=False, **kwargs):
        
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
        if device:
            self.device = device
        else:
            self.device=_set_device()
            
        if run_name:
            self.run_name = run_name
            
            
        self.ModelOuts = _ModelOutput(outpath)
        
        self.diffusion = diffusion
        self.network_model, self.integration_function = _formulate_network_model(diffusion, self.device, **kwargs)
        self.hyper_parameters, self.ux_preferences, self._InvalidKwargs =  _preflight_parameters(self.network_model)
        self.TrainingMonitor = _TrainingMonitor(self.hyper_parameters.smoothing_momentum)
        model_instantiation_msg = "Neural DiffEq defined as:"
        print("{}\n\n {}".format(licorice.font_format(model_instantiation_msg, ["BOLD", "CYAN"]), self.network_model))
        
    def preflight(self, adata, n_batches=50, overfit=False, use='X', time_key='time', **kwargs):
        
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
        
        self.n_batches=n_batches
        self.use = use
        self.overfit = overfit
        self.time_key = time_key
        self.hyper_parameters, self.ux_preferences, self._InvalidKwargs =  _preflight_parameters(self.network_model, **kwargs)
        self.adata, self.perform_validation = _split_adata(adata, self.hyper_parameters, self.ux_preferences, self.overfit)       
        
    def learn(self, use=False, time_key=False, plot=True, **kwargs):
        
        """
        Parameters:
        -----------
        use
        
        time_key
        
        plot
                
        Returns:
        --------
        None: self.Learner is updated
        
        Notes:
        ------
        (1)
        """
        
        self.hyper_parameters, self.ux_preferences, self._InvalidKwargs =  _preflight_parameters(self.network_model, **kwargs)
            
        if use:
            self.use = use
        if time_key:
            self.time_key = time_key
            
        self.Learner = _learn_diffeq(self.adata, 
                                     self.network_model,
                                     self.n_batches,
                                     self.device,
                                     self.diffusion,
                                     self.integration_function,
                                     self.hyper_parameters, 
                                     self.TrainingMonitor,
                                     self.perform_validation,
                                     self.use,
                                     self.time_key,
                                     plot,
                                    )
        
    def evaluate(self, use=False, time_key=False, plot_evaluation=True, plot_save_path=False):
        
        """
        Evaluate the instantiated model.
        
        Notes:
        ------
        (1) Model can be trained or untrained to run this function. 
        
        """
                
        if use:
            self.use = use
        if time_key:
            self.time_key = time_key
        
        self.plot_save_path = plot_save_path
        
        if not self.perform_validation:
            self.adata.obs["test"] = True
        
        self.Evaluator = _evaluate_diffeq(self,
                                          n_batches=self.n_batches,
                                          device=self.device,
                                          plot=plot_evaluation,
                                          plot_save_path=plot_save_path,
                                          use=self.use,
                                          time_key=self.time_key,
                                          plot_title_fontsize=self.ux_preferences.plot_title_fontsize)
        
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
        
        