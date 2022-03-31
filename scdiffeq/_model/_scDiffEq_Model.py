
__module_name__ = "_scDiffEq_Model.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import numpy as np
import torch


# import local dependencies #
# ------------------------- #
from ._model_functions._formulate_network_model import _formulate_network_model
from ._model_functions._run_trainer import _run_trainer
from ._model_utilities._organize_run_info import _organize_run_info
from ._model_utilities._log import _setup_logfile

class _scDiffEq_Model:

    """
    --------------------------------------------
    *** scDiffEq - DEVELOPMENT VERSION 1.1.0 ***
    --------------------------------------------

    Main class for the Neural DE model that directly interfaces with adata.
    """

    def __init__(
        self,
        adata,
        lr=1e-4,
        outdir="./",
        run_name="test",
        run_group="just_testing",
        seed=1992,
        dt=False,
        diffusion=True,
        device="cpu",
        use_key="X_pca",
        lineage_index_key="clone_idx",
        time_key="Time point",
        layers=2,
        nodes=5,
        activation_function=torch.nn.Tanh(),
        batch_size=1,
        brownian_size=1,
        noise_type="general",
        sde_type="ito",
        evaluate_only=False,
        notebook=True,
        verbose=False,
        silent=False,
        **kwargs,
    ):

        """

        Parameters:
        -----------
        adata
            AnnData object
            type: anndata._core.anndata.AnnData

        dt
            An artificial `dt` to take smaller steps in-between observed timepoints.
            type: float

        diffusion
            default: True
            type: bool



        Returns:
        --------
        None, modifies the class in-place.

        Notes:
        ------
        (1) still not sure if dt or n_steps is the right approach. I think `dt` is good.
        (2) The only required argument is `adata`
        (3) Eventually the run signature should reflect a schedule of learning rates
        """
        self._version = "dev1.1.0"
        self._run_group = run_group
        self._run_name = run_name
        self._outdir = outdir
        self._seed = torch.manual_seed(seed).seed()
        self._adata = adata
        self._X_data = self._adata.obsm[use_key]
        self._dt = dt
        self._in_dim = self._X_data.shape[1]
        self._out_dim = self._X_data.shape[1]
        self._nodes = nodes
        self._layers = layers
        self._lr = lr
        self._run_count = 0
        self._verbose = verbose
        self._silent=silent
        if self._silent:
            self._verbose = False
        
        self._nn_func, self._int_func = _formulate_network_model(
            in_dim=self._in_dim,
            out_dim=self._out_dim,
            nodes=self._nodes,
            layers=self._layers,
            silent=self._silent,
        )
        self._epoch_counter = 0
        self._optimizer = torch.optim.RMSprop(self._nn_func.parameters(), lr=self._lr)
        self._device = device
        self._batch_size = self._nn_func.batch_size = batch_size
        self._best_loss = np.inf
        self._evaluate_only = evaluate_only

        self._RunInfo = _organize_run_info(
            self._outdir,
            self._evaluate_only,
            self._run_group,
            self._run_name,
            self._version,
            self._nodes,
            self._layers,
            self._lr,
            self._device,
            self._seed,
            self._verbose,
        )
        
        self._notebook = notebook
        self._LossTracker = {}
        if not self._evaluate_only:
            self._status_file = _setup_logfile(self._RunInfo.run_outdir,
                                               columns=["epoch", "d2", "d4", "d6", "total"])

    def train(self, epochs, batch_size=False, lr=False):

        """
        Iteratively train the scDiffEq model.

        Parameters:
        -----------
        epochs
        """

        if lr:
            self._lr = lr
            self._optimizer = torch.optim.RMSprop(
                self._nn_func.parameters(), lr=self._lr
            )

        if batch_size:
            self._batch_size = batch_size
            
        self._run_count += 1
        self._LossTracker[self._run_count] = []
        _run_trainer(self._RunInfo.run_outdir, 
                     epochs,
                     self._adata,
                     self._X_data,
                     self._nn_func,
                     self._int_func, 
                     self._optimizer,
                     self._LossTracker,
                     self._status_file,
                     self._epoch_counter,
                     notebook=self._notebook,
                    )
                
    def save(self):

        """Saves model and working adata"""

        # TO-DO: _model_saver()
        # TO-DO: _adata_saver()

    def load(self, model_path, device=False):

        """Loads model and optionally, working adata"""
        
        self._model_path = model_path
        if device:
            if type(device) is int:
                device = "cuda:{}".format(device)
            self._device = device
        
        self._nn_func.load_state_dict(torch.load(self._model_path, map_location=self._device))

        # TO-DO: _model_loader()
        # TO-DO: _adata_loader()