
__module_name__ = "_scDiffEq_Model.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import numpy as np
import torch
from tqdm.notebook import tqdm

# import local dependencies #
# ------------------------- #
from .._tools._OptimalTransportLoss import _OptimalTransportLoss
from .._preprocessing._format_lineage_data_for_model_input import _format_lineage_data_for_model_input
from .._utilities._LogFile import _LogFile

from ._model_functions._formulate_network_model import _formulate_network_model
from ._model_functions._run_trainer import _run_trainer
from ._model_functions._augment_NDE import _augment_NDE
from ._model_utilities._organize_run_info import _organize_run_info

from ._model_utilities._test_bool import _test_bool
from ._model_utilities._read_saved_status_log import _read_saved_status_log

from ._model_functions._run_epoch import _run_epoch
from ._model_utilities._save_model_checkpoint import _save_model_checkpoint

from ._model_functions._run_epoch_no_lineages import _run_epoch_no_lineages
from ._model_functions._prepare_data_no_lineages import _prepare_data_no_lineages

class _scDiffEq_Model:

    """
    --------------------------------------------
    *** scDiffEq - DEVELOPMENT VERSION 1.1.0 ***
    --------------------------------------------

    Main class for the Neural DE model that directly interfaces with adata.
    """

    def __init__(
        self,
        adata=False,
        lr=1e-4,
        outdir="./",
        run_name="test",
        run_group="just_testing",
        seed=1992,
        dt=False,
        diffusion=True,
        test_frequency=5,
        view_frequency=5,
        device="cpu",
        use_key="X_pca",
        lineage_index_key="clone_idx",
        time_key="Time point",
        groupby='dt_type',
        layers=2,
        nodes=5,
        activation_function=torch.nn.Tanh(),
        batch_size=1,
        brownian_size=1,
        noise_type="general",
        sde_type="ito",
        format_lineages=False,
        evaluate_only=False,
        notebook=True,
        verbose=False,
        silent=False,
        dry=False,
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
        
        
        self._view_frequency = view_frequency
        
        self._version = "dev1.1.0"
        self._run_group = run_group
        self._run_name = run_name
        self._dry = dry
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
        self._nn_func.to(self._device)
        self._best_loss = np.inf
        self._evaluate_only = evaluate_only
        self._test_frequency = test_frequency

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
            self._dry,
        )
        
        self._LossFunction = _OptimalTransportLoss(self._device)
        self._notebook = notebook
        self._LossTracker = {}
        
        if not self._dry:
            if not self._evaluate_only:
                self._LogFile = _LogFile(outdir=self._RunInfo.run_outdir,
                                         columns=["epoch", "d2", "d4", "d6", "total", "mode"])
        
        if format_lineages:
            self._FormattedData = _format_lineage_data_for_model_input(adata,
                                                                       use_key=use_key,
                                                                       lineage_key=lineage_index_key,
                                                                       groupby=groupby,
                                                                       time_key=time_key,
                                                                      )

            
    def train(self, lineage_key=None, epochs=25000, batch_size=False, lr=False, plot=True):

        """
        Iteratively train the scDiffEq model.

        Parameters:
        -----------
        epochs
        """
        
        self._run_count += 1
        self._LossTracker[self._run_count] = []

        if lr:
            self._lr = lr
            self._optimizer = torch.optim.RMSprop(
                self._nn_func.parameters(), lr=self._lr
            )

        if batch_size:
            self._batch_size = batch_size
            
            
        if lineage_key:
            for epoch in range(1, epochs+1):
                self._X_pred, self._loss_df = _run_epoch(
                           FormattedData=self._FormattedData,
                           func=self._nn_func,
                           optimizer=self._optimizer,
                           status_file=self._LogFile,
                           test=_test_bool(epoch, self._test_frequency),
                           loss_function=self._LossFunction,
                           dry=self._dry,
                           silent=self._silent,
                           epoch=epoch,
                           device=self._device,
                          )

                if (epoch % 10) == 0 and not self._dry:
                    _save_model_checkpoint(self._nn_func, epoch, self._RunInfo.run_outdir, silent=self._silent)
        
        
        else:
            
            X_train, t_train = _prepare_data_no_lineages(self._adata)
            if not batch_size:
                self._batch_size = 2000
            for epoch in tqdm(range(1, epochs+1)):
                self._epoch_loss = _run_epoch_no_lineages(X_train,
                                       t_train,
                                       self._nn_func,
                                       self._optimizer,
                                       self._LossFunction,
                                       device=self._device,
                                       batch_size=self._batch_size)
                
                self._LogFile.update(epoch=self._epoch_counter,
                                     epoch_loss=self._epoch_loss.tolist(),
                                     mode="TRAIN")
                
                if (epoch % self._view_frequency) == 0:
                    _read_saved_status_log(self._RunInfo, plot=plot)
                                
                if (epoch % 10) == 0 and not self._dry:
                    _save_model_checkpoint(self._nn_func, self._epoch_counter, self._RunInfo.run_outdir, silent=self._silent)
                self._epoch_counter += 1
            
    def view(self, plot=True):
        
        """ Check the status of a run based on what's written to the log file"""
        
        _read_saved_status_log(self._RunInfo, plot=plot)
    
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
