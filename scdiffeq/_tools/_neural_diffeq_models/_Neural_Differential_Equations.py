
# package imports #
# --------------- #
import torch.nn as nn
import os

# local imports #
# ------------- #
from ._ODE._ODE import Neural_ODE
from ._SDE._SDE import Neural_SDE

from .._machine_learning._preflight import _preflight
from .._machine_learning._learn_neural_ODE import _learn_neural_ODE
from .._machine_learning._split_AnnData_test_train_validation import _split_test_train
from .._machine_learning._plot_test_predictions import _plot_predicted_test_data
from .._machine_learning._evaluate import _evaluate
from .._machine_learning._save import _save

class scDiffEq:
    
    def __init__(
        self,
        network_type="ODE",
        outdir=False,
        in_dim=2,
        out_dim=2,
        n_layers=4,
        nodes_n=50,
        nodes_m=50,
        activation_function=nn.Tanh(),
        silent=False,
    ):

        """
        """
        
        if outdir:
            self.outdir = outdir
        self.available_network_types = ["SDE", "ODE"]

        assert network_type in self.available_network_types, print(
            "Choose from available network types: {}".format(
                self.available_network_types
            )
        )
        if network_type == "ODE":
            self.network = Neural_ODE(
                in_dim=in_dim,
                out_dim=out_dim,
                n_layers=n_layers,
                nodes_n=nodes_n,
                nodes_m=nodes_m,
                activation_function=activation_function,
            )
            
        if not silent:
            print(self.network)

    def preflight(
        self,
        adata,
        outdir=False,
        validation_frequency=2,
        visualization_frequency=10,
        loss_function="MSELoss",
        optimizer="RMSprop",
        learning_rate=1e-3,
        trajectory_column="trajectory",
        proportion_train=0.60,
        proportion_validation=0.20,
        time_column="time",
        silent=False,
        return_split_data=False,
    ):

        """"""

        _preflight(
            self,
            adata,
            validation_frequency=validation_frequency,
            visualization_frequency=visualization_frequency,
            loss_function=loss_function,
            optimizer=optimizer,
            learning_rate=learning_rate,
        )
        
        _split_test_train(self.adata,
            trajectory_column=trajectory_column,
            proportion_train=proportion_train,
            proportion_validation=proportion_validation,
            time_column=time_column,
            silent=silent,
            return_split_data=return_split_data,
        )
        
        self.adata.uns['ODE'] = self.network
        if outdir:
            self.outdir = outdir
        

    def learn(
        self,
        n_batches=10,
        n_epochs=1500,
        learning_rate=False,
        validation_frequency=False,
        plot_progress=False,
        plot_summary=True,
        smoothing_factor=3,
        visualization_frequency=False,
        notebook=True,
        save_frequency=5,
        outdir=False,
        
    ):

        """"""

        # make any necessary updates
        if n_batches:
            self.n_batches = n_batches
        if visualization_frequency:
            self.visualization_frequency = visualization_frequency
        if validation_frequency:
            self.validation_frequency = validation_frequency
        if learning_rate:
            self.learning_rate = learning_rate
               
        _learn_neural_ODE(
            self,
            n_epochs=n_epochs,
            plot_progress=plot_progress,
            plot_summary=plot_summary,
            smoothing_factor=smoothing_factor,
            visualization_frequency=self.visualization_frequency,
            notebook=notebook,
            save_frequency=save_frequency,
        )
        if outdir:
            self.outdir = outdir


    def evaluate(self,):

        """"""

        self.test_accuracy = _evaluate(self.adata)
        _plot_predicted_test_data(self.adata)


    def save(self,
             outdir=False,
             pickle_dump_list = ["pca", "loss"], 
             pass_keys = ["split_data", "data_split_keys", "RunningAverageMeter"]):

        """"""
        
        if not outdir:
            try:
                self.outdir
            except:
                self.outdir = os.getcwd()
            
        try:
            self.epoch = self.adata.uns['last_epoch']
        except:
            self.epoch = self.epoch
        _save(
            self,
            outdir=self.outdir,
            pickle_dump_list=pickle_dump_list,
            pass_keys=pass_keys
        )
