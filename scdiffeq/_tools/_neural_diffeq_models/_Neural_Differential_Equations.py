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
from .._machine_learning._create_outs_structure import _create_outs_structure
from ..._plotting._plot_vectorfield import _plot_meshgrid_vector_field
from .._general_tools._calculate_KernelDensity import _calculate_KernelDensity
from .._general_tools._calculate_global_quasi_potential import (
    _calculate_global_quasi_potential,
)
from .._general_tools._find_cell_quasi_potential import _find_cell_quasi_potential
from ..._plotting._plot_quasi_potential import _plot_quasi_potential


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
        _create_outs_structure(self)

        self.available_network_types = ["SDE", "ODE"]
        self.epoch = 0

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

        _split_test_train(
            self.adata,
            trajectory_column=trajectory_column,
            proportion_train=proportion_train,
            proportion_validation=proportion_validation,
            time_column=time_column,
            silent=silent,
            return_split_data=return_split_data,
        )

        self.adata.uns["ODE"] = self.network
        if outdir:
            self.outdir = outdir

    def learn(
        self,
        n_batches=20,
        n_epochs=1500,
        mode="parallel",
        time_column="time",
        learning_rate=False,
        validation_frequency=False,
        plot_progress=True,
        plot_summary=True,
        smoothing_factor=3,
        visualization_frequency=10,
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

        fig_save_path = self._imgs_path + "epoch_{}_training_progress.png".format(
            self.epoch
        )

        _learn_neural_ODE(
            self,
            n_epochs=n_epochs,
            n_batches=self.n_batches,
            mode="parallel",
            time_column=time_column,
            plot_progress=plot_progress,
            plot_summary=plot_summary,
            smoothing_factor=smoothing_factor,
            visualization_frequency=self.visualization_frequency,
            notebook=notebook,
            save_frequency=save_frequency,
            save_path=fig_save_path,
        )
        if outdir:
            self.outdir = outdir

    def evaluate(
        self,
        figsize=(6, 5.5),
        plot_predicted=True,
        plot_vectorfield=True,
        plot_KernelDensity=True,
        reset_KernelDensity=False,
        KDE_figure_legend_loc=4,
    ):

        """"""

        try:
            self.epoch
        except:
            self.epoch = "untrained"

        #         figure_save_path = os.path.join(self.outdir, "results_figures", "scdiffeq_outs/results_figures/")

        if type(self.epoch) == str:
            epoch_ = ""
        else:
            epoch_ = "epoch_"
        test_predict_path = self._imgs_path + "{}{}_predicted_test_data.png".format(
            epoch_, self.epoch
        )
        vector_field_path = self._imgs_path + "{}{}_predicted_vector_field.png".format(
            epoch_, self.epoch
        )

        self.test_accuracy = _evaluate(self.adata)

        if plot_predicted:
            _plot_predicted_test_data(
                self.adata, figsize=figsize, save_path=test_predict_path
            )
        if plot_vectorfield:
            _plot_meshgrid_vector_field(
                self.adata, figsize=figsize, save_path=vector_field_path
            )
        if plot_KernelDensity:
            _calculate_KernelDensity(
                self,
                clear_DensityDict=reset_KernelDensity,
                figure_legend_loc=KDE_figure_legend_loc,
                plot=plot_KernelDensity,
            )

    def compute_quasi_potential(
        self,
        calculation="probability_density",
        cmap="viridis",
        surface_opacity=0.9,
        cell_color="azure",
    ):

        """
        """
        if type(self.epoch) == str:
            epoch_ = ""
        else:
            epoch_ = "epoch_"

        qp_plot_path = self._imgs_path + "{}{}_quasi_potential_plot.html".format(
            epoch_, self.epoch
        )

        _calculate_global_quasi_potential(self, calculation=calculation)
        _find_cell_quasi_potential(self)
        _plot_quasi_potential(
            self,
            cmap=cmap,
            surface_opacity=surface_opacity,
            cell_color=cell_color,
            save_path=qp_plot_path,
        )

    def save(
        self,
        outdir=False,
        pickle_dump_list=["pca", "loss"],
        pass_keys=["split_data", "data_split_keys", "RunningAverageMeter"],
    ):

        """"""

        if not outdir:
            try:
                self.outdir
            except:
                self.outdir = os.getcwd()

        try:
            self.epoch = self.adata.uns["last_epoch"]
        except:
            self.epoch = self.epoch
        _save(
            self,
            outdir=self._outs_path,
            pickle_dump_list=pickle_dump_list,
            pass_keys=pass_keys,
        )
