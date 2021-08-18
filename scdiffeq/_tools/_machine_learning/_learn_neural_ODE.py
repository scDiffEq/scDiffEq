import os
import vintools as v

from ._forward_integration_functions._parallel_batch_time._forward_integrate_epoch_parallel_time import (
    _forward_integrate_epoch_parallel_time,
)
from ._forward_integration_functions._individual_trajectories._forward_integrate_epoch import (
    _forward_integrate_epoch,
)
from ._plot_neural_ODE_training import _plot_loss
from ._save_adata_uns import _save_torch_model
from .._machine_learning._save import _save


def _learn_neural_ODE(
    self,
    n_epochs=100,
    n_batches=20,
    mode="parallel",
    time_column="time",
    plot_progress=True,
    plot_summary=True,
    notebook=False,
    smoothing_factor=3,
    visualization_frequency=5,
    save_frequency=5,
    save_path=False,
):

    """"""

    model_checkpoint_path = os.path.join(self._outs_path, "model_checkpoints")
    v.ut.mkdir_flex(model_checkpoint_path)

    if notebook:
        from tqdm import tqdm_notebook as tqdm

        bar_format = None
    else:
        from tqdm import tqdm as tqdm

        bar_format = "{l_bar}{bar:60}{r_bar}{bar:-60b}"

    for epoch in tqdm(
        (range(1, n_epochs + 1)), desc="Training progress", bar_format=bar_format
    ):

        self.epoch = self.adata.uns["last_epoch"] = epoch

        if mode == "parallel":
            _forward_integrate_epoch_parallel_time(
                self, epoch, time_column=time_column, n_batches=n_batches
            )
        else:
            _forward_integrate_epoch(self, epoch)

        if plot_progress:
            if epoch % self.visualization_frequency == 0:
                test_predict_path = (
                    self._imgs_path
                    + "epoch_{}_training_progress.png".format(self.epoch)
                )
                _plot_loss(
                    self.adata, groupsize=smoothing_factor, save_path=test_predict_path
                )
                torch_model_path = os.path.join(
                    model_checkpoint_path, "model_{}".format(self.epoch)
                )
                _save_torch_model(self, torch_model_path)

    if plot_summary:
        _plot_loss(self.adata, groupsize=smoothing_factor, save_path=save_path)
