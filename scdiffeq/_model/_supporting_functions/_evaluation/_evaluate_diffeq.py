
from ._plot_evaluation import _plot_evaluation
    
import torch
import numpy as np

def _get_y0_idx(df, time_key):

    """"""

    y0_idx = df.index[np.where(df[time_key] == df[time_key].min())]

    return y0_idx


def _get_adata_y0(adata, time_key):

    y0_idx = _get_y0_idx(adata.obs, time_key)
    adata_y0 = adata[y0_idx].copy()

    return adata_y0


def _get_y0(adata, use, time_key):

    adata_y0 = _get_adata_y0(adata, time_key)

    if use == "X":
        return torch.Tensor(adata_y0.X)

    elif use in adata.obsm_keys():
        return torch.Tensor(adata_y0.obsm[use])

    else:
        print("y0 not properly defined!")


def _fetch_data(adata, use="X", time_key="time"):

    """

    Assumes parallel time.
    """

    y = torch.Tensor(adata.X)
    y0 = _get_y0(adata, use, time_key)
    t = torch.Tensor(adata.obs[time_key].unique())

    return y, y0, t


class Evaluator:
    def __init__(self, network_model, diffusion, integration_function, loss_function):

        self.parallel = True
        self.network_model = network_model
        self.diffusion = diffusion
        self.integration_function = integration_function
        self.loss_function = loss_function

    def forward_integrate(self, adata, use="X", time_key="time"):

        self.y, self.y0, self.t = _fetch_data(adata, use, time_key)

        with torch.no_grad():
            if self.parallel and not self.diffusion:

                self.pred_y = self.integration_function(
                    self.network_model.f, self.y0, self.t
                )

    def calculate_loss(self):

        self.test_loss = self.loss_function(
            self.pred_y, self.y.reshape(self.pred_y.shape)
        ).item()


def _evaluate_diffeq(DiffEq, plot=True):

    """"""

    test_adata = DiffEq.adata[DiffEq.adata.obs["test"]]
    evaluator = Evaluator(
        DiffEq.network_model,
        DiffEq.diffusion,
        DiffEq.integration_function,
        DiffEq.hyper_parameters.loss_function,
    )

    evaluator.forward_integrate(test_adata)
    evaluator.calculate_loss()

    if plot:
        _plot_evaluation(evaluator)

    print("Test loss: {:.4f}".format(evaluator.test_loss))

    return evaluator