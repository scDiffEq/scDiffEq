
from ._base_ancilliary._SinkhornDivergence import SinkhornDivergence
from neural_diffeqs import NeuralSDE, NeuralODE
from pytorch_lightning import LightningModule
from ._batch_forward import BatchForward
from torch_composer import TorchNet
import torch


# -- Lightning base: -----
class LightningBase(LightningModule):
    """Base pytorch-lightning model wrapped / trained within models.scDiffEq"""

    def __init__(
        self,
        func: [NeuralSDE, NeuralODE, TorchNet],
        dt=0.1,
        optimizer_kwargs={"lr": 1e-4},
        scheduler_kwargs={"step_size": 20, "gamma": 0.1},
        ignore_t0=True,
    ):
        """To-do: docs"""
        super(LightningBase, self).__init__()
        self.__register_inputs__(func, dt, ignore_t0)

    def __register_inputs__(self, func, dt, ignore_t0):
        """To-do: docs"""

        self.func = func
        self.dt = dt
        self.save_hyperparameters(ignore=["func"])
        self.__configure_forward_step__(ignore_t0)

    def __configure_forward_step__(self, ignore_t0):
        """To-Do: docs"""
        forward_step = BatchForward(self.func, loss_function = SinkhornDivergence(ignore_t0=ignore_t0))
        setattr(self, "forward", getattr(forward_step, "__call__"))

    def configure_optimizers(self):
        """To-Do: docs"""
        optimizer = torch.optim.RMSprop(
            self.parameters(), **self.hparams["optimizer_kwargs"]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, **self.hparams["scheduler_kwargs"]
        )
        return [optimizer], [scheduler]