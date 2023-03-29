
from ._lightning_sde import LightningSDE
from .mix_ins import PotentialMixIn


class LightningPotentialSDE(LightningSDE, PotentialMixIn):
    def __init__(
        self,
        func,
        dt=0.1,
        lr=1e-5,
        step_size=10,
        optimizer=torch.optim.RMSprop,
        lr_scheduler=torch.optim.lr_scheduler.StepLR,
    ):
        super(LightningPotentialSDE, self).__init__(
            func=func,
            dt=dt,
            lr=lr,
            step_size=step_size,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )