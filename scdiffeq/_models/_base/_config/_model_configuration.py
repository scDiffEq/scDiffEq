
from ._trainer_configuration import TrainerConfiguration

class ModelConfiguration:
    """Manage the interaction with: LightningModel, Trainer, and LightningDataModule"""

    def __init__(
        self,
        model_save_dir: str = "scDiffEq_model",
        log_name: str = "lightning_logs",
        version=None,
        prefix="",
        flush_logs_every_n_steps=5,
        max_epochs=1500,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=5,
        **kwargs,
    ):
        self.trainer_kwargs = {}

    def _configure_lightning_model(self):
        pass
#         should return lightningmodelconfig
#         self.lightningmodelconfig = 

    def _conigure_trainer(self):
        self.trainer_config = TrainerConfiguration(**self.trainer_kwargs)

    def _configure_data_module(self):
        pass

    @property
    def LightingModel(self):
        self._configure_lightning_model()
        pass

    @property
    def Trainer(self):
        self._conigure_trainer()
        return self.trainer_config.trainer

    @property
    def DataModule(self):
        self._configure_data_module()
        pass