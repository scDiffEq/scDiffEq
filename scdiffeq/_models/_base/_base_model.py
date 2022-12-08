
# -- import packages: --------------------------------------------------------------------
from abc import ABC


# -- import local dependencies: ----------------------------------------------------------
from ._config import ModelConfiguration


# -- base model class: -------------------------------------------------------------------
class BaseModel(ABC):
    """
    Base model to interface PyTorch-Lightning model with a
    Lightning Trainer, an AnnData / LightningDataModule.
    """

    def __init__(self, **kwargs):
        pass

    def __configure__(self, params):
        self.model_configs = ModelConfiguration(**params)
        self.trainer = self.model_configs.Trainer

    def fit(self):
        self.trainer.fit(self.LightningModel, self.DataModule)

    def test(self):
        self.test_pred = self.trainer.test(self, self.DataModule)

    def predict(self):
        self.pred = self.trainer.predict(self, self.DataModule)
