
__module_name__ = "_configuration.py"
__doc__ = """To-do."""
__version__ = """0.0.45"""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


class scDiffEqConfiguration:
    """
    Manage the interaction with: LightningModel, Trainer, and LightningDataModule
    # called from within scDiffEq
    """
    def __init__(self):
        pass
    
    def _configure_lightning_model(self):
        """sets self._LightningModel"""
        pass

    def _configure_lightning_trainer(self):
        """sets self._LightningTrainer"""
        pass

    def _configure_lightning_data_module(self):
        """sets self._LightningDataModule"""
        pass

    @property
    def LightingModel(self):
        self._configure_lightning_model()
        return self._LightningModel

    @property
    def LightningTrainer(self):
        self._configure_lightning_trainer()
        return self._LightningTrainer

    @property
    def LightningDataModule(self):
        self._configure_lightning_data_module()
        return self._LightningDataModule
