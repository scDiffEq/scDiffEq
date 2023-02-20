
from . import configs, lightning_models, utils
from .. import tools

import torch
from tqdm.notebook import tqdm
from pytorch_lightning import Trainer

NoneType = type(None)
import os


class scDiffEq(utils.AutoParseBase):
    def __init__(
        self,
        adata,
        func=None,
        h5ad_path=None,
        model_name="scDiffEq_model",
        use_key='X_pca',
        time_key="Time point",
        t0_idx=None,
        train_val_split=[0.9, 0.1],
        batch_size=2000,
        num_workers=os.cpu_count(),
        use_adjoint=False,
        obs_keys=['W'],
        groupby='Time point',
        train_key='train',
        val_key='val',
        test_key='test',
        predict_key='predict',
        silent=True,
    ):
        super(scDiffEq, self).__init__()

        self.__config__(locals())

    def configure_model(self, func):
        
        LightningModels = {
            ("TorchNet",  False): lightning_models.LightningDriftNet,
            ("TorchNet",  True):  lightning_models.LightningPotentialDriftNet,
            ("NeuralODE", False): lightning_models.LightningODE,
            ("NeuralODE", True):  lightning_models.LightningPotentialODE,
            ("NeuralSDE", False): lightning_models.LightningSDE,
            ("NeuralSDE", True):  lightning_models.LightningPotentialSDE,
        }

        self.credentials = configs.function_credentials(
            func, adjoint=self.use_adjoint
        )
        ftype = self.credentials["func_type"]
        potential = self.credentials["mu_is_potential"]
        self.DiffEq = LightningModels[(ftype, potential)](func=func)

    def _check_passed_time_args(self):
        """If time_key is passed"""
        if utils.not_none(self.t0_idx):
            self.time_key = None
        elif sum([utils.not_none(self.time_key), utils.not_none(self.t0_idx)]) < 1:
            "Must provide t0_idx or time_key. If both are provided, t0_idx overrules time_key"

    def __config__(self, kwargs):

        func = kwargs.pop("func")
        self.__parse__(kwargs, ignore=["self"])

        self._check_passed_time_args()
        self.time_attributes = configs.configure_time(
            self.adata, time_key=self.time_key, t0_idx=self.t0_idx
        )
        
        LitDataKwargs = utils.extract_func_kwargs(
            func = configs.LightningData,
            kwargs = kwargs,
        )

        self.LitDataModule = configs.LightningData(**LitDataKwargs)

        if isinstance(func, NoneType):
            n_dim = self.LitDataModule.train_dataset.X.shape[-1]
            func = utils.default_NeuralSDE(n_dim)

        self.configure_model(func=func)
        self.DiffEqLogger = utils.scDiffEqLogger(model_name=self.model_name)
        self.DiffEqLogger()
        self.TrainerGenerator = configs.LightningTrainerConfiguration(
            self.DiffEqLogger.versioned_model_outdir
        )

    def fit(
        self,
        epochs=500,
        callbacks=[],
        ckpt_frequency: int = 25,
        keep_ckpts: int = -1,
        monitor = None,
        accelerator=None,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=0.25,
        devices=None,
        **kwargs
    ):
        
        # bring all kwargs / args into the same place
        kwargs.update(locals())
        kwargs.pop("kwargs")
        
        if self.train_val_split[1] == 0:
            kwargs.update({'check_val_every_n_epoch': 0, 'limit_val_batches': 0})

        trainer_kwargs = utils.extract_func_kwargs(func=self.TrainerGenerator, kwargs=locals())
        trainer_kwargs.update(utils.extract_func_kwargs(func=Trainer, kwargs=kwargs))

        self.fit_trainer = self.TrainerGenerator(
            max_epochs=epochs, stage = "fit", **trainer_kwargs
        )
        self.fit_trainer.fit(self.DiffEq, self.LitDataModule)

    def test(self, callbacks=[]):
        
        if hasattr(self, "fit_trainer"):
            self.test_trainer = self.fit_trainer
        else:
            self.test_trainer = self.TrainerGenerator(stage="test", callbacks=callbacks)
        
        self.test_outs = self.test_trainer.test(self.DiffEq, self.LitDataModule)
        
        return self.test_outs
    
    def simulate(self, X0, t, N = 1, device="cuda:0"):
        return self.DiffEq.to(device).forward(X0.to(device), t.to(device))

    def predict(self, t, t0_idx=None,  predict_key="predict", n=2000, callbacks=[]): # use_key="X_pca", 
        
        # (1) Annotate self.LitDataModule.adata with a subset of cells to serve as prediction cells.
        #     Re-configure the LitDataModule to serve us with predict_dataloader
        tools.annotate_cells(self.LitDataModule.adata, idx=t0_idx, key=predict_key)
        
        # (2) Configure trainer
        if hasattr(self, "fit_trainer"):
            self.predict_trainer = self.fit_trainer
        else:
            self.predict_trainer = self.TrainerGenerator(stage="predict", callbacks=callbacks, enable_progress_bar=False)
        
        # (3) adjust t
        self.DiffEq.t = t
        
        # (4) loop over range(n) to make predictions
        self.predictions = {}
        for i in tqdm(range(n)):
            predicted = self.predict_trainer.predict(self.DiffEq, self.LitDataModule)
            self.predictions[i] = predicted
        
        return self.predictions


    def load(self, ckpt_path):
        """Assmes that self.DiffEq.func is the same as what you are trying to load."""
        self.DiffEq = self.DiffEq.load_from_checkpoint(ckpt_path, func=self.DiffEq.func)
        
    def __repr__(self):
        # TODO: add description of model / training status / data / etc.
        return "⏩ scDiffEq Model: {}".format(self.model_name)
