

# -- import packages: ----------------------------------------------------------
from tqdm.notebook import tqdm
import lightning
import torch
import glob
import os


# -- import local dependencies: ------------------------------------------------
from . import configs, lightning_models, utils
from .. import tools


# -- type setting: -------------------------------------------------------------
from typing import Union, List
NoneType = type(None)


class scDiffEq(utils.ABCParse):
    def __init__(
        self,
        adata,
        # -- general params: ----------------------------------------------------
        latent_dim=20,
        model_name="scDiffEq_model",
        use_key="X_scaled",
        time_key="Time point",
        obs_keys=["W"],
        kNN_key="X_pca_scDiffEq",
        pretrain_epochs=500,
        pretrain_lr=1e-3,
        pretrain_optimizer=torch.optim.Adam,
        pretrain_step_size=100,
        pretrain_scheduler=torch.optim.lr_scheduler.StepLR,
        train_epochs=1500,
        train_lr=1e-5,
        train_optimizer=torch.optim.RMSprop,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        train_step_size=10,
        dt=0.1,
        seed=617,
        t0_idx=None,
        train_val_split=[0.9, 0.1],
        batch_size=2000,
        num_workers=os.cpu_count(),
        adjoint=False,
        groupby="Time point",
        train_key="train",
        val_key="val",
        test_key="test",
        predict_key="predict",
        silent=True,
        scale_input_counts=True,
        reduce_dimensions=True,
        build_kNN=True,
        # -- DiffEq params: ----------------------------------------------------
        mu_hidden: Union[List[int], int] = [400, 400, 400],
        sigma_hidden: Union[List[int], int] = [400, 400, 400],
        mu_activation: Union[str, List[str]] = "LeakyReLU",
        sigma_activation: Union[str, List[str]] = "LeakyReLU",
        mu_dropout: Union[float, List[float]] = 0.1,
        sigma_dropout: Union[float, List[float]] = 0.1,
        mu_bias: bool = True,
        sigma_bias: List[bool] = True,
        mu_output_bias: bool = True,
        sigma_output_bias: bool = True,
        mu_n_augment: int = 0,
        sigma_n_augment: int = 0,
        sde_type: str = "ito",
        noise_type: str = "general",
        brownian_dim: int = 1,
        coef_drift: float = 1.0,
        coef_diffusion: float = 1.0,
        coef_prior_drift: float = 1.0,
        DiffEq_type: str = "SDE",
        potential_type: str = "prior",
        # -- Encoder params: ---------------------------------------------------
        encoder_n_hidden: int = 4,
        encoder_power: float = 2,
        encoder_activation: Union[str, List[str]] = "LeakyReLU",
        encoder_dropout: Union[float, List[float]] = 0.2,
        encoder_bias: bool = True,
        encoder_output_bias: bool = True,
        # -- Decoder params: ---------------------------------------------------
        decoder_n_hidden: int = 4,
        decoder_power: float = 2,
        decoder_activation: Union[str, List[str]] = "LeakyReLU",
        decoder_dropout: Union[float, List[float]] = 0.2,
        decoder_bias: bool = True,
        decoder_output_bias: bool = True,
    ):
        super().__init__()

        self.__config__(locals())

    def _configure_data(self, kwargs):

        """Configure data (including time time)"""

        self.time_attributes = configs.configure_time(
            self.adata, time_key=self._time_key, t0_idx=self._t0_idx
        )

        self.LitDataModule = configs.LightningData(
            **utils.extract_func_kwargs(func=configs.LightningData, kwargs=kwargs)
        )
        self._data_dim = self.LitDataModule.n_dim

    def _configure_model(self, kwargs):

        self._LitModelConfig = configs.LightningModelConfiguration(
            data_dim=self._data_dim,
            latent_dim=self._latent_dim,
            DiffEq_type=self._DiffEq_type,
            potential_type=self._potential_type,
        )

        self.DiffEq = self._LitModelConfig(kwargs)
        self._INFO(f"Using the specified parameters, {self.DiffEq} has been called.")

    def _configure_dimension_reduction(self):
        self.reducer = tools.DimensionReduction(self.adata)
        if self._scale_input_counts:
            self._INFO("Scaling input counts (for dimension reduction).")
            self.reducer.scale()
            self.adata.obsm["X_scaled_scDiffEq"] = self.reducer.X_scaled
        else:
            self.reducer.X_scaled_train = reducer.X_train
            self.reducer.X_scaled = reducer.X

        if self._reduce_dimensions:
            self.reducer.pca()
            self.adata.obsm["X_pca_scDiffEq"] = self.reducer.X_pca
            self.reducer.umap()
            self.adata.obsm["X_umap_scDiffEq"] = self.reducer.X_umap

    def _configure_logger(self):
        self.DiffEqLogger = utils.scDiffEqLogger(model_name=self._model_name)
        self.DiffEqLogger()

    def _configure_trainer_generator(self):
        self.TrainerGenerator = configs.LightningTrainerConfiguration(
            self.DiffEqLogger.versioned_model_outdir
        )

    def _configure_kNN_graph(self):
        train_adata = self.adata[self.adata.obs[self._train_key]]
        self._INFO(f"Bulding Annoy kNN Graph on adata.obsm['{self._kNN_key}']")
        self.kNN_Graph = utils.FastGraph(adata=train_adata, use_key=self._kNN_key)

    def __config__(self, kwargs):

        """
        Run on model.__init__()

        Step 0: Parse all kwargs
        Step 1: Set up info messaging [TODO: eventually replace this with more sophisticated logging]
        Step 2: Configure data
        Step 3: Configure lightning model
        Step 4: Configure dimension reduction models
        Step 5: Configure kNN Graph.
        Step 6: Configure logger
        Step 7: Configure TrainerGenerator
        """

        self.__parse__(kwargs, public=["adata"])
        self._INFO = utils.InfoMessage()
        self._configure_data(kwargs)
        self._configure_model(kwargs)
        if kwargs["reduce_dimensions"]:
            self._configure_dimension_reduction()
        if kwargs["build_kNN"]:
            self._configure_kNN_graph()
        self._configure_logger()
        self._configure_trainer_generator()

    def freeze(self):
        """Freeze lightning model"""
        self.DiffEq.freeze()

    def load(self, ckpt_path, freeze=True):
        self.ckpt_path = ckpt_path
        self.DiffEq.load_from_checkpoint(ckpt_path)
        if freeze:
            self.DiffEq.freeze()

    def _stage_log_path(self, stage):
        log_path = glob.glob(self.DiffEqLogger.versioned_model_outdir + f"/{stage}*")[0]
        self._INFO(f"Access logs at: {log_path}")

    def _configure_pretrain_step(self, epochs):

        STAGE = "pretrain"
        self._INFO(f"Configuring fit step: {STAGE}")

        if not isinstance(epochs, NoneType):
            self._pretrain_epochs = epochs
            self._PARAMS["_pretrain_epochs"] = epochs

        trainer_kwargs = utils.extract_func_kwargs(
            func=self.TrainerGenerator,
            kwargs=self._PARAMS,
        )
        trainer_kwargs.update(
            utils.extract_func_kwargs(
                func=lightning.Trainer,
                kwargs=self._PARAMS,
            )
        )
        self.pre_trainer = self.TrainerGenerator(
            max_epochs=self._pretrain_epochs, stage=STAGE, **trainer_kwargs
        )
        self._stage_log_path(STAGE)

    def pretrain(
        self,
        epochs=None,
    ):
        """If any of the keyword arguments are passed, they will replace the previously-stated arguments from __init__ and re-configure the DiffEq."""

        self._configure_pretrain_step(epochs)
        self.pre_trainer.fit(self.DiffEq, self.LitDataModule)

    def _configure_train_step(self, epochs, kwargs):

        STAGE = "train"

        self._INFO(f"Configuring fit step: {STAGE}")

        if not isinstance(epochs, NoneType):
            self._train_epochs = epochs
            self._PARAMS["train_epochs"] = epochs

        trainer_kwargs = utils.extract_func_kwargs(
            func=self.TrainerGenerator,
            kwargs=self._PARAMS,
        )
        trainer_kwargs.update(
            utils.extract_func_kwargs(
                func=lightning.Trainer,
                kwargs=self._PARAMS,
            )
        )
        if self._train_val_split[1] == 0:
            trainer_kwargs.update(
                {"check_val_every_n_epoch": 0, "limit_val_batches": 0}
            )

        self.trainer = self.TrainerGenerator(
            max_epochs=self._train_epochs, stage=STAGE, **trainer_kwargs
        )

        self._stage_log_path(STAGE)

    def train(
        self,
        epochs=500,
        callbacks=[],
        ckpt_frequency: int = 25,
        save_last_ckpt: bool = True,
        keep_ckpts: int = -1,
        monitor=None,
        accelerator=None,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
        devices=None,
        deterministic=False,
        **kwargs,
    ):

        self._configure_train_step(epochs, locals())
        self.trainer.fit(self.DiffEq, self.LitDataModule)

    def fit(
        self,
        train_epochs=200,
        pretrain_epochs=500,
        callbacks=[],
        ckpt_frequency: int = 25,
        save_last_ckpt: bool = True,
        keep_ckpts: int = -1,
        monitor=None,
        accelerator=None,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
        devices=None,
        deterministic=False,
        **kwargs,
    ):

        if pretrain_epochs > 0:
            self.pretrain(epochs=pretrain_epochs)
        if train_epochs > 0:
            self.train(
                epochs=train_epochs, **utils.extract_func_kwargs(self.train, locals())
            )


# class scDiffEq(utils.AutoParseBase):
#     def __init__(
#         self,
#         adata,
#         latent_dim=20,
        
#         # -- sde params: ----
#         mu_hidden: Union[List[int], int] = [400, 400, 400],
#         sigma_hidden: Union[List[int], int] = [400, 400, 400],
#         mu_activation: Union[str, List[str]] = 'LeakyReLU',
#         sigma_activation: Union[str, List[str]] = 'LeakyReLU',
#         mu_dropout: Union[float, List[float]] = 0.1,
#         sigma_dropout: Union[float, List[float]] = 0.1,
#         mu_bias: bool = True,
#         sigma_bias: List[bool] = True,
#         mu_output_bias: bool = True,
#         sigma_output_bias: bool = True,
#         mu_n_augment: int = 0,
#         sigma_n_augment: int = 0,
#         sde_type='ito',
#         noise_type='general',
#         brownian_dim=1,
#         coef_drift: float = 1.0,
#         coef_diffusion: float = 1.0,
#         coef_prior_drift: float = 1.0,
        
#         DiffEq_type: str = "SDE",
#         potential_type="prior",
#         pretrain_epochs=1500,
# #         func=None,
# #         h5ad_path=None,
#         model_name="scDiffEq_model",
#         use_key='X_scaled',
#         time_key="Time point",
#         dt=0.1,
#         lr=1e-4,
#         seed = 617,
#         step_size=10,
# #         optimizer=torch.optim.RMSprop,
# #         lr_scheduler=torch.optim.lr_scheduler.StepLR,
#         t0_idx=None,
#         train_val_split=[0.9, 0.1],
#         batch_size=2000,
#         num_workers=os.cpu_count(),
#         adjoint=False,
#         obs_keys=['W'],
#         groupby='Time point',
#         train_key='train',
#         val_key='val',
#         test_key='test',
#         predict_key='predict',
#         silent=True,
        
#         # -- encoder parameters: -------
#         encoder_n_hidden: int = 4,
#         encoder_power: float = 2,
#         encoder_activation: Union[str, List[str]] = 'LeakyReLU',
#         encoder_dropout: Union[float, List[float]] = 0.2,
#         encoder_bias: bool = True,
#         encoder_output_bias: bool = True,
        
#         # -- decoder parameters: -------
#         decoder_n_hidden: int = 4,
#         decoder_power: float = 2,
#         decoder_activation: Union[str, List[str]] = 'LeakyReLU',
#         decoder_dropout: Union[float, List[float]] = 0.2,
#         decoder_bias: bool = True,
#         decoder_output_bias: bool = True,
        
#     ):
#         super(scDiffEq, self).__init__()

#         self.__config__(locals())        

#     def _check_passed_time_args(self):
#         """If time_key is passed"""
#         if utils.not_none(self.t0_idx):
#             self.time_key = None
#         elif sum([utils.not_none(self.time_key), utils.not_none(self.t0_idx)]) < 1:
#             "Must provide t0_idx or time_key. If both are provided, t0_idx overrules time_key"

#     def __config__(self, kwargs):

# #         func = kwargs.pop("func")
# #         self._check_passed_time_args()

#         self.__parse__(kwargs, ignore=["self"])
#         seed_everything(self.seed, workers=True)
#         self.time_attributes = configs.configure_time(
#             self.adata, time_key=self.time_key, t0_idx=self.t0_idx
#         )
        
#         LitDataKwargs = utils.extract_func_kwargs(
#             func = configs.LightningData,
#             kwargs = kwargs,
#         )
#         self.LitDataModule = configs.LightningData(**LitDataKwargs)   
        
#         self.LitModelConfig = configs.LightningModelConfiguration(
#             data_dim = self.LitDataModule.n_dim,
#             latent_dim = kwargs['latent_dim'],
#             DiffEq_type = kwargs['DiffEq_type'],
#             potential_type = kwargs['potential_type'],
#         )
#         self.DiffEq = self.LitModelConfig(kwargs)

# #         if isinstance(func, NoneType):
# #             n_dim = self.LitDataModule.train_dataset.X.shape[-1]
# #             func = utils.default_NeuralSDE(n_dim)

# #         self.ModelConfig = configs.LightningModelConfiguration(
# #             func=func,
# #             optimizer=torch.optim.RMSprop,
# #             lr_scheduler=torch.optim.lr_scheduler.StepLR,
# #             adjoint=self.adjoint,
# #         )
                                                               
# #         self.DiffEq = self.ModelConfig(kwargs)
# #         self.DiffEq = lightning_models.LightningSDE(state_size = 50) # _LatentPotential(
# # #             func,
# # #             dt=self.dt,
# # #             lr=self.lr,
# # #             logqp=True,
# # #             step_size=self.step_size,
# # #             optimizer=torch.optim.RMSprop,
# # #             lr_scheduler=torch.optim.lr_scheduler.StepLR,
# # #         )
        
#         self.DiffEqLogger = utils.scDiffEqLogger(model_name=self.model_name)
#         self.DiffEqLogger()
#         self.TrainerGenerator = configs.LightningTrainerConfiguration(
#             self.DiffEqLogger.versioned_model_outdir
#         )
        

#     def fit(
#         self,
#         epochs=500,
#         callbacks=[],
#         ckpt_frequency: int = 25,
#         save_last_ckpt: bool = True,
#         keep_ckpts: int = -1,
#         monitor = None,
#         accelerator=None,
#         log_every_n_steps=1,
# #         swa_lrs=1e-8,
#         reload_dataloaders_every_n_epochs=1,
# #         gradient_clip_val=0.75,
#         devices=None,
#         deterministic=False,
#         **kwargs
#     ):  
#         # bring all kwargs / args into the same place
#         kwargs.update(locals())
#         kwargs.pop("kwargs")
        
#         if self.train_val_split[1] == 0:
#             kwargs.update({'check_val_every_n_epoch': 0, 'limit_val_batches': 0})
            
#         lr = self.lr

#         trainer_kwargs = utils.extract_func_kwargs(func=self.TrainerGenerator, kwargs=locals())
#         trainer_kwargs.update(utils.extract_func_kwargs(func=Trainer, kwargs=kwargs))

#         self.fit_trainer = self.TrainerGenerator(
#             max_epochs=epochs, stage = "fit", **trainer_kwargs
#         )
#         self.fit_trainer.fit(self.DiffEq, self.LitDataModule)

#     def test(self, callbacks=[]):
        
#         if hasattr(self, "fit_trainer"):
#             self.test_trainer = self.fit_trainer
#         else:
#             self.test_trainer = self.TrainerGenerator(stage="test", callbacks=callbacks)
        
#         self.test_outs = self.test_trainer.test(self.DiffEq, self.LitDataModule)
        
#         return self.test_outs
    
#     def simulate(self, X0, t, N = 1, device="cuda:0"):
#         return self.DiffEq.to(device).forward(X0.to(device), t.to(device))

#     def predict(self, t, t0_idx=None,  predict_key="predict", n=2000, callbacks=[]): # use_key="X_pca", 
        
#         # (1) Annotate self.LitDataModule.adata with a subset of cells to serve as prediction cells.
#         #     Re-configure the LitDataModule to serve us with predict_dataloader
#         tools.annotate_cells(self.LitDataModule.adata, idx=t0_idx, key=predict_key)
        
#         # (2) Configure trainer
#         if hasattr(self, "fit_trainer"):
#             self.predict_trainer = self.fit_trainer
#         else:
#             self.predict_trainer = self.TrainerGenerator(stage="predict", callbacks=callbacks, enable_progress_bar=False)
        
#         # (3) adjust t
#         self.DiffEq.t = t
        
#         # (4) loop over range(n) to make predictions
#         self.predictions = {}
#         for i in tqdm(range(n)):
#             predicted = self.predict_trainer.predict(self.DiffEq, self.LitDataModule)
#             self.predictions[i] = predicted
        
#         return self.predictions


#     def load(self, ckpt_path, freeze=True):
#         """
#         Loads a saved checkpoint file specified by ckpt_path into the DiffEq attribute of the input instance.
#         If freeze is True, DiffEq state is frozen to avoid parameter modification.


#         Parameters:
#         -----------
#         ckpt_path
#             Path to a saved checkpoint file.
#             type: str
        
#         freeze
#             indicates whether or not to freeze the DiffEq attribute after loading the checkpoint.
#             type: bool
#             default: True
        
#         Returns:
#         --------
#         None, modifies self.DiffEq.state_dict()
        
        
#         Notes:
#         ------
#         Assumes that self.DiffEq.func is the same composition as what you are trying to load.
#         """
        
#         self.DiffEq = self.DiffEq.load_from_checkpoint(ckpt_path, func=self.DiffEq.func)
#         if freeze:
#             self.DiffEq.freeze()
            
#     def __repr__(self):
#         # TODO: add description of model / training status / data / etc.
#         return "⏩ scDiffEq Model: {}".format(self.model_name)
