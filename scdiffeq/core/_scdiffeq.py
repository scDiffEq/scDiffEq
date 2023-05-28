

# -- import packages: ----------------------------------------------------------
from tqdm.notebook import tqdm
import lightning
import anndata
import pandas as pd
import torch
import glob
import os


# -- import local dependencies: ------------------------------------------------
from . import configs, lightning_models, utils, callbacks
from .. import tools, __version__


# -- type setting: -------------------------------------------------------------
from typing import Union, List
NoneType = type(None)

import warnings

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

class scDiffEq(utils.ABCParse):
    def __init__(
        self,
        
        # -- data params: -------------------------------------------------------
        adata: anndata.AnnData,
        latent_dim: int = 20,
        model_name: str = "scDiffEq_model",
        use_key: str = "X_scaled",
        obs_keys: List[str] = ["W"],
        kNN_key: str = "X_pca_scDiffEq",
        seed: int = 0,
        
        # -- pretrain params: ---------------------------------------------------
        pretrain_epochs: int = 500,
        pretrain_lr: float = 1e-3,
        pretrain_optimizer=torch.optim.Adam,
        pretrain_step_size: int = 100,
        pretrain_scheduler=torch.optim.lr_scheduler.StepLR,
        
        # -- train params: ------------------------------------------------------
        train_epochs: int =1500,
        train_lr: float = 1e-5,
        train_optimizer = torch.optim.RMSprop,
        train_scheduler = torch.optim.lr_scheduler.StepLR,
        train_step_size: int = 10,
        train_val_split: List[float] = [0.9, 0.1],
        batch_size: int = 2000,
        
        train_key: str = "train",
        val_key: str = "val",
        test_key: str = "test",
        predict_key: str = "predict",
        
        # -- general params: ----------------------------------------------------
        num_workers: int = os.cpu_count(),
        silent: bool = True,
        scale_input_counts: bool = True,
        reduce_dimensions: bool = True,
        build_kNN: bool = True,
        fate_bias_csv_path: Union[str, NoneType] = None,
        fate_bias_multiplier: float = 1,
        viz_frequency: int = 1,
        
        # -- time params: -------------------------------------------------------
        time_key: Union[str, NoneType] = None,
        t0_idx: Union[pd.Index, NoneType] = None,
        t_min: float = 0,
        t_max: float = 1,
        dt: float = 0.1,
        time_cluster_key=None,
        t0_cluster=None,
        shuffle_time_labels = False,
        
        # -- DiffEq params: ----------------------------------------------------
        mu_hidden: Union[List[int], int] = [400, 400, 400],
        mu_activation: Union[str, List[str]] = "LeakyReLU",
        mu_dropout: Union[float, List[float]] = 0.1,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        
        sigma_hidden: Union[List[int], int] = [400, 400, 400],
        sigma_activation: Union[str, List[str]] = "LeakyReLU",
        sigma_dropout: Union[float, List[float]] = 0.1,
        sigma_bias: List[bool] = True,
        sigma_output_bias: bool = True,
        sigma_n_augment: int = 0,
        
        adjoint: bool = False,
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
        
        version: str = __version__,
        
        *args,
        **kwargs,
    ):
        self.__config__(locals())

    def _configure_obs_idx(self):
        
        self._PROVIDED_OBS_IDX = self.adata.obs.index
        self._PROVIDED_VAR_IDX = self.adata.var.index
        
        if self.adata.obs.index[0] != "1":
            self.adata = utils.idx_to_int_str(self.adata)
        
        
    def _configure_data(self, kwargs):

        """Configure data (including time time)"""
        
        self._configure_obs_idx()
        
        self.t, self.t_config = configs.configure_time(
            **utils.extract_func_kwargs(func = configs.configure_time, kwargs = kwargs),
        )
        kwargs['groupby'] = self.t_config.attributes['time_key']
        kwargs.update(self.t_config.attributes)
        self.LitDataModule = configs.LightningData(
            **utils.extract_func_kwargs(
                func=configs.LightningData,
                kwargs=kwargs,
            )
        )
        self._data_dim = self.LitDataModule.n_dim

    def _configure_dimension_reduction(self):
        self.reducer = tools.DimensionReduction(self.adata, save_path = self.DiffEqLogger.default_model_outdir)
        if self._scale_input_counts:
            self._INFO("Scaling input counts (for dimension reduction).")
            self.reducer.fit_scaler()
            self.adata.obsm["X_scaled_scDiffEq"] = self.reducer.X_scaled
        else:
            self.reducer.X_scaled_train = reducer.X_train
            self.reducer.X_scaled = reducer.X

        if self._reduce_dimensions:
            self.reducer.fit_pca()
            self.adata.obsm["X_pca_scDiffEq"] = self.reducer.X_pca
            self.reducer.fit_umap()
            self.adata.obsm["X_umap_scDiffEq"] = self.reducer.X_umap

    def _configure_logger(self):

        self.DiffEqLogger = utils.scDiffEqLogger(model_name=self._model_name)
        self.DiffEqLogger()

    def _configure_trainer_generator(self):
        
        self.TrainerGenerator = configs.LightningTrainerConfiguration(
            self.DiffEqLogger.versioned_model_outdir
        )
        self._PRETRAIN_CONFIG_COUNT = 0
        self._TRAIN_CONFIG_COUNT = 0

    def _configure_kNN_graph(self):
        
        # -- prep data: ------
        train_adata = self.adata[self.adata.obs[self._train_key]].copy()
        
        # -- doesn't seem necessary any longer: ------
#         train_adata.obs = train_adata.obs.reset_index(drop=True)
#         train_adata.obs.index = train_adata.obs.index.astype(str)
        
        self._INFO(f"Bulding Annoy kNN Graph on adata.obsm['{self._kNN_key}']")
        self.kNN_Graph = tools.kNN(adata = train_adata, use_key = self._kNN_key)

    def _configure_model(self, kwargs):

        self._LitModelConfig = configs.LightningModelConfiguration(
            data_dim=self._data_dim,
            latent_dim=self._latent_dim,
            DiffEq_type=self._DiffEq_type,
            potential_type=self._potential_type,
            fate_bias_csv_path = self._fate_bias_csv_path,
        )
        if hasattr(self, "reducer"):
            kwargs['PCA'] = self.reducer.PCA
        
        if hasattr(self, "kNN_Graph"):
            kwargs['kNN_Graph'] = self.kNN_Graph

        self.DiffEq = self._LitModelConfig(kwargs)
        self._INFO(f"Using the specified parameters, {self.DiffEq} has been called.")
        self._component_loader = utils.FlexibleComponentLoader(self)
    
    def __config__(self, kwargs):

        """
        Run on model.__init__()

        Step 0: Parse all kwargs
        Step 1: Set up info messaging [TODO: eventually replace this with more sophisticated logging]
        Step 2: Configure data
        Step 4: Configure dimension reduction models
        Step 5: Configure kNN Graph.
        Step 3: Configure lightning model
        Step 6: Configure logger
        Step 7: Configure TrainerGenerator
        """

        self.__parse__(kwargs, public=["adata"])
        self._INFO = utils.InfoMessage()
        self._configure_data(kwargs)
        self._configure_logger()
        if kwargs["reduce_dimensions"]:
            self._configure_dimension_reduction()
        if kwargs["build_kNN"]:
            self._configure_kNN_graph()
        self._configure_model(kwargs)
        self._configure_trainer_generator()
        
        lightning.seed_everything(self._seed)
        
    def to(self, device):
        self.DiffEq.to(device)

    def freeze(self):
        """Freeze lightning model"""
        self.DiffEq.freeze()
               
    def load_DiffEq_from_ckpt(self, ckpt_path):
        
        self._component_loader.load_DiffEq_state(ckpt_path)
        self._PARAMS[
            "diffeq_ckpt_path"
        ] = self._diffeq_ckpt_path = self._component_loader._diffeq_ckpt_path
        self.DiffEq._update_lit_diffeq_hparams(self._PARAMS)
        
                
    def load_encoder_from_ckpt(self, ckpt_path):
        
        self._component_loader.load_encoder_state(ckpt_path)
        self._PARAMS[
            "encoder_ckpt_path"
        ] = self._encoder_ckpt_path = self._component_loader._encoder_ckpt_path
        self.DiffEq._update_lit_diffeq_hparams(self._PARAMS)
        
        
    def load_decoder_from_ckpt(self, ckpt_path):
        
        self._component_loader.load_decoder_state(ckpt_path)
        self._PARAMS[
            "decoder_ckpt_path"
        ] = self._decoder_ckpt_path = self._component_loader._decoder_ckpt_path
        self.DiffEq._update_lit_diffeq_hparams(self._PARAMS)
        
    def load_VAE_from_ckpt(self, ckpt_path):
        # TODO: add ability to freeze these once loaded
        
        self.load_encoder_from_ckpt(ckpt_path)
        self.load_decoder_from_ckpt(ckpt_path)

    def load(self, ckpt_path, freeze=True):
        self.ckpt_path = ckpt_path
        self.DiffEq = self.DiffEq.load_from_checkpoint(ckpt_path)
        if freeze:
            self.DiffEq.freeze()

    def _stage_log_path(self, stage):
        log_path = glob.glob(self.DiffEqLogger.versioned_model_outdir + f"/{stage}*")[0]
        self._INFO(f"Access logs at: {log_path}")
        
    @property
    def _VERSION(self):
        return int(os.path.basename(self.DiffEqLogger.versioned_model_outdir).split("_")[1])
    
    def _check_disable_validation(self, trainer_kwargs):
        if self._train_val_split[1] == 0:
            trainer_kwargs.update(
                {
                    'check_val_every_n_epoch': 0,
                    'limit_val_batches': 0.0,
                    'num_sanity_val_steps': 0.0,
                    'val_check_interval': 0.0,
                },

            )
            
        return trainer_kwargs
            
    def _configure_pretrain_step(self, epochs, callbacks=[]):
        
        STAGE = "pretrain"
        self._INFO(f"Configuring fit step: {STAGE}")
        
        self.DiffEq._update_lit_diffeq_hparams(self._PARAMS)

        trainer_kwargs = utils.extract_func_kwargs(
            func=self.TrainerGenerator,
            kwargs=self._PARAMS,
            ignore = ['version'],
        )
        trainer_kwargs.update(
            utils.extract_func_kwargs(
                func=lightning.Trainer,
                kwargs=self._PARAMS,
                ignore = ['version'],
            )
        )
        trainer_kwargs.update(
            utils.extract_func_kwargs(
                func=lightning.Trainer,
                kwargs=locals(),
                ignore = ['version'],
            )
        )
        trainer_kwargs = self._check_disable_validation(trainer_kwargs)
        
        self.pre_trainer = self.TrainerGenerator(
            max_epochs=self._pretrain_epochs,
            stage=STAGE,
            working_dir = self.DiffEqLogger.wd,
            version = self._VERSION,
            pretrain_version=self._PRETRAIN_CONFIG_COUNT,
            train_version=self._TRAIN_CONFIG_COUNT,
            **trainer_kwargs
        )
        self._stage_log_path(STAGE)

    def pretrain(
        self,
        epochs=None,
        pretrain_callbacks = [],
    ):
        """If any of the keyword arguments are passed, they will replace the previously-stated arguments from __init__ and re-configure the DiffEq."""

        self._configure_pretrain_step(epochs, pretrain_callbacks)
        self.pre_trainer.fit(self.DiffEq, self.LitDataModule)

    def _configure_train_step(self, epochs, kwargs):
                    
        STAGE = "train"
        
        kwargs['callbacks'] = kwargs.pop("train_callbacks")

        self._INFO(f"Configuring fit step: {STAGE}")
        
        self.DiffEq._update_lit_diffeq_hparams(self._PARAMS)

        trainer_kwargs = utils.extract_func_kwargs(
            func=self.TrainerGenerator,
            kwargs=self._PARAMS,
            ignore = ['version'],
        )
        trainer_kwargs.update(
            utils.extract_func_kwargs(
                func=lightning.Trainer,
                kwargs=self._PARAMS,
                ignore = ['version'],
            )
        )
        trainer_kwargs.update(
            utils.extract_func_kwargs(
                func=lightning.Trainer,
                kwargs=kwargs,
                ignore = ['version'],
            )
        )
        
        trainer_kwargs = self._check_disable_validation(trainer_kwargs)
        
        self.trainer = self.TrainerGenerator(
            max_epochs=self._train_epochs,
            stage=STAGE,
            working_dir = self.DiffEqLogger.wd,
            version = self._VERSION,
            pretrain_version=self._PRETRAIN_CONFIG_COUNT,
            train_version=self._TRAIN_CONFIG_COUNT,
            **trainer_kwargs,
        )

        self._stage_log_path(STAGE)

    def train(
        self,
        epochs=500,
        train_callbacks=[],
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
        
        self.DiffEq._update_lit_diffeq_hparams(self._PARAMS)
        self._configure_train_step(epochs, locals())
        self.trainer.fit(self.DiffEq, self.LitDataModule)

    def fit(
        self,
        train_epochs=200,
        pretrain_epochs=500,
        train_lr = None,
        pretrain_callbacks: List = [],
        train_callbacks: List = [],
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
                
        if pretrain_epochs > 0 and (not self._LitModelConfig.use_vae):
            pretrain_epochs = 0
            
            
        self.__update__(locals())

        if pretrain_epochs > 0:
            self.pretrain(
                epochs=pretrain_epochs, **utils.extract_func_kwargs(self.pretrain, locals()))
        if train_epochs > 0:
            self.train(
                epochs=train_epochs, **utils.extract_func_kwargs(self.train, locals())
            )
            
        # TO-DO: eventually replace how this works...
        self._PRETRAIN_CONFIG_COUNT += 1
        self._TRAIN_CONFIG_COUNT += 1
        
    @property
    def tracker(self):
        return callbacks.ModelTracker(version=self._VERSION)


    @property
    def loss(self):
        utils.display_tracked_loss(self.DiffEqLogger)
        
