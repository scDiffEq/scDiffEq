

# -- import packages: ----------------------------------------------------------
from tqdm.notebook import tqdm
import lightning
import anndata
import pandas as pd
import torch
import glob
import os
import ABCParse
import pathlib
import autodevice

# -- import local dependencies: ------------------------------------------------
from . import configs, lightning_models, utils, callbacks
from .. import tools, __version__



import warnings

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")


# -- type setting: -------------------------------------------------------------
from typing import Union, List, Optional, Dict


import anndata
import ABCParse
from typing import Optional

class kNNMixIn(ABCParse.ABCParse, object):
    def __init__(self, *args, **kwargs):
        self.__parse__(locals())
        
    @property
    def _adata_kNN_fit(self):
        return self.adata[self.adata.obs[self._kNN_fit_subset]].copy()
        
    def _kNN_info_msg(self):
        
        self._INFO(f"Bulding Annoy kNN Graph on adata.obsm['{self._kNN_fit_subset}']")
        
    def configure_kNN(
        self,
        adata: Optional[anndata.AnnData] = None,
        kNN_key: Optional[str] = None,
        kNN_fit_subset: Optional[str] = None,
    ):
        
        """
        subset key should point to a col in adata.obs of bool vals
        """
        self.__update__(locals())
        
        self._kNN_info_msg()
        self._kNN = tools.kNN(
            adata = self._adata_kNN_fit, use_key = self._kNN_key,
        )

    @property
    def kNN(self):
        if not hasattr(self, "_kNN"):
            self.configure_kNN(
                adata = self._adata_kNN_fit,
                kNN_key = self._kNN_key,
                kNN_fit_subset = self._kNN_fit_subset,
            )
        return self._kNN

# -- model class, faces API: ---------------------------------------------------
class scDiffEq(kNNMixIn, ABCParse.ABCParse):
    def __init__(
        self,        
        
        # -- data params: -------------------------------------------------------
        adata: anndata.AnnData = None,
        latent_dim: int = 50,
        name: Optional[str] = None,
        use_key: str = "X_pca",
        obs_keys: List[str] = ["W"],
        seed: int = 0,
        backend: str = "auto",
        
        # -- kNN keys: [optional]: ----------------------------------------------
        build_kNN: Optional[bool] = False,
        kNN_key: Optional[str] = "X_pca",
        kNN_fit_subset: Optional[str] = "train",
        
        # -- pretrain params: ---------------------------------------------------
        pretrain_epochs: int = 500,
        pretrain_lr: float = 1e-3,
        pretrain_optimizer = torch.optim.Adam,
        pretrain_step_size: int = 100,
        pretrain_scheduler = torch.optim.lr_scheduler.StepLR,
        
        # -- train params: ------------------------------------------------------
        train_epochs: int = 1500,
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
        logger: Optional = None,
        num_workers: int = os.cpu_count(),
        silent: bool = True,
        scale_input_counts: bool = False,
        reduce_dimensions: bool = False,
        
        fate_bias_csv_path: Optional[Union[pathlib.Path, str]] = None,
        fate_bias_multiplier: float = 1,
        viz_frequency: int = 1,
        working_dir: Union[pathlib.Path, str] = os.getcwd(),
        
        # -- time params: -------------------------------------------------------
        time_key: Optional[str] = None,
        t0_idx: Optional[pd.Index] = None,
        t_min: float = 0,
        t_max: float = 1,
        dt: float = 0.1,
        time_cluster_key: Optional[str] = None,
        t0_cluster: Optional[str] = None,
        shuffle_time_labels: bool = False,
        
        # -- DiffEq params: ----------------------------------------------------
        mu_hidden: Union[List[int], int] = [400, 400],
        mu_activation: Union[str, List[str]] = "LeakyReLU",
        mu_dropout: Union[float, List[float]] = 0.1,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        
        sigma_hidden: Union[List[int], int] = [400, 400],
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
        potential_type: Union[None, str] = None,
        # other options: "fixed" or "prior"
        
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
        
        ckpt_path: Optional[Union[pathlib.Path, str]] = None,
        version: str = __version__,
        
        *args,
        **kwargs,
    ):
        self.__config__(locals())
        

    def _configure_dimension_reduction(self):
        self.reducer = tools.DimensionReduction(self.adata, save_path = "TESTING_SCDIFFEQ_MODEL")

        if self._scale_input_counts:
            self._INFO("Scaling input counts (for dimension reduction).")
            self.reducer.fit_scaler()
            self.adata.obsm["X_scaled_scDiffEq"] = self.reducer.X_scaled
        else:
            self.reducer.X_scaled_train = self.reducer.X_train
            self.reducer.X_scaled = self.reducer.X

        if self._reduce_dimensions:
            self.reducer.fit_pca()
            self.adata.obsm["X_pca_scDiffEq"] = self.reducer.X_pca
            self.reducer.fit_umap()
            self.adata.obsm["X_umap_scDiffEq"] = self.reducer.X_umap
            
            
    def _configure_CSVLogger(self):
        return lightning.pytorch.loggers.CSVLogger(
                save_dir=self._working_dir,
                name=self._name,
                version=None,
                prefix="",
                flush_logs_every_n_steps=1,
            )
    
    @property
    def logger(self):
        if not hasattr(self, "_LOGGER"):
            self._csv_logger = self._configure_CSVLogger()
            if hasattr(self, "_logger") and (not self._logger is None):
                    self._LOGGER = [self._csv_logger, self._logger]
            else:
                self._LOGGER = [self._csv_logger]

        return self._LOGGER

    @property
    def version(self) -> int:
        return self.logger[0].version
    
    @property
    def _metrics_path(self):
        return pathlib.Path(self.logger[0].log_dir).joinpath("metrics.csv")
    
    @property
    def metrics(self):
        return pd.read_csv(self._metrics_path)

    def _configure_trainer_generator(self):
        
        self.TrainerGenerator = configs.LightningTrainerConfiguration(self._name)
        self._PRETRAIN_CONFIG_COUNT = 0
        self._TRAIN_CONFIG_COUNT = 0

    def configure_model(self, DiffEq: Optional[lightning.LightningModule] = None):
        
        if DiffEq is None:
        
            self._LitModelConfig = configs.LightningModelConfiguration(
                data_dim=self._data_dim,
                latent_dim=self._latent_dim,
                DiffEq_type=self._DiffEq_type,
                potential_type=self._potential_type,
                fate_bias_csv_path = self._fate_bias_csv_path,
            )
            if hasattr(self, "reducer"):
                self._PARAMS['PCA'] = self.reducer.PCA
            DiffEq = self._LitModelConfig(self._PARAMS, self._ckpt_path)
            
        self.DiffEq = DiffEq
        
        self._name = self.DiffEq.hparams.name
        self._INFO(f"Using the specified parameters, {self.DiffEq} has been called.")
        self._component_loader = utils.FlexibleComponentLoader(self)
        
        lightning.seed_everything(self._seed)
        
        # -- Step 5: configure bridge to lightning logger ----------------------
        # was its own step before: now in-line here, since it
        # doesn't make sense to separate it, functionally
        self._LOGGING = utils.LoggerBridge(self.DiffEq)
        self._configure_trainer_generator()

    def configure_data(self, adata: anndata.AnnData):
        
        """Step 3. Can be called internally or externally."""
        
        self.adata = adata.copy()
        
        self._DATA_CONFIG = configs.DataConfiguration()
        self._DATA_CONFIG(scDiffEq = self)
        self._INFO(f"Input data configured.")
    
    def __config__(self, kwargs):

        """
        Run on model.__init__()

        Step 1: Parse kwargs, Set up info messaging
        Step 2: Configure data
            If adata is passed, you can do the remaining steps
            Step 3: Configure kNN [ optional ]
            Step 4: Configure model
            Step 5: Configure Logging/Trainer
            Step 6: extras (dimension reduce)
        """

        # -- Step 1: parse kwargs, set up info msg -----------------------------
        self.__parse__(kwargs, public = [None], ignore=["adata"])

        # -- TODO: eventually replace this with more sophisticated logging -----
        self._INFO = utils.InfoMessage()

        # -- Step 2: configure data ------------------------------------------
        if not kwargs['adata'] is None:
            self.configure_data(adata = kwargs['adata'])
            
            # -- Step 3: configure kNN ---------------------------------------
            if self._PARAMS["build_kNN"]:
                self._PARAMS['kNN'] = self.kNN

            # -- Step 4: configure model -------------------------------------------
            self.configure_model(DiffEq = None)

            # -- Step 6: extras: ---------------------------------------------------
#             if kwargs["reduce_dimensions"]:
#                 self._configure_dimension_reduction()

    def to(self, device):
        self.DiffEq = self.DiffEq.to(device)

    def freeze(self):
        """Freeze lightning model"""
        self.DiffEq.freeze()
        
    def load(self, ckpt_path: Union[str, pathlib.Path], freeze=True, device = autodevice.AutoDevice()):
        
        self.__update__(locals())
        
        self.DiffEq = self.DiffEq.load_from_checkpoint(self._ckpt_path)
        self.DiffEq = self.DiffEq.to(self._device)
        if freeze:
            self.freeze()

#     def _stage_log_path(self, stage):
#         log_path = glob.glob(self.DiffEqLogger.VERSIONED_MODEL_OUTDIR + f"/{stage}*")[0]
#         self._INFO(f"Access logs at: {log_path}")
        
#     @property
#     def _VERSION(self):
#         return int(os.path.basename(self.DiffEqLogger.VERSIONED_MODEL_OUTDIR).split("_")[1])
    
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
            ignore = ['version', 'working_dir'],
        )
        trainer_kwargs.update(
            utils.extract_func_kwargs(
                func=lightning.Trainer,
                kwargs=self._PARAMS,
                ignore = ['version', 'working_dir'],
            )
        )
        trainer_kwargs.update(
            utils.extract_func_kwargs(
                func=lightning.Trainer,
                kwargs=locals(),
                ignore = ['version', 'working_dir'],
            )
        )
        trainer_kwargs = self._check_disable_validation(trainer_kwargs)
        
        self.pre_trainer = self.TrainerGenerator(
            max_epochs=self._pretrain_epochs,
            stage=STAGE,
            working_dir = self.DiffEqLogger._WORKING_DIR,
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
        """
        If any of the keyword arguments are passed, they will replace the previously-stated
        arguments from __init__ and re-configure the DiffEq.
        """

        self._configure_pretrain_step(epochs, pretrain_callbacks)
        self.pre_trainer.fit(self.DiffEq, self.LitDataModule)

    def _configure_train_step(self, epochs, kwargs):
                    
        STAGE = "train"
        kwargs.update(self._PARAMS)
        kwargs['callbacks'] = kwargs.pop("train_callbacks")

        self._INFO(f"Configuring fit step: {STAGE}")
        
        self.DiffEq._update_lit_diffeq_hparams(self._PARAMS)
        
        ignore = ['version', 'working_dir', 'logger']
        funcs = [self.TrainerGenerator, lightning.Trainer]
        
        trainer_kwargs = {}
        
        for func in funcs:
            trainer_kwargs.update(
                utils.extract_func_kwargs(
                    func = func, kwargs = kwargs, ignore = ignore,
                )
            )
            

#         trainer_kwargs = utils.extract_func_kwargs(
#             func=self.TrainerGenerator,
#             kwargs=self._PARAMS,
#             ignore = ignore,
#         )
#         trainer_kwargs.update(
#             utils.extract_func_kwargs(
#                 func=lightning.Trainer,
#                 kwargs=self._PARAMS,
#                 ignore = ignore,
#             )
#         )
#         trainer_kwargs.update(
#             utils.extract_func_kwargs(
#                 func=lightning.Trainer,
#                 kwargs=kwargs,
#                 ignore = ignore,
#             )
#         )
        
        self.trainer_kwargs = self._check_disable_validation(trainer_kwargs)
        
        self.trainer = self.TrainerGenerator(
            logger = self.logger,
            max_epochs=self._train_epochs,
            stage=STAGE,
            working_dir = self._working_dir,
            version = self.version,
            pretrain_version=self._PRETRAIN_CONFIG_COUNT,
            train_version=self._TRAIN_CONFIG_COUNT,
#             callbacks = train_callbacks,
            **self.trainer_kwargs,
        )
        
        if hasattr(self, "_csv_logger"):
            DIR = self._csv_logger.log_dir
            self._INFO(f"Logging locally to: {DIR}")
        
        
        self._TRAIN_CONFIG_COUNT += 1

#         self._stage_log_path(STAGE)

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
        devices=1,
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
#         self._TRAIN_CONFIG_COUNT += 1
        
    @property
    def tracker(self):
        return callbacks.ModelTracker(version=self.version)


    def cell_potential(
        self,
        use_key: str = 'X_pca',
        raw_key_added: str = '_psi',
        norm_key_added: str = 'psi',
        device: Union[str, torch.device] = torch.device('cuda:0'),
        seed: int = 0,
        normalize: bool = True,
        return_raw_array: bool = False,
        q: float = 0.05,
        knn_smoothing_iters: int = 5,
        use_tqdm: bool = True,
    ):

        tools.cell_potential(
            adata = self.adata,
            model = self,
            use_key = use_key,
            raw_key_added = raw_key_added,
            norm_key_added = norm_key_added,
            device = device,
            seed = seed,
            normalize = normalize,
            return_raw_array = return_raw_array,
            q = q,
            knn_smoothing_iters = knn_smoothing_iters,
            use_tqdm = use_tqdm,
        )

            

# if you passed adata, configure the rest of the model here:
        # -- 1a. parse adata: --------------------------------------------------
#             self._DATA_CONFIG = configs.DataConfiguration()
#             self._DATA_CONFIG(
#                 adata = self.adata, scDiffEq = self, scDiffEq_kwargs = self._PARAMS,
#             )
        
#     # -- could cordon off into a kNNMixIn: ----
#     def _configure_kNN_graph(self):      
#         train_adata = self.adata[self.adata.obs[self._train_key]].copy()
#         self._INFO(f"Bulding Annoy kNN Graph on adata.obsm['{self._kNN_key}']")        
#         self._kNN_graph = tools.kNN(adata = train_adata, use_key = self._kNN_key)
        
#     @proprety
#     def kNN(self):
#         if not hasattr(self, "_kNN_graph"):
#             self._configure_kNN_graph()
#         return self._kNN_graph
#     # -----------------------------------------



#     @property
#     def loss(self):
#         utils.display_tracked_loss(self.logger)
        
# -- ADD AS MIXINS: ----
#     def load_DiffEq_from_ckpt(self, ckpt_path):
        
#         self._component_loader.load_DiffEq_state(ckpt_path)
#         self._PARAMS[
#             "diffeq_ckpt_path"
#         ] = self._diffeq_ckpt_path = self._component_loader._diffeq_ckpt_path
#         self.DiffEq._update_lit_diffeq_hparams(self._PARAMS)
        
                
#     def load_encoder_from_ckpt(self, ckpt_path):
        
#         self._component_loader.load_encoder_state(ckpt_path)
#         self._PARAMS[
#             "encoder_ckpt_path"
#         ] = self._encoder_ckpt_path = self._component_loader._encoder_ckpt_path
#         self.DiffEq._update_lit_diffeq_hparams(self._PARAMS)
        
        
#     def load_decoder_from_ckpt(self, ckpt_path):
        
#         self._component_loader.load_decoder_state(ckpt_path)
#         self._PARAMS[
#             "decoder_ckpt_path"
#         ] = self._decoder_ckpt_path = self._component_loader._decoder_ckpt_path
#         self.DiffEq._update_lit_diffeq_hparams(self._PARAMS)
        
#     def load_VAE_from_ckpt(self, ckpt_path):
#         # TODO: add ability to freeze these once loaded
        
#         self.load_encoder_from_ckpt(ckpt_path)
#         self.load_decoder_from_ckpt(ckpt_path)
