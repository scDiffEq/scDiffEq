
# -- type setting: -------------------------------------------------------------
from typing import Union, List, Optional, Dict


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
from . import _mix_ins as mix_ins
from .. import __version__ # tools

import warnings

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")


# -- model class, faces API: ---------------------------------------------------
class scDiffEq(
    mix_ins.ModelConfigMixIn,
    mix_ins.kNNMixIn,
    mix_ins.StateCharacterizationMixIn,
    mix_ins.UtilityIOMixIn,
    mix_ins.LoggingMixIn,
    mix_ins.PreTrainMixIn,
    mix_ins.TrainMixIn,
    ABCParse.ABCParse,
):
        
    """scDiffeq model class"""
    def __init__(
        self,
        # -- data params: -------------------------------------------------------
        adata: Optional[anndata.AnnData] = None,
        latent_dim: int = 50,
        name: Optional[str] = None,
        use_key: str = "X_pca",
        weight_key: str = 'W',
        obs_keys: List[str] = [],
        seed: int = 0,
        backend: str = "auto",
        gradient_clip_val: float = 0.5,
        # -- velocity ratio keys: -----------------------------------------------
        velocity_ratio_target: float = 0,  # off by default
        velocity_ratio_enforce: float = 0, # off by default
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
        logger: Optional["lightning.logger"] = None,
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
    ) -> None:

        """Initialize the scDiffEq model.

        This class is responsible for bringing together three critical
        components: the lightning model, the lightning trainer, and the
        lightning data. All else is superfluous to model operation. 

        Args:
            arg1 (arg1_type): description of arg1. **Default**: None
        
        Returns:
            None
        """
        self.__config__(locals())
        
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
    ) -> None:

        """Fit the scDiffEq model to some data.

        Extended description.

        Args:
            arg1 (arg1_type): description of arg1. **Default**: None
        
        Returns:
            None
        """
        
        self.train(**ABCParse.function_kwargs(self.train, locals()))


    def simulate(self) -> anndata.AnnData:
        ...
        
    def __repr__(self) -> str:
        return "scDiffEq"
