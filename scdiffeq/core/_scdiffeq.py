# -- import packages: ---------------------------------------------------------
import ABCParse
import anndata
import lightning
import logging
import os
import pandas as pd
import pathlib
import torch
import warnings

# -- import local dependencies: -----------------------------------------------
from . import _mix_ins as mix_ins
from .. import __version__


# -- set type hints: ----------------------------------------------------------
from typing import Dict, List, Literal, Optional, Union


# -- setup logging: -----------------------------------------------------------
logger = logging.getLogger(__name__)

# -- remove specific unnecessary warnings from dependency: --------------------
warnings.filterwarnings(
    "ignore",
    ".*Consider increasing the value of the `num_workers` argument*",
)


# -- model class, faces API: --------------------------------------------------
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
    """scDiffeq model class

    Parameters:
    ----------
    adata : Optional[anndata.AnnData], optional
        Annotated data matrix, by default None
    latent_dim : int, optional
        Number of latent dimensions, by default 50
    name : Optional[str], optional
        Model name, by default None
    use_key : str, optional
        Key to use for data, by default "X_pca"
    weight_key : str, optional
        Key to use for weights, by default "W"
    obs_keys : List[str], optional
        List of observation keys, by default []
    seed : int, optional
        Random seed, by default 0
    backend : str, optional
        Backend to use, by default "auto"
    gradient_clip_val : float, optional
        Gradient clipping value, by default 0.5
    velocity_ratio_params : Dict[str, Union[float, bool]], optional
        Parameters for velocity ratio, by default {"target": 2, "enforce": 100, "method": "square"}
    build_kNN : Optional[bool], optional
        Whether to build kNN, by default False
    kNN_key : Optional[str], optional
        Key to use for kNN, by default "X_pca"
    kNN_fit_subset : Optional[str], optional
        Subset to fit kNN, by default "train"
    pretrain_epochs : int, optional
        Number of pretrain epochs, by default 500
    pretrain_lr : float, optional
        Learning rate for pretraining, by default 1e-3
    pretrain_optimizer : torch.optim.Optimizer, optional
        Optimizer for pretraining, by default torch.optim.Adam
    pretrain_step_size : int, optional
        Step size for pretraining scheduler, by default 100
    pretrain_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Scheduler for pretraining, by default torch.optim.lr_scheduler.StepLR
    train_epochs : int, optional
        Number of training epochs, by default 1500
    train_lr : float, optional
        Learning rate for training, by default 1e-5
    train_optimizer : torch.optim.Optimizer, optional
        Optimizer for training, by default torch.optim.RMSprop
    train_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Scheduler for training, by default torch.optim.lr_scheduler.StepLR
    train_step_size : int, optional
        Step size for training scheduler, by default 10
    train_val_split : List[float], optional
        Train-validation split, by default [0.9, 0.1]
    batch_size : int, optional
        Batch size, by default 2000
    train_key : str, optional
        Key for training data, by default "train"
    val_key : str, optional
        Key for validation data, by default "val"
    test_key : str, optional
        Key for test data, by default "test"
    predict_key : str, optional
        Key for prediction data, by default "predict"
    logger : Optional["lightning.logger"], optional
        Logger, by default None
    num_workers : int, optional
        Number of workers, by default os.cpu_count()
    silent : bool, optional
        Whether to silence output, by default True
    scale_input_counts : bool, optional
        Whether to scale input counts, by default False
    reduce_dimensions : bool, optional
        Whether to reduce dimensions, by default False
    fate_bias_csv_path : Optional[Union[pathlib.Path, str]], optional
        Path to fate bias CSV, by default None
    fate_bias_multiplier : float, optional
        Multiplier for fate bias, by default 1
    viz_frequency : int, optional
        Frequency of visualization, by default 1
    working_dir : Union[pathlib.Path, str], optional
        Working directory, by default os.getcwd()
    time_key : Optional[str], optional
        Key for time data, by default None
    t0_idx : Optional[pd.Index], optional
        Index for initial time, by default None
    t_min : float, optional
        Minimum time, by default 0
    t_max : float, optional
        Maximum time, by default 1
    dt : float, optional
        Time step, by default 0.1
    time_cluster_key : Optional[str], optional
        Key for time cluster, by default None
    t0_cluster : Optional[str], optional
        Initial time cluster, by default None
    shuffle_time_labels : bool, optional
        Whether to shuffle time labels, by default False
    mu_hidden : Union[List[int], int], optional
        Hidden layers for drift function, by default [400, 400]
    mu_activation : Union[str, List[str]], optional
        Activation function for drift, by default "LeakyReLU"
    mu_dropout : Union[float, List[float]], optional
        Dropout rate for drift, by default 0.1
    mu_bias : bool, optional
        Whether to use bias in drift, by default True
    mu_output_bias : bool, optional
        Whether to use output bias in drift, by default True
    mu_n_augment : int, optional
        Number of augmented dimensions for drift, by default 0
    sigma_hidden : Union[List[int], int], optional
        Hidden layers for diffusion function, by default [400, 400]
    sigma_activation : Union[str, List[str]], optional
        Activation function for diffusion, by default "LeakyReLU"
    sigma_dropout : Union[float, List[float]], optional
        Dropout rate for diffusion, by default 0.1
    sigma_bias : List[bool], optional
        Whether to use bias in diffusion, by default True
    sigma_output_bias : bool, optional
        Whether to use output bias in diffusion, by default True
    sigma_n_augment : int, optional
        Number of augmented dimensions for diffusion, by default 0
    adjoint : bool, optional
        Whether to use adjoint method, by default False
    sde_type : str, optional
        Type of SDE, by default "ito"
    noise_type : str, optional
        Type of noise, by default "general"
    brownian_dim : int, optional
        Dimension of Brownian motion, by default 1
    coef_drift : float, optional
        Coefficient for drift, by default 1.0
    coef_diffusion : float, optional
        Coefficient for diffusion, by default 1.0
    coef_prior_drift : float, optional
        Coefficient for prior drift, by default 1.0
    DiffEq_type : str, optional
        Type of differential equation, by default "SDE"
    potential_type : Union[None, str], optional
        Type of potential, by default None
    encoder_n_hidden : int, optional
        Number of hidden layers for encoder, by default 4
    encoder_power : float, optional
        Power for encoder, by default 2
    encoder_activation : Union[str, List[str]], optional
        Activation function for encoder, by default "LeakyReLU"
    encoder_dropout : Union[float, List[float]], optional
        Dropout rate for encoder, by default 0.2
    encoder_bias : bool, optional
        Whether to use bias in encoder, by default True
    encoder_output_bias : bool, optional
        Whether to use output bias in encoder, by default True
    decoder_n_hidden : int, optional
        Number of hidden layers for decoder, by default 4
    decoder_power : float, optional
        Power for decoder, by default 2
    decoder_activation : Union[str, List[str]], optional
        Activation function for decoder, by default "LeakyReLU"
    decoder_dropout : Union[float, List[float]], optional
        Dropout rate for decoder, by default 0.2
    decoder_bias : bool, optional
        Whether to use bias in decoder, by default True
    decoder_output_bias : bool, optional
        Whether to use output bias in decoder, by default True
    ckpt_path : Optional[Union[pathlib.Path, str]], optional
        Path to checkpoint, by default None
    version : str, optional
        Version of the model, by default __version__
    """

    def __init__(
        self,
        # -- data params: ------------------------------------------------------
        adata: Optional[anndata.AnnData] = None,
        latent_dim: int = 50,
        name: Optional[str] = None,
        use_key: str = "X_pca",
        weight_key: str = "W",
        obs_keys: List[str] = [],
        seed: int = 0,
        backend: str = "auto",
        gradient_clip_val: float = 0.5,
        # -- velocity ratio keys: ----------------------------------------------
        velocity_ratio_params: Dict[str, Union[float, bool]] = {
            "target": 2,
            "enforce": 100,  # zero to disable
            "method": "square",  # abs -> calls torch.abs or torch.square
        },
        # -- kNN keys: [optional]: ----------------------------------------------
        build_kNN: Optional[bool] = False,
        kNN_key: Optional[str] = "X_pca",
        kNN_fit_subset: Optional[str] = "train",
        # -- pretrain params: --------------------------------------------------
        pretrain_epochs: int = 500,
        pretrain_lr: float = 1e-3,
        pretrain_optimizer=torch.optim.Adam,
        pretrain_step_size: int = 100,
        pretrain_scheduler=torch.optim.lr_scheduler.StepLR,
        # -- train params: -----------------------------------------------------
        train_epochs: int = 2500,
        train_lr: float = 1e-5,
        train_optimizer=torch.optim.RMSprop,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        train_step_size: int = 10,
        train_val_split: List[float] = [0.9, 0.1],
        batch_size: int = 2048,
        train_key: str = "train",
        val_key: str = "val",
        test_key: str = "test",
        predict_key: str = "predict",
        # -- general params: ---------------------------------------------------
        logger: Optional["lightning.logger"] = None,
        num_workers: int = 0,
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
        # -- DiffEq params: ---------------------------------------------------
        mu_hidden: Union[List[int], int] = [512, 512],
        mu_activation: Union[str, List[str]] = "LeakyReLU",
        mu_dropout: Union[float, List[float]] = 0.1,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        sigma_hidden: Union[List[int], int] = [32, 32],
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
        potential_type: Literal["fixed", "prior"] = "fixed",
        # -- Encoder params: ---------------------------------------------------
        encoder_n_hidden: int = 4,
        encoder_power: float = 2,
        encoder_activation: Union[str, List[str]] = "LeakyReLU",
        encoder_dropout: Union[float, List[float]] = 0.2,
        encoder_bias: bool = True,
        encoder_output_bias: bool = True,
        # -- Decoder params: --------------------------------------------------
        decoder_n_hidden: int = 4,
        decoder_power: float = 2,
        decoder_activation: Union[str, List[str]] = "LeakyReLU",
        decoder_dropout: Union[float, List[float]] = 0.2,
        decoder_bias: bool = True,
        decoder_output_bias: bool = True,
        ckpt_path: Optional[Union[pathlib.Path, str]] = None,
        monitor_hardware: bool = False,
        version: str = __version__,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the scDiffEq model.

        This class is responsible for bringing together three critical
        components: the lightning model, the lightning trainer, and the
        lightning data. All else is superfluous to model operation.

        Args:
            adata (Optional[anndata.AnnData], optional): Annotated data matrix. Default is None.
            latent_dim (int, optional): Number of latent dimensions. Default is 50.
            name (Optional[str], optional): Model name. Default is None.
            use_key (str, optional): Key to use for data. Default is "X_pca".
            weight_key (str, optional): Key to use for weights. Default is "W".
            obs_keys (List[str], optional): List of observation keys. Default is [].
            seed (int, optional): Random seed. Default is 0.
            backend (str, optional): Backend to use. Default is "auto".
            gradient_clip_val (float, optional): Gradient clipping value. Default is 0.5.
            velocity_ratio_params (Dict[str, Union[float, bool]], optional): Parameters for velocity ratio. Default is {"target": 2, "enforce": 100, "method": "square"}.
            build_kNN (Optional[bool], optional): Whether to build kNN. Default is False.
            kNN_key (Optional[str], optional): Key to use for kNN. Default is "X_pca".
            kNN_fit_subset (Optional[str], optional): Subset to fit kNN. Default is "train".
            pretrain_epochs (int, optional): Number of pretrain epochs. Default is 500.
            pretrain_lr (float, optional): Learning rate for pretraining. Default is 1e-3.
            pretrain_optimizer (torch.optim.Optimizer, optional): Optimizer for pretraining. Default is torch.optim.Adam.
            pretrain_step_size (int, optional): Step size for pretraining scheduler. Default is 100.
            pretrain_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler for pretraining. Default is torch.optim.lr_scheduler.StepLR.
            train_epochs (int, optional): Number of training epochs. Default is 1500.
            train_lr (float, optional): Learning rate for training. Default is 1e-5.
            train_optimizer (torch.optim.Optimizer, optional): Optimizer for training. Default is torch.optim.RMSprop.
            train_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler for training. Default is torch.optim.lr_scheduler.StepLR.
            train_step_size (int, optional): Step size for training scheduler. Default is 10.
            train_val_split (List[float], optional): Train-validation split. Default is [0.9, 0.1].
            batch_size (int, optional): Batch size. Default is 2000.
            train_key (str, optional): Key for training data. Default is "train".
            val_key (str, optional): Key for validation data. Default is "val".
            test_key (str, optional): Key for test data. Default is "test".
            predict_key (str, optional): Key for prediction data. Default is "predict".
            logger (Optional["lightning.logger"], optional): Logger. Default is None.
            num_workers (int, optional): Number of workers. Default is 0.
            silent (bool, optional): Whether to silence output. Default is True.
            scale_input_counts (bool, optional): Whether to scale input counts. Default is False.
            reduce_dimensions (bool, optional): Whether to reduce dimensions. Default is False.
            fate_bias_csv_path (Optional[Union[pathlib.Path, str]], optional): Path to fate bias CSV. Default is None.
            fate_bias_multiplier (float, optional): Multiplier for fate bias. Default is 1.
            viz_frequency (int, optional): Frequency of visualization. Default is 1.
            working_dir (Union[pathlib.Path, str], optional): Working directory. Default is os.getcwd().
            time_key (Optional[str], optional): Key for time data. Default is None.
            t0_idx (Optional[pd.Index], optional): Index for initial time. Default is None.
            t_min (float, optional): Minimum time. Default is 0.
            t_max (float, optional): Maximum time. Default is 1.
            dt (float, optional): Time step. Default is 0.1.
            time_cluster_key (Optional[str], optional): Key for time cluster. Default is None.
            t0_cluster (Optional[str], optional): Initial time cluster. Default is None.
            shuffle_time_labels (bool, optional): Whether to shuffle time labels. Default is False.
            mu_hidden (Union[List[int], int], optional): Hidden layers for drift function. Default is [400, 400].
            mu_activation (Union[str, List[str]], optional): Activation function for drift. Default is "LeakyReLU".
            mu_dropout (Union[float, List[float]], optional): Dropout rate for drift. Default is 0.1.
            mu_bias (bool, optional): Whether to use bias in drift. Default is True.
            mu_output_bias (bool, optional): Whether to use output bias in drift. Default is True.
            mu_n_augment (int, optional): Number of augmented dimensions for drift. Default is 0.
            sigma_hidden (Union[List[int], int], optional): Hidden layers for diffusion function. Default is [400, 400].
            sigma_activation (Union[str, List[str]], optional): Activation function for diffusion. Default is "LeakyReLU".
            sigma_dropout (Union[float, List[float]], optional): Dropout rate for diffusion. Default is 0.1.
            sigma_bias (List[bool], optional): Whether to use bias in diffusion. Default is True.
            sigma_output_bias (bool, optional): Whether to use output bias in diffusion. Default is True.
            sigma_n_augment (int, optional): Number of augmented dimensions for diffusion. Default is 0.
            adjoint (bool, optional): Whether to use adjoint method. Default is False.
            sde_type (str, optional): Type of SDE. Default is "ito".
            noise_type (str, optional): Type of noise. Default is "general".
            brownian_dim (int, optional): Dimension of Brownian motion. Default is 1.
            coef_drift (float, optional): Coefficient for drift. Default is 1.0.
            coef_diffusion (float, optional): Coefficient for diffusion. Default is 1.0.
            coef_prior_drift (float, optional): Coefficient for prior drift. Default is 1.0.
            DiffEq_type (str, optional): Type of differential equation. Default is "SDE".
            potential_type (Literal["fixed","prior"]): Type of potential. Default is "fixed".
            encoder_n_hidden (int, optional): Number of hidden layers for encoder. Default is 4.
            encoder_power (float, optional): Power for encoder. Default is 2.
            encoder_activation (Union[str, List[str]], optional): Activation function for encoder. Default is "LeakyReLU".
            encoder_dropout (Union[float, List[float]], optional): Dropout rate for encoder. Default is 0.2.
            encoder_bias (bool, optional): Whether to use bias in encoder. Default is True.
            encoder_output_bias (bool, optional): Whether to use output bias in encoder. Default is True.
            decoder_n_hidden (int, optional): Number of hidden layers for decoder. Default is 4.
            decoder_power (float, optional): Power for decoder. Default is 2.
            decoder_activation (Union[str, List[str]], optional): Activation function for decoder. Default is "LeakyReLU".
            decoder_dropout (Union[float, List[float]], optional): Dropout rate for decoder. Default is 0.2.
            decoder_bias (bool, optional): Whether to use bias in decoder. Default is True.
            decoder_output_bias (bool, optional): Whether to use output bias in decoder. Default is True.
            ckpt_path (Optional[Union[pathlib.Path, str]], optional): Path to checkpoint. Default is None.
            version (str, optional): Version of the model. Default is __version__.

        Returns: None
        """
        self.__config__(locals())

    def fit(
        self,
        train_epochs: int = 2500,
        pretrain_epochs: int = 0,
        train_lr: Optional[float] = None,
        pretrain_callbacks: List = [],
        train_callbacks: List = [],
        ckpt_frequency: int = 25,
        save_last_ckpt: bool = True,
        keep_ckpts: int = -1,
        monitor: Optional[Union[str, List[str]]] = None,
        accelerator: Optional[Union[str, List[str]]] = None,
        log_every_n_steps: int = 1,
        reload_dataloaders_every_n_epochs: int = 1,
        devices: int = 1,
        deterministic: bool = False,
        **kwargs: dict,
    ) -> None:
        """Fit the scDiffEq model to some data.

        Parameters
        ----------
        train_epochs : int, optional
            Number of training epochs, by default 200
        pretrain_epochs : int, optional
            Number of pretrain epochs, by default 500
        train_lr : float, optional
            Learning rate for training, by default None
        pretrain_callbacks : List, optional
            List of pretrain callbacks, by default []
        train_callbacks : List, optional
            List of train callbacks, by default []
        ckpt_frequency : int, optional
            Checkpoint frequency, by default 25
        save_last_ckpt : bool, optional
            Whether to save the last checkpoint, by default True
        keep_ckpts : int, optional
            Number of checkpoints to keep, by default -1
        monitor : optional
            Monitor for early stopping, by default None
        accelerator : optional
            Accelerator to use, by default None
        log_every_n_steps : int, optional
            Log every n steps, by default 1
        reload_dataloaders_every_n_epochs : int, optional
            Reload dataloaders every n epochs, by default 1
        devices : int, optional
            Number of devices to use, by default 1
        deterministic : bool, optional
            Whether to use deterministic algorithms, by default False

        Returns: None
        """
        self.train(**ABCParse.function_kwargs(self.train, locals()))

    def simulate(self) -> anndata.AnnData:
        """Simulate the scDiffEq model.

        Returns: anndata.AnnData
            Simulated data
        """
        ...

    def __repr__(self) -> str:
        """String representation of the scDiffEq model.
        Returns: str
        """
        if hasattr(self, "DiffEq"):
            return f"scDiffEq[{self.DiffEq.__repr__()}]"
        return "scDiffEq"
