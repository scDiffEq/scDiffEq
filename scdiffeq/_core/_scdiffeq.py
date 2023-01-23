
__module_name__ = "_scdiffeq.py"
__doc__ = """API-facing model module."""
__version__ = "0.0.45"
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- import packages: --------------------------------------------------------------------
from pytorch_lightning import LightningDataModule, seed_everything
from neural_diffeqs import NeuralODE, NeuralSDE
from torch.utils.data import DataLoader
from autodevice import AutoDevice
from torch_nets import TorchNet
from annoyance import kNN
import torch_adata
import anndata
import torch
import os


# -- import local dependencies: ----------------------------------------------------------
from . import configs, utils
from .._utilities import Base
from typing import Union
from .lightning_model.forward._test_manager import TestManager
from .._tools import UMAP


# -- API-facing model class: -------------------------------------------------------------
class scDiffEq(Base):
    def __config__(self, kwargs, ignore=["self", "kwargs", "__class__", "hide"]):
        
        kwargs['aux_keys'] = kwargs['velocity_key']
        self.__parse__(kwargs, ignore=ignore)
        self.config = configs.scDiffEqConfiguration(**self._KWARGS)
        for attr in self.config.__dir__():
            if (not attr.startswith("_")) and (not attr.startswith("Prediction")):
                setattr(self, attr, getattr(self.config, attr))
                
    def __preflight__(self):
        
        self.UMAP = UMAP(use_key=self.use_key)
        if 'X_umap' in self.adata.obsm_keys():
            self.adata.obsm['X_umap_sdq'] = self.UMAP(self.adata)
        print("Building kNN Graph...")
        self.kNN_Graph = kNN(self.adata)
        self.kNN_Graph.build()
        self.kNN_idx = self.kNN_Graph.idx
        
        
    def __init__(
        self,
        adata: anndata.AnnData = None,
        func: Union[NeuralODE, NeuralSDE, torch.nn.Module] = None,
        seed=0,
        use_key="X_pca",
        time_key="Time point",
        obs_keys=['W'],
        train_key="train",
        val_key="val",
        test_key="test",
        predict_key="predict",
        accelerator="gpu",
        velocity_key=None,
        V_coefficient = 0,
        stdev: Union[torch.nn.Parameter, torch.Tensor, float] = torch.Tensor([0.]),
        # torch.nn.Parameter(torch.tensor(0.5, requires_grad=True, device=AutoDevice())),
        V_scaling: Union[torch.nn.Parameter, torch.Tensor, float] = torch.Tensor([0.]),
        # torch.nn.Parameter(torch.tensor(1.0, requires_grad=True, device=AutoDevice()))
        tau: Union[torch.nn.Parameter, torch.Tensor, float]= torch.Tensor([0.]),
        t0_idx=None,
        n_steps=40,
        adjoint=False,
        devices=1,
        batch_size=2000,
        num_workers=os.cpu_count(),
        n_groups=None,
        train_val_percentages=[0.8, 0.2],
        remainder_idx=-1,
        predict_all=True,
        attr_names={"obs": [], "aux": []},
        one_hot=False,
        aux_keys=None,
        silent=True,
        optimizer="RMSprop",
        lr_scheduler="StepLR",
        lr=0.0001,
        step_size=20,
        dt=0.1,
        fate_scale=0,
        model_save_dir: str = "scDiffEq_model",
        log_name: str = "lightning_logs",
        version=None,
        prefix="",
        flush_logs_every_n_steps=5,
        max_epochs=1500,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=5,
        ckpt_outputs_frequency=50,
        t=None,
        N=False,
        disable_velocity=False,
        disable_potential=False,
        disable_fate_bias=False,
        skip_positional_backprop=False,
        skip_positional_velocity_backprop=False,
        skip_potential_backprop=False,
        skip_fate_bias_backprop=False,
        do_preflight=True,
        **kwargs
        # TODO: ENCODER/DECODER KWARGS
    ):
        
        """
        Primary user-facing model.
        
        Parameters:
        -----------
        adata
            type: anndata.AnnData
            default: None


        DataModule
            For data already processed and formatted into a LightningDataModule from pytorch-lightning.
            Users familiar with pytorch-lightning may wish to preformat their data in this way. If used,
            steps to generate a LightningDataModule from AnnData are bypassed.
            type: pytorch_lightning.LightningDataModule
            default: None            

        func
            type: [NeuralODE, NeuralSDE, torch.nn.Module]
            default: None
        
        Keyword Arguments:
        ------------------


        Returns:
        --------
        None
        
        Notes:
        ------

        """
        
        seed = seed_everything(seed)
        
        self.__config__(locals())
        if do_preflight:
            self.__preflight__()
        
    def __repr__(self):
        # TODO: add a nice self-report method to be returned as a str
        return "scDiffEq model"

    def fit(self):
        self.LightningTrainer.fit(self.LightningModel, self.LightningDataModule)
        
    def test(self,
             n_predictions = 25,
             test_adata = None,
             use_key = "X_pca",
             groupby = "Time point",
             batch_size = 10_000,
             shuffle = True,
             savename="test_predictions.npy",
            ):
        
        self._TestManager = TestManager(diffeq=self,
                    n_predictions = n_predictions,
                    test_adata = test_adata,
                    use_key = use_key,
                    groupby = groupby,
                    batch_size = batch_size,
                    shuffle = shuffle,
                   )
        
        self.test_predictions = self._TestManager(savename=savename)

    @property
    def log_dir(self):
        return os.path.join(
            self.KWARGS["TRAINER"]["model_save_dir"],
            self.KWARGS["TRAINER"]["log_name"],
        )

    def load(self, ckpt_path):
        
        self.IncompatibleKeys = self.LightingModel.load_state_dict(torch.load(ckpt_path)["state_dict"])
        if not any([self.IncompatibleKeys.missing_keys, self.IncompatibleKeys.unexpected_keys]):
            print("Successfully Loaded scDiffEq Model")
        else:
            print("Problem loading scDiffEq Model")
            