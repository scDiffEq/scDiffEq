
# -- import packages: --------------------------------------------------------
import pytorch_lightning
import torch_adata
import anndata
import torch

from typing import Union

# -- import local dependencies: ----------------------------------------------
from ..utils import extract_func_kwargs


# -- main class: -------------------------------------------------------------
class LightningDataModuleConfig:
    """Builds self._LightningDataModule"""
    kwargs = {}
    def __init__(
        self,
        data: Union[
            anndata.AnnData,
            torch.utils.data.Dataset,
            pytorch_lightning.LightningDataModule,
        ],
        use_key="X_pca",
        groupby="Time point",
        silent=True,
        **kwargs,
    ):

        """


        Steps:
        ------
        1.  Get the format of the input data.
        2.  If format is AnnData:
            Transform -> torch_adata.AnnDataset

        3.  If format is torch_adata.AnnDataset or torch.utils.data.Dataset:
            Dataset -> pytorch_lightning.LightningDataModule(kwargs)

        4.  If format is pytorch_lightning.LightningDataModule, Done.
        """
        
        self.__parse__(locals())
        
        print(self.kwargs)
        
        self._AnnDataset_kwargs = extract_func_kwargs(
            func=torch_adata.AnnDataset, kwargs=self.kwargs
        )
        self.LightningDataModule_kwargs = extract_func_kwargs(
            func=pytorch_lightning.LightningDataModule, kwargs=self.kwargs
        )

    def __parse__(self, kwargs, ignore=['self']):
        
        for key, val in kwargs.items():
            if not key in ignore:
                self.kwargs[key] = val
                setattr(self, key, val)
            elif key == "kwargs":
                for k, v in val.items():
                    self.kwargs[k] = v
                    setattr(self, k, v)
                    
    def _data_is(self, pyclass):
        return [isinstance(self.data, pyclass), pyclass]    

    # -- Properties: ----------------------------------------------------
    @property
    def is_AnnData(self):
        return self._data_is(anndata.AnnData)

    @property
    def is_torch_Dataset(self):
        return self._data_is(torch.utils.data.Dataset)

    @property
    def is_LightningDataModule(self):
        return self._data_is(pytorch_lightning.LightningDataModule)

    @property
    def data_structures(self):
        return [attr for attr in self.__dir__() if attr.startswith("is_")]

    @property
    def data_format(self):
        for fmt in self.data_structures:
            if getattr(self, fmt)[0]:
                return getattr(self, fmt)[1]

    # -- Data-transform Methods: -----------------------------------------
    def _augment_obs_keys(
        self, weight_key="W", velocity_key="V", fate_key="F", other_keys=[]
    ):
        obs_cols = self.data.obs.columns.tolist()
        self.obs_keys = [weight_key, velocity_key, fate_key] + other_keys

        for key in self.obs_keys:
            if not key in obs_cols:
                self.data.obs[key] = 1
                
    def AnnData_to_AnnDataset(self, silent=True):
        print("Converting AnnData -> AnnDataset (torch Dataset)")
        self._augment_obs_keys()
        
#         print(self._AnnDataset_kwargs)

        return torch_adata.AnnDataset(
            self.data,
            obs_keys=self.obs_keys,
            **self._AnnDataset_kwargs
        )

    def Dataset_to_LightningDataModule(self):
        print("Converting Dataset -> LightningDataModule")
        self._LightningDataModule = LightningAnnDataModule(self.data)

    # -- Controller: ------------------------------------------------------
    def _setup_LightningDataModule(self):

        if self.is_AnnData:
            self.data = self.AnnData_to_AnnDataset()
        if self.is_torch_Dataset:
            self.Dataset_to_LightningDataModule()

    @property
    def LightningDataModule(self):
        if not hasattr(self, "_LightningDataModule"):
            self._setup_LightningDataModule()
        return self._LightningDataModule