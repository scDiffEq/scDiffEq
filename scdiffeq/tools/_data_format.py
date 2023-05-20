import numpy as np
import autodevice
import torch
import anndata

from typing import Union

from ..core import utils

class DataFormat(utils.ABCParse):
    def __init__(self, data: Union[torch.Tensor, np.ndarray]):
        self.__parse__(locals(), public=["data"])

    @property
    def device_type(self):
        if hasattr(self.data, "device"):
            return self.data.device.type
        return "cpu"
    
    @property
    def is_ArrayView(self):
        return isinstance(self.data, anndata._core.views.ArrayView)

    @property
    def is_numpy_array(self):
        return isinstance(self.data, np.ndarray)

    @property
    def is_torch_Tensor(self):
        return isinstance(self.data, torch.Tensor)

    @property
    def on_cpu(self):
        return self.device_type == "cpu"

    @property
    def on_gpu(self):
        return self.device_type in ["cuda", "mps"]

    def to_numpy(self):
        if self.is_torch_Tensor:
            if self.on_gpu:
                return self.data.detach().cpu().numpy()
            return self.data.numpy()
        elif self.is_ArrayView:
            return self.data.toarray()
        return self.data

    def to_torch(self, device=autodevice.AutoDevice()):
        self.__update__(locals())
        if self.is_torch_Tensor:
            return self.data.to(device)
        elif self.is_ArrayView:
            self.data = self.data.toarray()
        return torch.Tensor(self.data).to(device)