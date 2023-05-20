
# -- import packages: -------------
from autodevice import AutoDevice
import torch_adata

def fetch(adata, use_key: str, device=AutoDevice()):
    return torch_adata.tl.fetch(adata, use_key).to(device)