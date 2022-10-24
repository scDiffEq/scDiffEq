
from torch.linalg import vector_norm
import torch

def autodevice():
    if torch.cuda.is_available():
        return torch.device("cuda:{}".format(torch.cuda.current_device()))
    return torch.device("cpu")

def local_arg_parser(kwargs, ignore=["self"]):

    parsed_kwargs = {}
    for k, v in kwargs.items():
        if not k in ignore:
            parsed_kwargs[k] = v

    return parsed_kwargs


class VectorNormDriftDiffusion:
    def __init__(self, adata, use_key="X_pca", subset=None, device=autodevice()):

        self.__parse__(locals())

    def __parse__(self, kwargs, ignore=["self"]):
        for k, v in kwargs.items():
            if not k in ignore:
                setattr(self, k, v)

        if self.subset:
            self.adata = self.adata[self.adata.obs[self.subset]].copy()

    def forward(self, func, X0_predict):
        return (
            vector_norm((func(X0_predict) - X0_predict), dim=1).detach().cpu().numpy()
        )

    def predict(self, model):

        self.X0_predict = torch.Tensor(self.adata.obsm[self.use_key]).to(self.device)
        self.f = model.func.mu.to(self.device)
        self.g = model.func.sigma.to(self.device)

        self.adata.obs["L1_drift"] = self.forward(self.f, self.X0_predict)
        self.adata.obs["L1_diffu"] = self.forward(self.g, self.X0_predict)
        self.adata.obs["L1_velo"] = self.forward(self.model, self.X0_predict)
        
        
def decomposed_velocity(
    adata, model, use_key="X_pca", subset=None, device=autodevice()
):
    """
    Pass each observed cell in adata through f, g, and the composite model.

    Parameters:
    -----------
    adata

    model

    use_key

    subset

    device


    Returns:
    --------
    None or adata

    Notes:
    ------
    (1)
    """

    VecNorm = VectorNormDriftDiffusion(adata, use_key, subset, device)
    VecNorm.predict()

    if subset:
        return VecNorm.adata