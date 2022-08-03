
__module_name__ = "_WassersteinDistance.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
from geomloss import SamplesLoss
import torch


class WassersteinDistance:

    """
    vector_a
        type: torch.Tensor
    vector_b
        type: torch.Tensor
    weight_a
        default: None
    weight_b
        default: None
    requires_grad
        default: True
    Notes:
    ------
    Example Implementation:
        a, b = torch.randint(0, 50, (10,2)), torch.randint(0, 50, (10,2))
        w = torch.ones(a.shape[0])
        w = w / w.sum()
        LF(a, b, weight_a=w, weight_b=w)
        LF(a, b)
    """

    def __init__(
        self,
        loss="sinkhorn",
        backed="online",
        p=2,
        blur=0.1,
        scaling=0.7,
        debias=True,
        gpu=0,
    ):

        self.device = torch.device(
            "cuda:" + str(gpu) if torch.cuda.is_available() else "cpu"
        )
        self._OT_solver = SamplesLoss(
            loss=loss, backend=backed, p=p, blur=blur, scaling=scaling, debias=True
        )

    def __call__(
        self, vector_a, vector_b, weight_a=None, weight_b=None, requires_grad=True
    ):

        self.vector_a = vector_a.float().to(self.device)
        self.vector_b = vector_b.float().to(self.device)

        if requires_grad:
            vector_a.requires_grad_()

        if weight_a != None and weight_b != None:
            self.weight_a = weight_a.float().to(self.device)
            self.weight_b = weight_b.float().to(self.device)

            if requires_grad:
                self.weight_a.requires_grad_()
                self.weight_b.requires_grad_()

            loss = self._OT_solver(
                self.weight_a, self.vector_a, self.weight_b, self.vector_b
            )
            return loss
        else:
            loss = self._OT_solver(self.vector_a, self.vector_b)
            return loss

    def compute(self, x_hat, x_obs, t):
        return torch.stack(
            [self.__call__(x_hat[i], x_obs[i]) for i in range(1, len(t))]
        )