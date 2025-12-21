
# -- import packages: ----------------------------------------------------------
from geomloss import SamplesLoss
import autodevice
import ABCParse


# -- API-facing class: ---------------------------------------------------------
class SinkhornDivergence(SamplesLoss, ABCParse.ABCParse):
    def __init__(
        self,
        device=autodevice.AutoDevice(),
        loss="sinkhorn",
        backend="online",
        p=2,
        blur=0.1,
        scaling=0.7,
        debias=True,
        sample_axis=1,
#         reach=None,
#         diameter=None,
#         truncate=5,
#         cost=None,
#         kernel=None,
#         cluster_scale=None,
#         potentials=False,
#         verbose=False,
        **kwargs
    ) -> None:
        super(SinkhornDivergence, self).__init__()

        self.__parse__(locals())
