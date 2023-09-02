
# -- import packages: ----------------------------------------------------------
from geomloss import SamplesLoss
from autodevice import AutoDevice
import ABCParse

# -- import local dependencies: ------------------------------------------------



# -- API-facing class: ---------------------------------------------------------
class SinkhornDivergence(SamplesLoss, ABCParse.ABCParse):
    def __init__(
        self,
        device=AutoDevice(),
        loss="sinkhorn",
        backend="online", # "auto",
#         diameter=,
        p=2,
        blur=0.1,
        scaling=0.7,
        debias=True,
        sample_axis=1,
        **kwargs
    ):
        super(SinkhornDivergence, self).__init__()

        self.__parse__(locals())
