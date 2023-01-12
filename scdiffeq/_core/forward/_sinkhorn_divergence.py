
__module_name__ = "_sinkhorn_divergence.py"
__doc__ = """To-do."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- import packages: --------------------------------------------------------------------
from geomloss import SamplesLoss
from autodevice import AutoDevice

from ..utils import Base

# -- API-facing class: -------------------------------------------------------------------
class SinkhornDivergence(SamplesLoss, Base):
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
