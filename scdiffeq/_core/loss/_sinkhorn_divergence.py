
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


# -- import local dependencies: ----------------------------------------------------------
from ..utils import autodevice


# -- API-facing class: -------------------------------------------------------------------
class SinkhornDivergence(SamplesLoss):
    def __init__(
        self,
        device=autodevice(),
        loss="sinkhorn",
        backend="auto",  # online
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

    def __parse__(self, kwargs, ignore=["self"]):
        for k, v in kwargs.items():
            if not k in ignore:
                setattr(self, k, v)
