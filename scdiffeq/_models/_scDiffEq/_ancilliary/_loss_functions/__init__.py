

__module_name__ = "__init__.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


from ._KL_Divergence import _KL_Divergence as KL_Divergence
from ._OptimalTransportLoss import _OptimalTransportLoss as OTLoss
from ._Loss import _calculate_loss as calculate_loss