
__module_name__ = "__init__.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


from ._configure_optimization import _configure_optimization as optim_config
from ._SinkhornDivergence import SinkhornDivergence
from ._formatting import _format_batched_inputs as format_batched_inputs
from ._accounting import _retain_gradients_for_potential as retain_gradients
from ._accounting import _count_params as count_params
from ._accounting import _update_loss_logs as update_loss_logs
from ._fate_bias_transform import _fate_bias_transform as fate_bias