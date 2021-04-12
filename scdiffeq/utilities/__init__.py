
# utilities __init__.py

__author__ = ', '.join([
    'Michael E. Vinyard'
])
__email__ = ', '.join([
    'vinyard@g.harvard.edu',
])

from .general_utility_functions import ensure_array

from .torch_device import set_device
from .torch_device import torch_device

from .subsetting_functions import subset_adata
from .subsetting_functions import randomly_subset_trajectories
from .subsetting_functions import isolate_trajectory

from .split_test_train import split_test_train

from .downsample_adata import downsample_adata

from .general_utility_functions import load_development_libraries as devlibs
from .general_utility_functions import use_embedding

from .save_adata import write_h5ad