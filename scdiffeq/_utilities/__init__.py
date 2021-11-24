# utilities __init__.py

__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])

from ._torch_device import _set_device as set_device
from ._torch_device import _torch_device as torch_device

from ._subsetting_functions import _group_adata_subset as group_adata_subset
from ._subsetting_functions import (
    _randomly_subset_trajectories as randomly_subset_trajectories,
)
from ._subsetting_functions import _isolate_trajectory as isolate_trajectory
from ._subsetting_functions import _check_df as check_df
from ._subsetting_functions import _get_subset_idx as get_subset_idx
from ._subsetting_functions import _subset_df as subset_df
from ._subsetting_functions import _subset_adata as subset_adata

from ._general_utility_functions import _load_development_libraries as devlibs
from ._general_utility_functions import _use_embedding

from ._save_adata import _write_h5ad as write_h5ad

from ._load_model import _load_model as load_model

from ._read_csv_to_anndata import _read_csv_to_anndata as read_csv_to_adata

from ._kmeans import _get_kmeans_inertia as get_kmeans_inertia
from ._kmeans import _kmeans as kmeans

from ._add_noise import _add_noise as add_noise

from ._preprocess import preprocess

# AnnData handling functions
from ._AnnData_handlers._read_write._write_AnnData import _write_AnnData as write_adata
from ._AnnData_handlers._read_write._read_AnnData import _read_AnnData as read_adata
from ._AnnData_handlers._split_AnnData_test_train_validation import (
    _split_test_train as split_test_train,
)

from ._flexible_multilevel_mkdir import _flexible_multilevel_mkdir as mkdir_flex


# deprecated:
# -----------
# from ._general_utility_functions import _ensure_array
# from ._split_test_train import _split_test_train as split_test_train
# from ._downsample_adata import _downsample_adata as downsample_adata
