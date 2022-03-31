
__module_name__ = "_sample_lineage_data.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import numpy as np
import torch


# import local dependencies #
# ------------------------- #
from ..._utilities._filter_fragmented_lineages import _filter_fragmented_lineages


def _get_unique_ordered(x, reverse=False):

    """
    Return the sorted, unique values of an array.

    Parameters:
    -----------
    x
        type: numpy.ndarray

    reverse
        optionally reverse the sorted order of the array.
        default: False
        type: bool

    Returns:
    --------
    unique, sorted array

    Notes:
    ------
    """

    if reverse:
        return np.sort(x.unique())[::-1]
    else:
        return np.sort(x.unique())


def _build_lineage_idx(grouped_by_lineage_and_time, t_all, lineage_idx):

    """"""

    LineageIdxDict = {}
    lineage_t_final = grouped_by_lineage_and_time.get_group(
        (lineage_idx, t_all[-1])
    ).index.astype(int)
    n_samples = len(lineage_t_final)

    for t in t_all[:-1]:
        LineageIdxDict[t] = (
            grouped_by_lineage_and_time.get_group((lineage_idx, t))
            .sample(n_samples, replace=True)
            .index.astype(int)
        )

    LineageIdxDict[t_all[-1]] = lineage_t_final

    return LineageIdxDict


def _sample_lineage(
    obs_df, lineage_idx, lineage_index_key="clone_idx", time_key="Time point"
):

    """
    Return a dictionary of lineage timepoints and the corresponding cells at each timepoint.

    adata
        AnnData object
        type: anndata._core.anndata.AnnData

    lineage_idx
        accessor in adata.obs['lineage_index_key']
        type: int

    lineage_index_key
        default: "clone_idx"
        type: str

    time_key
        default: "Time point"
        type: str
    """

    grouped_by_lineage_and_time = obs_df.groupby([lineage_index_key, time_key])
    t_all = _get_unique_ordered(obs_df[time_key])
    LineageIdxDict = _build_lineage_idx(grouped_by_lineage_and_time, t_all, lineage_idx)

    return LineageIdxDict


def _sample_lineage_from_data(X_data, obs_df, lineage_indices):

    """

    adata
        AnnData
        type: anndata._core.anndata.AnnData

    lineage_indices
        array or list of lineage indices
        type: numpy.ndarray or list

    X_key
        default: "X_pca"
        type: str
    """

    combined_sampled_lineages = []
    for lineage_idx in lineage_indices:
        lineage_dict = _sample_lineage(
            obs_df, lineage_idx, lineage_index_key="clone_idx", time_key="Time point"
        )
        lineage_idx_sampling = torch.Tensor(
            X_data[np.vstack(list(lineage_dict.values())).astype(int)]
        )
        combined_sampled_lineages.append(lineage_idx_sampling)

    return combined_sampled_lineages


class Lineage:

    """
    A **"lineage"** is broadly defined as a group of cells for which there exists some prior belief
    or likelihood that they stem from one another. This definition is useful in dealing with multiple
    classes of data - that which is denoted with clonal lineages and/or real time and that which is
    ordered using pseudotemporal ordering algorithms.

    Under this definition, a lineage can have $T$ timepoints can have any number of cells at each
    timepoint as it progress from from $t_{0}$ towards $t_{T}$.
    """

    def __init__(
        self,
        adata,
        use_key="X_pca",
        lineage_index_key="clone_idx",
        time_key="Time point",
    ):

        """

        Parameters:
        -----------
        adata
            AnnData object
            type: anndata._core.anndata.AnnData

        use_key
            default: "X_pca"
            type: str

        lineage_index_key
            default: "clone_idx"
            type: str

        time_key
            default: "Time point"
            type: str

        Returns:
        --------
        None, modifies class in-place.

        Notes:
        ------

        """

        self._lineage_indices = _filter_fragmented_lineages(adata)
        self._adata = adata
        self._obs_df = adata.obs.copy()
        self._X_data = adata.obsm[use_key]
        self._lineage_index_key = lineage_index_key
        self._time_key = time_key

    def sample(self, lineage_idx=False, n_lineages=1, replace_sampling=False):

        """"""

        if not lineage_idx:
            lineage_idx = np.random.choice(
                self._lineage_indices, n_lineages, replace_sampling
            )
        self._lineage_idx = lineage_idx

        self._sampled_lineage_data = _sample_lineage_from_data(
            self._X_data, self._obs_df, self._lineage_idx
        )


def _sample_lineage_data(
    adata,
    use_key="X_pca",
    lineage_index_key="clone_idx",
    time_key="Time point",
    lineage_idx=False,
    n_lineages=1,
    replace_sampling=False,
):

    """

    Parameters:
    -----------
    adata
        AnnData object
        type: anndata._core.anndata.AnnData

    use_key
        default: "X_pca"
        type: str

    lineage_index_key
        default: "clone_idx"
        type: str

    time_key
        default: "Time point"
        type: str

    lineage_idx
        default: False
        type: bool

    n_lineages
        default: 1
        type: int

    replace_sampling
        default: False
        type: bool

    Returns:
    --------
    SampledLineageData
        type: list(torch.Tensors)

    Notes:
    ------

    """

    LineageData = Lineage(adata, use_key, lineage_index_key, time_key)
    LineageData.sample(lineage_idx, n_lineages, replace_sampling)

    return LineageData._sampled_lineage_data