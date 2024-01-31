
import numpy as np


def get_iterative_indices(
    indices,
    index,
    n_recursive_neighbors: int = 2,
    max_neighs=None,
):
    """Iterative index searching"""

    def iterate_indices(indices, index, n_recursive_neighbors):
        if n_recursive_neighbors > 1:
            index = iterate_indices(indices, index, n_recursive_neighbors - 1)
        ix = np.append(index, indices[index])  # direct and indirect neighbors
        if np.isnan(ix).any():
            ix = ix[~np.isnan(ix)]
        return ix.astype(int)

    indices = np.unique(iterate_indices(indices, index, n_recursive_neighbors))
    if max_neighs is not None and len(indices) > max_neighs:
        indices = np.random.choice(indices, max_neighs, replace=False)
    return indices
