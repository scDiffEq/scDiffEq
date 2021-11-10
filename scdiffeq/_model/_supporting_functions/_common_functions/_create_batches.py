import numpy as np


def _create_batches(adata, n_batches):

    """

    Returns

    Creates n_batches + modulo size batch to cover a single epoch of trajectories. very stiff currently.

    """

    n_trajs = adata.obs.trajectory.nunique()
    batched = np.random.choice(
        adata.obs.trajectory.unique(),
        [n_batches, np.floor(n_trajs / n_batches).astype(int)],
        replace=False,
    )

    BatchAssignments = {}
    for batch in range(n_batches):
        BatchAssignments[batch] = batched[batch]

    if n_trajs % n_batches != 0:
        BatchAssignments[batch + 1] = np.setdiff1d(adata.obs.trajectory.unique(), batched.flatten())

    return BatchAssignments