import numpy as np


def _create_batches(adata, n_batches):

    """

    Returns

    Creates n_batches + modulo size batch to cover a single epoch of trajectories. very stiff currently.

    """

    n_trajs = adata.obs.trajectory.nunique()
    if n_trajs % n_batches != 0:
        n_batches_adj = n_batches + 1
    else:
        n_batches_adj = n_batches
    # to fit in all trajectories (last batch is modulo of the divisor)

    batched = np.random.choice(
        np.arange(n_trajs),
        [n_batches_adj, np.floor(n_trajs / n_batches_adj).astype(int)],
        replace=False,
    )

    BatchAssignments = {}
    for batch in range(n_batches):
        BatchAssignments[batch] = batched[batch]

    if n_trajs % n_batches != 0:
        if n_trajs % n_batches_adj != 0:
            BatchAssignments[batch + 1] = np.setdiff1d(
                np.arange(n_trajs), batched.flatten()
            )

    return BatchAssignments