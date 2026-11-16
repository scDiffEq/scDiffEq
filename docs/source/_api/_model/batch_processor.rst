==================
``BatchProcessor``
==================

.. title:: BatchProcessor

.. automodule:: scdiffeq.core.lightning_models.base._batch_processor.BatchProcessor
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Observation routing and migration
---------------------------------

Batches produced by scDiffEq have the layout
``[time, X, weights, *auxiliary_observations]``. The configured ``weight_key``
is always first among observations, regardless of its position in ``obs_keys``.
Custom consumers must account for this ordering. The remaining observations
retain their requested order and are ignored by ordinary training losses.

Fate indices require explicit opt-in::

    batch = BatchProcessor(tensors, batch_idx, fate_bias=True)
    initial_fate_ids = batch.F_idx

Direct callers previously relying on automatic fourth-tensor detection must
add ``fate_bias=True``. This also applies to
``model.DiffEq.process_batch(tensors, batch_idx, fate_bias=True)``.
Without the flag, ``F_idx`` returns ``None``. Built-in fate mixin consumers
already pass the flag.

Fate-aware batches must contain exactly ``[time, X, weights, fate_idx]``.
Supply exactly one auxiliary numeric integer ID column in ``obs_keys``;
including ``weight_key`` is optional. IDs must have shape
``[batch_size, n_times, 1]`` and match the fate table's index after conversion
to integer strings. Only the initial-time IDs are used, as before. Extra
observations are now rejected, as are missing, non-finite, fractional, and
multi-column encoded IDs. A single numeric column is explicitly interpreted
as IDs by fate-aware consumers; this contract cannot infer its meaning from
its values, so callers must select the correct column.

Categorical integration remains pending `torch-adata #20
<https://github.com/mvinyard/torch-adata/issues/20>`_. The numeric routing fix
can proceed independently and does not update the dependency requirement.
``obs_keys`` alone does not configure supervision, condition dynamics, or
correct batch effects; annotations remain in ``adata.obs`` when it is omitted.
