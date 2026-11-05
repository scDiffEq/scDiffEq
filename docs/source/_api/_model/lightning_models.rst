
================================================
Lightning models (``sdq.core.lightning_models``)
================================================

This sub-package contains the registered Lightning models selected by
``scDiffEq`` and available as ``model.DiffEq``.

Selecting a model
=================

``sdq.scDiffEq`` selects a model from its constructor arguments and the width of
the input representation. By default, that representation is
``adata.obsm["X_pca"]``.

.. list-table:: Model selection settings
    :header-rows: 1
    :widths: 30 70

    * - Setting
      - Effect
    * - ``DiffEq_type``
      - ``"SDE"`` (default) or ``"ODE"``.
    * - ``latent_dim``
      - A VAE is selected when input width exceeds this value (default 50).
    * - ``potential_type``
      - ``"fixed"`` (default), ``"prior"``, or ``None``.
    * - ``velocity_ratio_params``
      - A dictionary selects a regularized SDE. ``None`` disables the regularizer.
    * - ``fate_bias_csv_path``
      - A path selects a fate-bias-aware variant when that combination exists.

Not every combination is implemented. Unsupported combinations raise a
``ValueError`` describing the selected configuration and registered alternatives.
The configuration is not silently changed to another model.

With default settings, 50-dimensional input selects
``LightningSDE_FixedPotential_RegularizedVelocityRatio``. Wider input selects
``LightningSDE_VAE_FixedPotential_RegularizedVelocityRatio``: an encoder maps the
input to 50 dimensions, the SDE evolves in latent space, and a decoder maps the
trajectory back to input space. Distribution matching uses the decoded
trajectory; velocity-ratio regularization uses latent drift and diffusion.

To model dynamics directly in an existing PCA representation, match the state
dimension to its width:

.. code-block:: python

    model = sdq.scDiffEq(
        adata,
        use_key="X_pca",
        latent_dim=adata.obsm["X_pca"].shape[1],
    )

To learn a smaller latent representation, set ``latent_dim`` below the input
width. The fixed-potential VAE supports velocity-ratio regularization:

.. code-block:: python

    model = sdq.scDiffEq(adata, use_key="X_pca", latent_dim=50)
    model.fit(pretrain_epochs=100, train_epochs=1500)

``pretrain_epochs`` in ``fit`` defaults to 0. A positive value first optimizes
encoder/decoder reconstruction, then ``train_epochs`` optimizes dynamics.
Choose epoch counts and latent dimensionality for the dataset.

Passing ``velocity_ratio_params=None`` selects an unregularized model.
Setting ``"enforce": 0`` inside a regularization dictionary retains the
regularized model class with zero penalty weight; it does not change which
class is selected.

Discovering available models
================================

List the classes registered in the installed version:

.. code-block:: python

    from scdiffeq.core import lightning_models

    available = sorted(
        name for name in dir(lightning_models) if name.startswith("Lightning")
    )
    print(available)

After construction, inspect the chosen class:

.. code-block:: python

    print(type(model.DiffEq).__name__)

Registered classes can also be imported from ``scdiffeq.core.lightning_models``
for direct Lightning use. The ``sdq.scDiffEq`` constructor configures the data,
training stages, and callbacks for the usual workflow.

.. title:: Lightning

.. toctree::
    :hidden:

    lightning_sde
    lightning_ode
    lightning_sde_fixed_potential
    lightning_ode_fixed_potential
    lightning_sde_fixed_potential_regularized_velocity_ratio
    lightning_ode_fixed_potential_regularized_velocity_ratio
    base_lightning_diffeq
    batch_processor
    
.. card:: ``sdq.core.lightning_models.lightning_sde``
    :link: lightning_sde.rst
    :text-align: right

    :octicon:`code;2em;sd-text-dark`

.. card:: ``sdq.core.lightning_models.lightning_ode``
    :link: lightning_ode.rst
    :text-align: right

    :octicon:`code;2em;sd-text-dark`

.. card:: ``sdq.core.lightning_models.LightningSDE_FixedPotential``
    :link: lightning_sde_fixed_potential.rst
    :text-align: right

    :octicon:`code;2em;sd-text-dark`

.. card:: ``sdq.core.lightning_models.LightningODE_FixedPotential``
    :link: lightning_ode_fixed_potential.rst
    :text-align: right

    :octicon:`code;2em;sd-text-dark`
    
.. card:: ``sdq.core.lightning_models.LightningSDE_FixedPotential_RegularizedVelocityRatio``
    :link: lightning_sde_fixed_potential_regularized_velocity_ratio.rst
    :text-align: right

    :octicon:`code;2em;sd-text-dark`

.. card:: ``sdq.core.lightning_models.LightningODE_FixedPotential_RegularizedVelocityRatio``
    :link: lightning_ode_fixed_potential_regularized_velocity_ratio.rst
    :text-align: right

    :octicon:`code;2em;sd-text-dark`
    
.. card:: ``sdq.core.lightning_models.base.BaseLightningDiffEq``
    :link: base_lightning_diffeq.rst
    :text-align: right

    :octicon:`code;2em;sd-text-dark`

.. card:: ``sdq.core.lightning_models.base.batch_processor``
    :link: batch_processor.rst
    :text-align: right

    :octicon:`code;2em;sd-text-dark`
