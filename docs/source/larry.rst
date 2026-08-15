=============
LARRY Dataset
=============

.. title:: LARRY

Lineage tracing data from Weinreb *et al.* (`Science
<https://doi.org/10.1126/science.aaw3381>`_, 2020), in which barcoded
hematopoietic progenitors are profiled across three timepoints, allowing observed
clonal fate to be compared against predicted fate.

.. code-block:: python

    import scdiffeq as sdq

    adata = sdq.datasets.larry()

See :doc:`data` for where files are cached and what to cite.


Variants
========

Two variants are available, and the difference between them matters:

.. list-table::
   :header-rows: 1
   :widths: 22 16 14 48

   * - ``variant``
     - Shape
     - ``X_pca``
     - Description
   * - ``None`` (default)
     - 130,887 × 2,492
     - **included**
     - The biology-rich object. Already gene-filtered, and ships precomputed
       ``X_pca``, ``X_umap`` and ``X_scaled``.
   * - ``"unprocessed"``
     - 130,887 × 25,289
     - **not included**
     - The upstream input, before gene filtering. Carries a ``var["use_genes"]``
       column that preprocessing filters by, reducing it to 2,447 genes. ``X_pca``
       is computed locally.

.. code-block:: python

    # default: ships its own X_pca
    adata = sdq.datasets.larry()

    # the unprocessed input; X_pca is computed during preprocessing
    adata = sdq.datasets.larry(variant="unprocessed")

.. deprecated:: 1.1.1
   ``variant="fate_prediction"`` is deprecated in favour of
   ``variant="unprocessed"``. The old name still works and emits a
   ``DeprecationWarning``.

   The rename corrects a misleading name. That variant's upstream filename is
   ``adata.Weinreb2020.in_vitro.gene_filtered.h5ad``, but the object it contains is
   the *unfiltered* one -- the ``gene_filtered`` name refers to the presence of the
   filtering metadata, not to filtering having been applied. Reading it as
   "already gene-filtered" led to the reasonable but incorrect expectation that it
   would carry a precomputed ``X_pca``.


Preprocessing
=============

With default arguments, :func:`~scdiffeq.datasets.larry` annotates the object with
precomputed CytoTRACE values, filters genes on ``var["use_genes"]``, then scales
and runs a 50-component PCA:

.. code-block:: python

    adata = sdq.datasets.larry(
        filter_genes=True,       # subset to var["use_genes"]
        reduce_dimensions=True,  # StandardScaler + PCA(50, random_state=0)
        cytotrace=True,          # merge precomputed CytoTRACE annotations
    )

Each flag is independent; ``cytotrace=True`` is honoured even when the other two
are disabled. Non-default combinations are cached separately, so
``reduce_dimensions=False`` does not overwrite the default cache.

The CytoTRACE annotations are keyed to the gene-filtered gene set, so on the
``"unprocessed"`` variant they cover 2,492 of its 25,289 genes; the remainder are
``NaN``. This is expected, not an error.


API
===

.. autofunction:: scdiffeq.datasets._larry_in_vitro.larry
