
===========================
Datasets (``sdq.datasets``)
===========================

.. title:: datasets

.. toctree::
    :hidden:

    _datasets/larry
    _datasets/human_hematopoiesis
    _datasets/pancreatic_endocrinogenesis

The datasets used by ``scdiffeq`` are downloaded on first use and cached locally.
They are hosted on Zenodo:

    **DOI:** `10.5281/zenodo.21947161 <https://doi.org/10.5281/zenodo.21947161>`_

Downloads require no authentication, and each file is verified against the md5
checksum published in the record before it is written to the cache.

.. note::

   The datasets were previously served from Figshare. Its download host now sits
   behind a web application firewall that rejects programmatic requests, so
   downloads are served from Zenodo instead. Figshare remains a fallback.


Loaders
=======

.. dropdown:: ``sdq.datasets.larry``

    .. autofunction:: scdiffeq.datasets._larry_in_vitro.larry
        :no-index:
    
    .. button-link:: _datasets/larry.rst
        :color: dark
        :outline:

        More: ``sdq.datasets.larry``

.. dropdown:: ``sdq.datasets.human_hematopoiesis``

    .. autofunction:: scdiffeq.datasets._human_hematopoiesis.human_hematopoiesis
        :no-index:
    
    .. button-link:: _datasets/human_hematopoiesis.rst
        :color: dark
        :outline:

        More: ``sdq.datasets.human_hematopoiesis``

.. dropdown:: ``sdq.datasets.pancreatic_endocrinogenesis``

    .. autofunction:: scdiffeq.datasets._pancreatic_endocrinogenesis.pancreatic_endocrinogenesis
        :no-index:
    
    .. button-link:: _datasets/pancreatic_endocrinogenesis.rst
        :color: dark
        :outline:

        More: ``sdq.datasets.pancreatic_endocrinogenesis``


Citing the data
===============

The Zenodo record redistributes data published by other groups. **Please cite the
original publications**, not only the record:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dataset
     - Publication
   * - LARRY *in vitro*
     - Weinreb C, Rodriguez-Fraticelli A, Camargo FD, Klein AM. Lineage tracing on
       transcriptional landscapes links state to fate during differentiation.
       *Science* (2020).
       `10.1126/science.aaw3381 <https://doi.org/10.1126/science.aaw3381>`_
   * - Human hematopoiesis
     - Qiu X, Zhang Y, Martin-Rufino JD, *et al.* Mapping transcriptomic vector
       fields of single cells. *Cell* (2022).
       `10.1016/j.cell.2021.12.045 <https://doi.org/10.1016/j.cell.2021.12.045>`_
   * - Pancreatic endocrinogenesis
     - Bastidas-Ponce A, Tritschler S, Dony L, *et al.* Comprehensive single cell
       mRNA profiling reveals a detailed roadmap for pancreatic endocrinogenesis.
       *Development* (2019).
       `10.1242/dev.173849 <https://doi.org/10.1242/dev.173849>`_


Where files are cached
======================

Each loader takes a ``data_dir`` argument (default: the current working
directory) and writes beneath ``<data_dir>/scdiffeq_data/``:

.. code-block:: text

    scdiffeq_data/
      larry/
        _larry.raw.h5ad                  # as downloaded
        larry.processed.h5ad             # after preprocessing
        scaler.pkl, pca.pkl              # fitted models
        larry.ct_obs_df.csv              # CytoTRACE annotations
        larry.ct_var_df.csv

The raw download and the preprocessed result are **separate files**. This means a
dataset obtained by any route -- including a manual download -- is still
preprocessed on first use, and a preprocessed file that fails a validity check is
regenerated from the raw file rather than re-downloaded.

.. note::

   Earlier versions stored both under a single name (``larry.h5ad``). Existing
   caches are detected and renamed into the new layout automatically; multi-GB
   files are **not** re-downloaded.

Preprocessing is re-run when the processed file is missing or fails validation.
To force it explicitly:

.. code-block:: python

    import scdiffeq as sdq

    # regenerate the processed file from the cached raw file (no re-download)
    adata = sdq.datasets.larry(force_preprocess=True)

    # re-download the raw file as well
    adata = sdq.datasets.larry(force_download=True)


Reproducibility of the PCA
==========================

Dimension reduction fits a 50-component PCA with ``random_state=0``, so repeated
runs produce an identical basis.

.. warning::

   This makes results reproducible **going forward**. It does *not* reproduce the
   basis distributed with the original publications, which was computed with an
   unseeded randomized SVD and now exists only as a stored array. If you need that
   exact basis, use the ``X_pca`` shipped inside the dataset rather than
   recomputing it.


Working without network access
==============================

Files can be fetched by hand from the `Zenodo record
<https://doi.org/10.5281/zenodo.21947161>`_ and placed in the cache directory
under the names above. The loaders will pick them up and preprocess them normally.
