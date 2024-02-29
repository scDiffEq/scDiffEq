=============
API Reference
=============

.. title:: API

Here you will find documentation of the scDiffEq API. This covers both the model as well as the supporting function modules.

Model
=====

.. grid:: 2
   :gutter: 1

   .. grid-item::
      :columns: 1

      .. button-link:: _api/model.rst
         :color: primary
   
         Go

   .. grid-item::
      :columns: 11

      .. dropdown:: Model (``sdq.scDiffEq``) modules
         :open:

         .. toctree::
            :maxdepth: 2
            :hidden:

            _api/model

         .. button-link:: _api/_model/model.rst
            :color: primary
            :outline:
            :expand:

            sdq.scDiffEq

         .. button-link:: _api/_model/lightning_models.rst
            :color: primary
            :outline:
            :expand:

            sdq.core.lightning_models


Data I/O
========

.. grid:: 2
   :gutter: 1

   .. grid-item::
      :columns: 1

      .. button-link:: _api/io.rst
         :color: primary
   
         Go

   .. grid-item::
      :columns: 11

      .. dropdown:: Data I/O (``sdq.io``) modules

         .. toctree::
            :maxdepth: 2
            :hidden:

            _api/io

         .. button-link:: _api/_io/read_h5ad.rst
            :color: primary
            :outline:
            :expand:

            sdq.io.read_h5ad

         .. button-link:: _api/_io/read_pickle.rst
            :color: primary
            :outline:
            :expand:

            sdq.io.read_pickle

         .. button-link:: _api/_io/write_pickle.rst
            :color: primary
            :outline:
            :expand:

            sdq.io.write_pickle


Datasets
========

.. grid:: 2
   :gutter: 1

   .. grid-item::
      :columns: 1

      .. button-link:: _api/datasets.rst
         :color: primary
   
         Go

   .. grid-item::
      :columns: 11

      .. dropdown:: Datasets (``sdq.datasets``) modules

         .. toctree::
            :maxdepth: 2
            :hidden:

            _api/datasets

         .. button-link:: _api/_datasets/pancreas.rst
            :color: primary
            :outline:
            :expand:

            sdq.datasets.pancreas
            
         .. button-link:: _api/_datasets/human_hematopoiesis.rst
            :color: primary
            :outline:
            :expand:

            sdq.datasets.human_hematopoiesis
            
Plotting
========

.. grid:: 2
   :gutter: 1

   .. grid-item::
      :columns: 1

      .. button-link:: _api/plotting.rst
         :color: primary
   
         Go

   .. grid-item::
      :columns: 11

      .. dropdown:: Plotting (``sdq.pl``) modules

         .. toctree::
            :maxdepth: 2
            :hidden:

            _api/datasets

         .. button-link:: _api/_plotting/velocity_stream.rst
            :color: primary
            :outline:
            :expand:

            sdq.pl.velocity_stream


Tools
=====

.. grid:: 2
   :gutter: 1

   .. grid-item::
      :columns: 1

      .. button-link:: _api/tools.rst
         :color: primary
   
         Go

   .. grid-item::
      :columns: 11

      .. dropdown:: Tools (``sdq.tl``) modules

         .. toctree::
            :maxdepth: 2
            :hidden:

            _api/tools

         .. button-link:: _api/_tools/annotate_cell_state.rst
            :color: primary
            :outline:
            :expand:

            sdq.tl.annotate_cell_state
            
         .. button-link:: _api/_tools/annotate_cell_fate.rst
            :color: primary
            :outline:
            :expand:

            sdq.tl.annotate_cell_fate

         .. button-link:: _api/_tools/knn.rst
            :color: primary
            :outline:
            :expand:

            sdq.tl.kNN

         .. button-link:: _api/_tools/simulate.rst
            :color: primary
            :outline:
            :expand:

            sdq.tl.simulate

         .. button-link:: _api/_tools/perturb.rst
            :color: primary
            :outline:
            :expand:

            sdq.tl.perturb

