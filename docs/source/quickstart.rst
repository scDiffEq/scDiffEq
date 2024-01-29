===========
Quick Start
===========

Install the library
-------------------

.. code-block:: shell

   pip install scdiffeq


Import the library
------------------

.. code-block:: python

   import scdiffeq as sdq


Get some sample data
--------------------

Let's say you have some temporally-resolved data, spanning three time points. There are 200 12-dimension samples at each time point.

.. code-block:: python

   adata = sdq.io.mouse_hematopoiesis()

Define and fit the model
------------------------

.. code-block:: python
   model = sdq.scDiffEq(
      adata=adata,
      latent_dim=50,
      use_key="X_pca", # 50 dim
      time_key="time",
      mu_hidden=[512,512],
      sigma_hidden=[32,32],
      potential_type="fixed",
   )

   model.fit(train_epochs = 500)


Run some simulations
--------------------

.. code-block:: python

   adata_sim = sdq.tl.simulate(adata, idx=idx, N=200)

   sdq.tl.drift(adata_sim, model)
   sdq.tl.diffusion(adata_sim, model)