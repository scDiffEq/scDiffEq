============
Installation
============


To begin using ``scdiffeq``, we recommend installing from PyPI for the 
stable release, or from GitHub for the latest developer version.


Install via pip (recommended for stable release)
""""""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: bash

    pip install scdiffeq


GitHub (Developer version)
""""""""""""""""""""""""""

To access the latest version of ``scdiffeq`` from GitHub, clone the 
repository and install the editable version. Installation generally only 
takes a few seconds.

Using uv (recommended)
"""""""""""""""""""""""

.. code-block:: bash

    git clone https://github.com/scDiffEq/scDiffEq.git; cd ./scDiffEq;
    
    # Install uv if you haven't already: curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync


Using pip
"""""""""

.. code-block:: bash

    git clone https://github.com/scDiffEq/scDiffEq.git; cd ./scDiffEq;
    pip install -e .


With documentation dependencies
"""""""""""""""""""""""""""""""

If you want to build the documentation locally:

.. code-block:: bash

    # Using uv
    uv sync --extra docs

    # Using pip
    pip install -e ".[docs]"


Optional dependency groups
""""""""""""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Extra
     - Contents
   * - ``docs``
     - Sphinx and theme packages needed to build this documentation.
   * - ``test``
     - ``pytest``, for running the test suite.
   * - ``dev``
     - Jupyter, ``ipykernel``, and ``pytest`` for interactive development.

.. code-block:: bash

    # run the test suite
    uv sync --extra test
    uv run pytest

Network-dependent tests are marked ``slow`` and excluded by default; run them
with ``pytest -m slow``. Note that they download multi-gigabyte datasets.


Troubleshooting
"""""""""""""""

The ``pykeops`` library creates a cache. Sometimes, when you switch devices
though retain the same disc (common when using a VM, for example), this cache
will no longer be compatible with the installed drivers for that device. To
clear and rewrite this cached, we can perform the following:

.. code-block:: python

    import pykeops

    pykeops.clean_pykeops()
