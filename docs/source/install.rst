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


Troubleshooting
"""""""""""""""

The ``pykeops`` library creates a cache. Sometimes, when you switch devices
though retain the same disc (common when using a VM, for example), this cache
will no longer be compatible with the installed drivers for that device. To
clear and rewrite this cached, we can perform the following:

.. code-block:: python

    import pykeops

    pykeops.clean_pykeops()
