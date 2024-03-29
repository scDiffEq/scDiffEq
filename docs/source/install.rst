============
Installation
============


To begin using ``scdiffeq``, recommend install from PYPI ().  GitHub.


Install via pip (recommended)
"""""""""""""""""""""""""""""
.. code-block:: python

    pip install scdiffeq


GitHub (Developer version)
""""""""""""""""""""""""""

To access the latest version of ``scdiffeq`` from GitHub, clone the 
repository, ``cd`` into the project's root directory and install the
editable version.

.. code-block:: python

    # Install the developer version via GitHub
    
    git clone https://github.com/scDiffEq/scDiffEq.git; cd ./scDiffEq;
    pip install -e .


Troubleshooting
"""""""""""""""

The ``pykeops`` library creates a cache. Sometimes, when you switch devices
though retain the same disc (common when using a VM, for example), this cache
will no longer be compatible with the installed drivers for that device. To
clear and rewrite this cached, we can perform the following:

.. code-block:: python

    import pykeops

    pykeops.clean_pykeops()
