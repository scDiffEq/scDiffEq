============
Dependencies
============

.. title:: Dependencies


To develop ``scdiffeq``, we built several supporting libraries. Many of these
libraries contain functions that are not necessarily directly relevant to 
``scdiffeq`` and we found that abstracting them into their own package was not
only useful for the development of other projects, but helped to clean up ``scdiffeq``.

That said, developers interested in working with and building upon 
``scdiffeq`` may find utility in documentation of these supporting libraries.
Several of them are linked below.

neural-diffeqs
^^^^^^^^^^^^^^
``neural-diffeqs`` is a PyTorch-based library for the instantiation of neural
differential equations, largely inspired by and compatible with Patrick
Kidger's ``torchsde`` library.

.. grid:: 2
    
    .. grid-item::
        :columns: 5

        .. button-link:: https://github.com/mvinyard/neural-diffeqs
            :color: secondary
            :shadow:
            :outline:
            :expand:
            
            :octicon:`mark-github;2em` GitHub/mvinyard/neural-diffeqs

    .. grid-item::
        :columns: 5

        .. button-link:: https://neural-diffeqs.readthedocs.io/en/latest/
            :color: primary
            :shadow:
            :outline:
            :expand:
            
            :octicon:`unfold;2em;` ReadTheDocs


torch-adata
^^^^^^^^^^^
``torch-adata`` is a framework for bridging data held in AnnData - the most
popular single-cell python-based data structure and companion to scanpy - to
the PyTorch Dataset class. torch-adata is meant to be structured but flexible
within the rules of both of these data structures while also being easy to use.

.. grid:: 2
    
    .. grid-item::
        :columns: 5

        .. button-link:: https://github.com/mvinyard/torch-adata
            :color: secondary
            :shadow:
            :outline:
            :expand:
            
            :octicon:`mark-github;2em` GitHub/mvinyard/torch-adata

    .. grid-item::
        :columns: 5

        .. button-link:: https://torch-adata.readthedocs.io/en/latest/
            :color: primary
            :shadow:
            :outline:
            :expand:
            
            :octicon:`unfold;2em;` ReadTheDocs

torch-nets
^^^^^^^^^^
.. grid:: 2
    
    .. grid-item::
        :columns: 5

        .. button-link:: https://github.com/mvinyard/torch-nets
            :color: secondary
            :shadow:
            :outline:
            :expand:
            
            :octicon:`mark-github;2em` GitHub/mvinyard/torch-nets

    .. grid-item::
        :columns: 5

        .. button-link:: https://github.com/mvinyard/torch-nets
            :color: primary
            :shadow:
            :outline:
            :expand:
            
            :octicon:`unfold;2em;` ReadTheDocs

adata-query
^^^^^^^^^^^
.. grid:: 2
    
    .. grid-item::
        :columns: 5

        .. button-link:: https://github.com/mvinyard/AnnDataQuery
            :color: secondary
            :shadow:
            :outline:
            :expand:
            
            :octicon:`mark-github;2em` GitHub/mvinyard/adata-query

    .. grid-item::
        :columns: 5

        .. button-link:: https://anndataquery.readthedocs.io/en/latest/
            :color: primary
            :shadow:
            :outline:
            :expand:
            
            :octicon:`unfold;2em;` ReadTheDocs
ABCParse
^^^^^^^^
.. grid:: 2
    
    .. grid-item::
        :columns: 5

        .. button-link:: https://github.com/mvinyard/ABCParse
            :color: secondary
            :shadow:
            :outline:
            :expand:
            
            :octicon:`mark-github;2em` GitHub/mvinyard/ABCParse

    .. grid-item::
        :columns: 5

        .. button-link:: https://github.com/mvinyard/ABCParse
            :color: primary
            :shadow:
            :outline:
            :expand:
            
            :octicon:`unfold;2em;` ReadTheDocs

autodevice
^^^^^^^^^^
.. grid:: 2
    
    .. grid-item::
        :columns: 5

        .. button-link:: https://github.com/mvinyard/autodevice
            :color: secondary
            :shadow:
            :outline:
            :expand:
            
            :octicon:`mark-github;2em` GitHub/mvinyard/autodevice

    .. grid-item::
        :columns: 5

        .. button-link:: https://github.com/mvinyard/autodevice
            :color: primary
            :shadow:
            :outline:
            :expand:
            
            :octicon:`unfold;2em;` ReadTheDocs

Other, external libraries on which ``scDiffEq`` depends:

* Lightning
* torch
* torchsde
