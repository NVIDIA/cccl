API Reference
=============

.. toctree::
   :maxdepth: 4

.. _cuda-cooperative-common:

Common Conventions, Types, and Primitives
`````````````````````````````````````````

This section documents common CUDA cooperative primitives that are used
throughout the library.

Typing & Type Hints
-------------------

The ``cuda.cooperative.experimental._typing`` module contains type hints
for common CUDA cooperative types.

.. automodule:: cuda.cooperative.experimental._typing
   :members:

Custom Helpers
--------------

ScanOp
~~~~~~

.. automodule:: cuda.cooperative.experimental._scan_op
   :members:

Loading & Storing
`````````````````

Block Load
----------

.. autofunction:: cuda.cooperative.experimental.block.load

Block Store
-----------

.. autofunction:: cuda.cooperative.experimental.block.store

Warp Load
---------

TBD

Warp Store
----------

TBD

Sorting
```````

Merge Sort
----------

Block Merge Sort Keys
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cuda.cooperative.experimental.block.merge_sort_keys

Warp Merge Sort Keys
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cuda.cooperative.experimental.warp.merge_sort_keys

Radix Sort
----------

Block Radix Sort Keys
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cuda.cooperative.experimental.block.radix_sort_keys

Block Radix Sort Keys (Descending)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cuda.cooperative.experimental.block.radix_sort_keys_descending

Warp Radix Sort
~~~~~~~~~~~~~~~

TBD

.. vim: set filetype=rst expandtab ts=8 sw=2 sts=2 tw=72 :
