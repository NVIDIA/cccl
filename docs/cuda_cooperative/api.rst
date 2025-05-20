API Reference
=============

.. toctree::
   :maxdepth: 4

.. _cuda-cooperative-common:

Primitives
``````````

Loading & Storing
-----------------

Block Load
~~~~~~~~~~

.. autofunction:: cuda.cooperative.experimental.block.load

Block Store
~~~~~~~~~~~

.. autofunction:: cuda.cooperative.experimental.block.store

Warp Load
~~~~~~~~~

TBD

Warp Store
~~~~~~~~~~

Sorting
-------

Merge Sort
~~~~~~~~~~

Block Merge Sort Keys
+++++++++++++++++++++

.. autofunction:: cuda.cooperative.experimental.block.merge_sort_keys

Warp Merge Sort Keys
++++++++++++++++++++

.. autofunction:: cuda.cooperative.experimental.warp.merge_sort_keys

Radix Sort
~~~~~~~~~~

Block Radix Sort Keys
+++++++++++++++++++++

.. autofunction:: cuda.cooperative.experimental.block.radix_sort_keys

Block Radix Sort Keys (Descending)
++++++++++++++++++++++++++++++++++

.. autofunction:: cuda.cooperative.experimental.block.radix_sort_keys_descending

Warp Radix Sort Keys
++++++++++++++++++++

Not yet implemented.

Warp Radix Sort Keys (Descending)
+++++++++++++++++++++++++++++++++

Not yet implemented.

Parallel Prefix Scans
---------------------

See also: :ref:`Parallel Prefix Scans: Overview <cuda_cooperative-scan-overview>`.

Block Scan
++++++++++

.. autofunction:: cuda.cooperative.experimental.block.scan

Prefix Sum
~~~~~~~~~~

Block Inclusive Sum
+++++++++++++++++++

.. autofunction:: cuda.cooperative.experimental.block.inclusive_sum

Block Exclusive Sum
+++++++++++++++++++

.. autofunction:: cuda.cooperative.experimental.block.exclusive_sum

Warp Inclusive Sum
++++++++++++++++++

Not yet implemented.

.. autofunction cuda.cooperative.experimental.warp.inclusive_sum

Warp Exclusive Sum
++++++++++++++++++

.. autofunction:: cuda.cooperative.experimental.warp.exclusive_sum

Prefix Scan
~~~~~~~~~~~

Block Inclusive Scan
++++++++++++++++++++

.. autofunction:: cuda.cooperative.experimental.block.inclusive_scan

Block Exclusive Scan
++++++++++++++++++++

.. autofunction:: cuda.cooperative.experimental.block.exclusive_scan

Warp Inclusive Scan
+++++++++++++++++++

Not yet implemented.

Warp Exclusive Scan
+++++++++++++++++++

Not yet implemented.

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



.. vim: set filetype=rst expandtab ts=8 sw=2 sts=2 tw=72 :
