.. _cuda_stf_experimental-module:

``cuda.stf._experimental`` API Reference
==========================================

.. warning::
  ``cuda.stf._experimental`` is experimental. The API is subject to change
  without notice.

The core context, logical-data, task, and place types are implemented as a compiled
extension; they are covered in the :ref:`narrative guide <cccl-python-stf>` and the
:ref:`C++ CUDASTF documentation <stf>`. The pure-Python helper layers are documented
below.

Record-once task graphs
-----------------------

.. automodule:: cuda.stf._experimental.task_graph
  :members:
  :undoc-members:

Device allocations
------------------

.. automodule:: cuda.stf._experimental.device_array
  :members:
  :undoc-members:

Path discovery
--------------

.. automodule:: cuda.stf._experimental.paths
  :members:
  :undoc-members:

Numba interop
-------------

.. automodule:: cuda.stf._experimental.interop.numba
  :members:
  :undoc-members:

PyTorch interop
---------------

.. automodule:: cuda.stf._experimental.interop.pytorch
  :members:
  :undoc-members:
