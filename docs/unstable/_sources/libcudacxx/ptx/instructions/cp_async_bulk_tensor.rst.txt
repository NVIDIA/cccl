.. _libcudacxx-ptx-instructions-cp-async-bulk-tensor:

cp.async.bulk.tensor
====================

-  PTX ISA:
   `cp.async.bulk.tensor <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor>`__

Changelog
---------

-  In earlier versions, ``cp_async_bulk_tensor_multicast`` was enabled
   for SM_90. This has been changed to SM_90a.

Unicast
-------

.. include:: generated/cp_async_bulk_tensor.rst

Multicast
---------

.. include:: generated/cp_async_bulk_tensor_multicast.rst

Scatter / Gather
----------------

.. include:: generated/cp_async_bulk_tensor_gather_scatter.rst
