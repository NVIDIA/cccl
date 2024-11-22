.. _libcudacxx-ptx-instructions-cp-async-bulk:

cp.async.bulk
=============

-  PTX ISA:
   `cp.async.bulk <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk>`__

Implementation notes
--------------------

**NOTE.** Both ``srcMem`` and ``dstMem`` must be 16-byte aligned, and
``size`` must be a multiple of 16.

Changelog
---------

-  In earlier versions, ``cp_async_bulk_multicast`` was enabled for
   SM_90. This has been changed to SM_90a.


Unicast
-------

.. include:: generated/cp_async_bulk.rst

Multicast
---------

.. include:: generated/cp_async_bulk_multicast.rst
