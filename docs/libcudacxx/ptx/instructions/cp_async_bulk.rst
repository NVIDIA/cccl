.. _libcudacxx-ptx-instructions-cp-async-bulk:

cp.async.bulk
=============

-  PTX ISA:
   `cp.async.bulk <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk>`__

Implementation notes
--------------------

**NOTE.** Both ``srcMem`` and ``dstMem`` must be 16-byte aligned, and
``size`` must be a multiple of 16.

**NOTE.** For global-to-shared copies into the executing CTA's own shared
memory, prefer the ``.shared::cta`` destination (``cuda::ptx::space_shared``,
PTX ISA 86) over ``.shared::cluster`` (``cuda::ptx::space_cluster``). When
compiling for SM_120, ``ptxas`` emits the ``.shared::cluster`` form with a
fallback path that calls into the driver, raising each thread's stack limit and requiring additional device memory.
See `#6302 <https://github.com/NVIDIA/cccl/issues/6302>`__ for more details.

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
