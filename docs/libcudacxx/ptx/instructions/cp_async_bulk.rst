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
fallback path that calls into the driver
(``__cuda_syscall_cp_async_bulk_unicast``) next to the native copy, see
`#6302 <https://github.com/NVIDIA/cccl/issues/6302>`__. On the first launch of
such a kernel, the driver was also observed to raise ``cudaLimitStackSize`` from
1024 to about 14.4 KB per thread, although ``ptxas`` reported a 0-byte stack
frame, and to keep the corresponding device memory reserved for the lifetime of
the context (about 1.3 GiB on a 70-SM GPU). This was seen with CUDA 13.4 and
driver 617.42 under WSL2; lowering the limit with ``cudaDeviceSetLimit`` was
undone by the next launch. The same copy with the ``.shared::cta`` destination
had no fallback path and left the limit unchanged. ``cuda::memcpy_async`` and
``cuda::device::memcpy_async_tx`` already use ``.shared::cta`` when PTX ISA 86
is available.

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
