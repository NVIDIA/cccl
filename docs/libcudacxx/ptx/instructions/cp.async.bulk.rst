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

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.dst.src.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [smem_bar]; // 1a. unicast PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename=void>
   __device__ static inline void cp_async_bulk(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* srcMem,
     const uint32_t& size,
     uint64_t* smem_bar);

cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.dst.src.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [rdsmem_bar]; // 2.  PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .shared::cta }
   template <typename=void>
   __device__ static inline void cp_async_bulk(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_shared_t,
     void* dstMem,
     const void* srcMem,
     const uint32_t& size,
     uint64_t* rdsmem_bar);

cp.async.bulk.global.shared::cta.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.dst.src.bulk_group [dstMem], [srcMem], size; // 3.  PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename=void>
   __device__ static inline void cp_async_bulk(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     void* dstMem,
     const void* srcMem,
     const uint32_t& size);

Multicast
---------

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk{.dst}{.src}.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem], size, [smem_bar], ctaMask; // 1.  PTX ISA 80, SM_90a
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename=void>
   __device__ static inline void cp_async_bulk(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* srcMem,
     const uint32_t& size,
     uint64_t* smem_bar,
     const uint16_t& ctaMask);
