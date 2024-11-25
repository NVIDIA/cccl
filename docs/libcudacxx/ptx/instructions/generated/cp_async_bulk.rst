..
   This file was automatically generated. Do not edit.

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.dst.src.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [smem_bar]; // 1a. unicast PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename = void>
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
   template <typename = void>
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
   template <typename = void>
   __device__ static inline void cp_async_bulk(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     void* dstMem,
     const void* srcMem,
     const uint32_t& size);
