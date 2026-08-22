..
   This file was automatically generated. Do not edit.

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem], size, [smem_bar], ctaMask; // PTX ISA 80, SM_90a, SM_100a, SM_110a
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* srcMem,
     const uint32_t& size,
     uint64_t* smem_bar,
     const uint16_t& ctaMask);
