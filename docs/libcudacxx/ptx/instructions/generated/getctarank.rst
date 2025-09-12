..
   This file was automatically generated. Do not edit.

getctarank.shared::cluster.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // getctarank.space.u32 dest, addr; // PTX ISA 78, SM_90
   // .space     = { .shared::cluster }
   template <typename = void>
   __device__ static inline uint32_t getctarank(
     cuda::ptx::space_cluster_t,
     const void* addr);
