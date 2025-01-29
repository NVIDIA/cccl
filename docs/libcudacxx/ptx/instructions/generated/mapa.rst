..
   This file was automatically generated. Do not edit.

mapa.shared::cluster.u32
^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // mapa.space.u32  dest, addr, target_cta; // PTX ISA 78, SM_90
   // .space     = { .shared::cluster }
   template <typename Tp>
   __device__ static inline Tp* mapa(
     cuda::ptx::space_cluster_t,
     const Tp* addr,
     uint32_t target_cta);
