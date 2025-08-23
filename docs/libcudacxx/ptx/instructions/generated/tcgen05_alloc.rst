..
   This file was automatically generated. Do not edit.

tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.alloc.cta_group.sync.aligned.shared::cta.b32 [dst], nCols; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_alloc(
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t* dst,
     const uint32_t& nCols);

tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.alloc.cta_group.sync.aligned.shared::cta.b32 [dst], nCols; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_alloc(
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t* dst,
     const uint32_t& nCols);

tcgen05.dealloc.cta_group::1.sync.aligned.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.dealloc.cta_group.sync.aligned.b32 taddr, nCols; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_dealloc(
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t taddr,
     const uint32_t& nCols);

tcgen05.dealloc.cta_group::2.sync.aligned.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.dealloc.cta_group.sync.aligned.b32 taddr, nCols; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_dealloc(
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t taddr,
     const uint32_t& nCols);

tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.relinquish_alloc_permit.cta_group.sync.aligned; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_relinquish_alloc_permit(
     cuda::ptx::cta_group_t<Cta_Group> cta_group);

tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.relinquish_alloc_permit.cta_group.sync.aligned; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_relinquish_alloc_permit(
     cuda::ptx::cta_group_t<Cta_Group> cta_group);
