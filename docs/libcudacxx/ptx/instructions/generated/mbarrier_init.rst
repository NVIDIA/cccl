..
   This file was automatically generated. Do not edit.

mbarrier.init.shared.b64
^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.init.shared.b64 [addr], count; // PTX ISA 70, SM_80
   template <typename = void>
   __device__ static inline void mbarrier_init(
     uint64_t* addr,
     const uint32_t& count);

mbarrier.init.layout::v0.shared.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.init.layout.shared.b64 [addr], count; // PTX ISA 94, SM_90
   // .layout    = { .layout::v0, .layout::v1 }
   template <cuda::ptx::dot_layout Layout>
   __device__ static inline void mbarrier_init(
     cuda::ptx::layout_t<Layout> layout,
     uint64_t* addr,
     const uint32_t& count);

mbarrier.init.layout::v1.shared.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.init.layout.shared.b64 [addr], count; // PTX ISA 94, SM_90
   // .layout    = { .layout::v0, .layout::v1 }
   template <cuda::ptx::dot_layout Layout>
   __device__ static inline void mbarrier_init(
     cuda::ptx::layout_t<Layout> layout,
     uint64_t* addr,
     const uint32_t& count);
