..
   This file was automatically generated. Do not edit.

mbarrier.check_layout.layout::v0.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.check_layout.layout.shared::cta.b64 p, [addr]; // PTX ISA 94, SM_90
   // .layout    = { .layout::v0, .layout::v1 }
   template <cuda::ptx::dot_layout Layout>
   __device__ static inline bool mbarrier_check_layout(
     cuda::ptx::layout_t<Layout> layout,
     const uint64_t* addr);

mbarrier.check_layout.layout::v1.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.check_layout.layout.shared::cta.b64 p, [addr]; // PTX ISA 94, SM_90
   // .layout    = { .layout::v0, .layout::v1 }
   template <cuda::ptx::dot_layout Layout>
   __device__ static inline bool mbarrier_check_layout(
     cuda::ptx::layout_t<Layout> layout,
     const uint64_t* addr);
