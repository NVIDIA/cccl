.. _libcudacxx-ptx-instructions-tensormap-replace:

tensormap.replace
=================

-  PTX ISA:
   `tensormap.replace <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace>`__

tensormap.replace.tile.global_address.global.b1024.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.global_address.space.b1024.b64    [tm_addr], new_val; // PTX ISA 83, SM_90a
   // .space     = { .global }
   template <typename B64>
   __device__ static inline void tensormap_replace_global_address(
     cuda::ptx::space_global_t,
     void* tm_addr,
     B64 new_val);

tensormap.replace.tile.global_address.shared::cta.b1024.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.global_address.space.b1024.b64    [tm_addr], new_val; // PTX ISA 83, SM_90a
   // .space     = { .shared::cta }
   template <typename B64>
   __device__ static inline void tensormap_replace_global_address(
     cuda::ptx::space_shared_t,
     void* tm_addr,
     B64 new_val);

tensormap.replace.tile.rank.global.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.rank.space.b1024.b32              [tm_addr], new_val; // PTX ISA 83, SM_90a
   // .space     = { .global }
   template <typename B32>
   __device__ static inline void tensormap_replace_rank(
     cuda::ptx::space_global_t,
     void* tm_addr,
     B32 new_val);

tensormap.replace.tile.rank.shared::cta.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.rank.space.b1024.b32              [tm_addr], new_val; // PTX ISA 83, SM_90a
   // .space     = { .shared::cta }
   template <typename B32>
   __device__ static inline void tensormap_replace_rank(
     cuda::ptx::space_shared_t,
     void* tm_addr,
     B32 new_val);

tensormap.replace.tile.box_dim.global.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.box_dim.space.b1024.b32           [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
   // .space     = { .global }
   template <int N32, typename B32>
   __device__ static inline void tensormap_replace_box_dim(
     cuda::ptx::space_global_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> ord,
     B32 new_val);

tensormap.replace.tile.box_dim.shared::cta.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.box_dim.space.b1024.b32           [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
   // .space     = { .shared::cta }
   template <int N32, typename B32>
   __device__ static inline void tensormap_replace_box_dim(
     cuda::ptx::space_shared_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> ord,
     B32 new_val);

tensormap.replace.tile.global_dim.global.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.global_dim.space.b1024.b32        [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
   // .space     = { .global }
   template <int N32, typename B32>
   __device__ static inline void tensormap_replace_global_dim(
     cuda::ptx::space_global_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> ord,
     B32 new_val);

tensormap.replace.tile.global_dim.shared::cta.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.global_dim.space.b1024.b32        [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
   // .space     = { .shared::cta }
   template <int N32, typename B32>
   __device__ static inline void tensormap_replace_global_dim(
     cuda::ptx::space_shared_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> ord,
     B32 new_val);

tensormap.replace.tile.global_stride.global.b1024.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.global_stride.space.b1024.b64     [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
   // .space     = { .global }
   template <int N32, typename B64>
   __device__ static inline void tensormap_replace_global_stride(
     cuda::ptx::space_global_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> ord,
     B64 new_val);

tensormap.replace.tile.global_stride.shared::cta.b1024.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.global_stride.space.b1024.b64     [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
   // .space     = { .shared::cta }
   template <int N32, typename B64>
   __device__ static inline void tensormap_replace_global_stride(
     cuda::ptx::space_shared_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> ord,
     B64 new_val);

tensormap.replace.tile.element_stride.global.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.element_stride.space.b1024.b32    [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
   // .space     = { .global }
   template <int N32, typename B32>
   __device__ static inline void tensormap_replace_element_size(
     cuda::ptx::space_global_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> ord,
     B32 new_val);

tensormap.replace.tile.element_stride.shared::cta.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.element_stride.space.b1024.b32    [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
   // .space     = { .shared::cta }
   template <int N32, typename B32>
   __device__ static inline void tensormap_replace_element_size(
     cuda::ptx::space_shared_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> ord,
     B32 new_val);

tensormap.replace.tile.elemtype.global.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.elemtype.space.b1024.b32          [tm_addr], new_val; // PTX ISA 83, SM_90a
   // .space     = { .global }
   template <int N32>
   __device__ static inline void tensormap_replace_elemtype(
     cuda::ptx::space_global_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> new_val);

tensormap.replace.tile.elemtype.shared::cta.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.elemtype.space.b1024.b32          [tm_addr], new_val; // PTX ISA 83, SM_90a
   // .space     = { .shared::cta }
   template <int N32>
   __device__ static inline void tensormap_replace_elemtype(
     cuda::ptx::space_shared_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> new_val);

tensormap.replace.tile.interleave_layout.global.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.interleave_layout.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a
   // .space     = { .global }
   template <int N32>
   __device__ static inline void tensormap_replace_interleave_layout(
     cuda::ptx::space_global_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> new_val);

tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.interleave_layout.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a
   // .space     = { .shared::cta }
   template <int N32>
   __device__ static inline void tensormap_replace_interleave_layout(
     cuda::ptx::space_shared_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> new_val);

tensormap.replace.tile.swizzle_mode.global.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.swizzle_mode.space.b1024.b32      [tm_addr], new_val; // PTX ISA 83, SM_90a
   // .space     = { .global }
   template <int N32>
   __device__ static inline void tensormap_replace_swizzle_mode(
     cuda::ptx::space_global_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> new_val);

tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.swizzle_mode.space.b1024.b32      [tm_addr], new_val; // PTX ISA 83, SM_90a
   // .space     = { .shared::cta }
   template <int N32>
   __device__ static inline void tensormap_replace_swizzle_mode(
     cuda::ptx::space_shared_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> new_val);

tensormap.replace.tile.fill_mode.global.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.fill_mode.space.b1024.b32         [tm_addr], new_val; // PTX ISA 83, SM_90a
   // .space     = { .global }
   template <int N32>
   __device__ static inline void tensormap_replace_fill_mode(
     cuda::ptx::space_global_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> new_val);

tensormap.replace.tile.fill_mode.shared::cta.b1024.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // tensormap.replace.tile.fill_mode.space.b1024.b32         [tm_addr], new_val; // PTX ISA 83, SM_90a
   // .space     = { .shared::cta }
   template <int N32>
   __device__ static inline void tensormap_replace_fill_mode(
     cuda::ptx::space_shared_t,
     void* tm_addr,
     cuda::ptx::n32_t<N32> new_val);
