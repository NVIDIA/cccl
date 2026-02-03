..
   This file was automatically generated. Do not edit.

ldmatrix.sync.aligned.m8n8.x1.shared.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m8n8.x1.space.b16 out, [smem_ptr]; // PTX ISA 65, SM_75
   // .space     = { .shared }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void ldmatrix_m8n8(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[1],
     const B16* smem_ptr);

ldmatrix.sync.aligned.m8n8.x2.shared.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m8n8.x2.space.b16 out, [smem_ptr]; // PTX ISA 65, SM_75
   // .space     = { .shared }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void ldmatrix_m8n8(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[2],
     const B16* smem_ptr);

ldmatrix.sync.aligned.m8n8.x4.shared.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m8n8.x4.space.b16 out, [smem_ptr]; // PTX ISA 65, SM_75
   // .space     = { .shared }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void ldmatrix_m8n8(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[4],
     const B16* smem_ptr);

ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m8n8.x1.trans.space.b16 out, [smem_ptr]; // PTX ISA 65, SM_75
   // .space     = { .shared }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void ldmatrix_m8n8_trans(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[1],
     const B16* smem_ptr);

ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m8n8.x2.trans.space.b16 out, [smem_ptr]; // PTX ISA 65, SM_75
   // .space     = { .shared }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void ldmatrix_m8n8_trans(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[2],
     const B16* smem_ptr);

ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m8n8.x4.trans.space.b16 out, [smem_ptr]; // PTX ISA 65, SM_75
   // .space     = { .shared }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void ldmatrix_m8n8_trans(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[4],
     const B16* smem_ptr);

ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m8n16.x1.space.b8x16.b6x16_p32 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
   // .space     = { .shared }
   template <typename = void>
   __device__ static inline void ldmatrix_m8n16_b8x16_b6x16_p32(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[1],
     const void* smem_ptr);

ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m8n16.x1.space.b8x16.b4x16_p64 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
   // .space     = { .shared }
   template <typename = void>
   __device__ static inline void ldmatrix_m8n16_b8x16_b4x16_p64(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[1],
     const void* smem_ptr);

ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b6x16_p32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m8n16.x2.space.b8x16.b6x16_p32 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
   // .space     = { .shared }
   template <typename = void>
   __device__ static inline void ldmatrix_m8n16_b8x16_b6x16_p32(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[2],
     const void* smem_ptr);

ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m8n16.x2.space.b8x16.b4x16_p64 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
   // .space     = { .shared }
   template <typename = void>
   __device__ static inline void ldmatrix_m8n16_b8x16_b4x16_p64(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[2],
     const void* smem_ptr);

ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m8n16.x4.space.b8x16.b6x16_p32 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
   // .space     = { .shared }
   template <typename = void>
   __device__ static inline void ldmatrix_m8n16_b8x16_b6x16_p32(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[4],
     const void* smem_ptr);

ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m8n16.x4.space.b8x16.b4x16_p64 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
   // .space     = { .shared }
   template <typename = void>
   __device__ static inline void ldmatrix_m8n16_b8x16_b4x16_p64(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[4],
     const void* smem_ptr);
