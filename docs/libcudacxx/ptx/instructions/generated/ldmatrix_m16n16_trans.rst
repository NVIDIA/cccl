..
   This file was automatically generated. Do not edit.

ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m16n16.x1.trans.space.b8 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
   // .space     = { .shared }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void ldmatrix_m16n16_trans(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[2],
     const B8* smem_ptr);

ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b6x16_p32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m16n16.x1.trans.space.b8x16.b6x16_p32 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
   // .space     = { .shared }
   template <typename = void>
   __device__ static inline void ldmatrix_m16n16_trans_b8x16_b6x16_p32(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[1],
     const void* smem_ptr);

ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b4x16_p64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m16n16.x1.trans.space.b8x16.b4x16_p64 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
   // .space     = { .shared }
   template <typename = void>
   __device__ static inline void ldmatrix_m16n16_trans_b8x16_b4x16_p64(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[1],
     const void* smem_ptr);

ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m16n16.x2.trans.space.b8 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
   // .space     = { .shared }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void ldmatrix_m16n16_trans(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[4],
     const B8* smem_ptr);

ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b6x16_p32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m16n16.x2.trans.space.b8x16.b6x16_p32 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
   // .space     = { .shared }
   template <typename = void>
   __device__ static inline void ldmatrix_m16n16_trans_b8x16_b6x16_p32(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[2],
     const void* smem_ptr);

ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b4x16_p64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ldmatrix.sync.aligned.m16n16.x2.trans.space.b8x16.b4x16_p64 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
   // .space     = { .shared }
   template <typename = void>
   __device__ static inline void ldmatrix_m16n16_trans_b8x16_b4x16_p64(
     cuda::ptx::space_shared_t,
     uint32_t (&out)[2],
     const void* smem_ptr);
