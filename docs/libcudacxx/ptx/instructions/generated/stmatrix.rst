..
   This file was automatically generated. Do not edit.

stmatrix.sync.aligned.m8n8.x1.shared.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // stmatrix.sync.aligned.m8n8.x1.shared.b16 [gmem_ptr], input; // PTX ISA 78, SM_90
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void stmatrix_m8n8(
     B16* gmem_ptr,
     const uint32_t (&input)[1]);

stmatrix.sync.aligned.m8n8.x2.shared.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // stmatrix.sync.aligned.m8n8.x2.shared.b16 [gmem_ptr], input; // PTX ISA 78, SM_90
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void stmatrix_m8n8(
     B16* gmem_ptr,
     const uint32_t (&input)[2]);

stmatrix.sync.aligned.m8n8.x4.shared.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // stmatrix.sync.aligned.m8n8.x4.shared.b16 [gmem_ptr], input; // PTX ISA 78, SM_90
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void stmatrix_m8n8(
     B16* gmem_ptr,
     const uint32_t (&input)[4]);

stmatrix.sync.aligned.m8n8.x1.trans.shared.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [gmem_ptr], input; // PTX ISA 78, SM_90
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void stmatrix_m8n8_trans(
     B16* gmem_ptr,
     const uint32_t (&input)[1]);

stmatrix.sync.aligned.m8n8.x2.trans.shared.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [gmem_ptr], input; // PTX ISA 78, SM_90
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void stmatrix_m8n8_trans(
     B16* gmem_ptr,
     const uint32_t (&input)[2]);

stmatrix.sync.aligned.m8n8.x4.trans.shared.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [gmem_ptr], input; // PTX ISA 78, SM_90
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void stmatrix_m8n8_trans(
     B16* gmem_ptr,
     const uint32_t (&input)[4]);

stmatrix.sync.aligned.m16n8.x1.trans.shared.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [gmem_ptr], input; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void stmatrix_m16n8_trans(
     B8* gmem_ptr,
     const uint32_t (&input)[1]);

stmatrix.sync.aligned.m16n8.x2.trans.shared.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [gmem_ptr], input; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void stmatrix_m16n8_trans(
     B8* gmem_ptr,
     const uint32_t (&input)[2]);

stmatrix.sync.aligned.m16n8.x4.trans.shared.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [gmem_ptr], input; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void stmatrix_m16n8_trans(
     B8* gmem_ptr,
     const uint32_t (&input)[4]);
