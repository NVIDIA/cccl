..
   This file was automatically generated. Do not edit.

bfind.u32
^^^^^^^^^
.. code-block:: cuda

   // bfind.u32 dest, a_reg; // PTX ISA 20, SM_50
   template <typename U32, enable_if_t<sizeof(U32) == 4 && is_integral_v<U32> && is_unsigned_v<U32>, bool> = true>
   __device__ static inline uint32_t bfind(
     U32 a_reg);

bfind.shiftamt.u32
^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // bfind.shiftamt.u32 dest, a_reg; // PTX ISA 20, SM_50
   template <typename U32, enable_if_t<sizeof(U32) == 4 && is_integral_v<U32> && is_unsigned_v<U32>, bool> = true>
   __device__ static inline uint32_t bfind_shiftamt(
     U32 a_reg);

bfind.u64
^^^^^^^^^
.. code-block:: cuda

   // bfind.u64 dest, a_reg; // PTX ISA 20, SM_50
   template <typename U64, enable_if_t<sizeof(U64) == 8 && is_integral_v<U64> && is_unsigned_v<U64>, bool> = true>
   __device__ static inline uint32_t bfind(
     U64 a_reg);

bfind.shiftamt.u64
^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // bfind.shiftamt.u64 dest, a_reg; // PTX ISA 20, SM_50
   template <typename U64, enable_if_t<sizeof(U64) == 8 && is_integral_v<U64> && is_unsigned_v<U64>, bool> = true>
   __device__ static inline uint32_t bfind_shiftamt(
     U64 a_reg);

bfind.s32
^^^^^^^^^
.. code-block:: cuda

   // bfind.s32 dest, a_reg; // PTX ISA 20, SM_50
   template <typename S32, enable_if_t<sizeof(S32) == 4 && is_integral_v<S32> && is_signed_v<S32>, bool> = true>
   __device__ static inline uint32_t bfind(
     S32 a_reg);

bfind.shiftamt.s32
^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // bfind.shiftamt.s32 dest, a_reg; // PTX ISA 20, SM_50
   template <typename S32, enable_if_t<sizeof(S32) == 4 && is_integral_v<S32> && is_signed_v<S32>, bool> = true>
   __device__ static inline uint32_t bfind_shiftamt(
     S32 a_reg);

bfind.s64
^^^^^^^^^
.. code-block:: cuda

   // bfind.s64 dest, a_reg; // PTX ISA 20, SM_50
   template <typename S64, enable_if_t<sizeof(S64) == 8 && is_integral_v<S64> && is_signed_v<S64>, bool> = true>
   __device__ static inline uint32_t bfind(
     S64 a_reg);

bfind.shiftamt.s64
^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // bfind.shiftamt.s64 dest, a_reg; // PTX ISA 20, SM_50
   template <typename S64, enable_if_t<sizeof(S64) == 8 && is_integral_v<S64> && is_signed_v<S64>, bool> = true>
   __device__ static inline uint32_t bfind_shiftamt(
     S64 a_reg);
