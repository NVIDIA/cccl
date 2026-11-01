..
   This file was automatically generated. Do not edit.

shr.b16
^^^^^^^
.. code-block:: cuda

   // shr.b16 dest, a_reg, b_reg; // PTX ISA 10, SM_50
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 shr(
     B16 a_reg,
     uint32_t b_reg);

shr.b32
^^^^^^^
.. code-block:: cuda

   // shr.b32 dest, a_reg, b_reg; // PTX ISA 10, SM_50
   template <typename B32, enable_if_t<sizeof(B32) == 4 && !(is_integral_v<B32> && is_signed_v<B32>), bool> = true>
   __device__ static inline B32 shr(
     B32 a_reg,
     uint32_t b_reg);

shr.b64
^^^^^^^
.. code-block:: cuda

   // shr.b64 dest, a_reg, b_reg; // PTX ISA 10, SM_50
   template <typename B64, enable_if_t<sizeof(B64) == 8 && !(is_integral_v<B64> && is_signed_v<B64>), bool> = true>
   __device__ static inline B64 shr(
     B64 a_reg,
     uint32_t b_reg);

shr.s16
^^^^^^^
.. code-block:: cuda

   // shr.s16 dest, a_reg, b_reg; // PTX ISA 10, SM_50
   template <typename = void>
   __device__ static inline int16_t shr(
     int16_t a_reg,
     uint32_t b_reg);

shr.s32
^^^^^^^
.. code-block:: cuda

   // shr.s32 dest, a_reg, b_reg; // PTX ISA 10, SM_50
   template <typename S32, enable_if_t<sizeof(S32) == 4 && is_integral_v<S32> && is_signed_v<S32>, bool> = true>
   __device__ static inline S32 shr(
     S32 a_reg,
     uint32_t b_reg);

shr.s64
^^^^^^^
.. code-block:: cuda

   // shr.s64 dest, a_reg, b_reg; // PTX ISA 10, SM_50
   template <typename S64, enable_if_t<sizeof(S64) == 8 && is_integral_v<S64> && is_signed_v<S64>, bool> = true>
   __device__ static inline S64 shr(
     S64 a_reg,
     uint32_t b_reg);
