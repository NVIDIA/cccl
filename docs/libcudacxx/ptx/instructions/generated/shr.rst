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
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 shr(
     B32 a_reg,
     uint32_t b_reg);

shr.b64
^^^^^^^
.. code-block:: cuda

   // shr.b64 dest, a_reg, b_reg; // PTX ISA 10, SM_50
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
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
   template <typename = void>
   __device__ static inline int32_t shr(
     int32_t a_reg,
     uint32_t b_reg);

shr.s64
^^^^^^^
.. code-block:: cuda

   // shr.s64 dest, a_reg, b_reg; // PTX ISA 10, SM_50
   template <typename = void>
   __device__ static inline int64_t shr(
     int64_t a_reg,
     uint32_t b_reg);
