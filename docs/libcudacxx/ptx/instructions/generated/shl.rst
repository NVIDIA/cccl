..
   This file was automatically generated. Do not edit.

shl.b16
^^^^^^^
.. code-block:: cuda

   // shl.b16 dest, a_reg, b_reg; // PTX ISA 10, SM_50
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 shl(
     B16 a_reg,
     uint32_t b_reg);

shl.b32
^^^^^^^
.. code-block:: cuda

   // shl.b32 dest, a_reg, b_reg; // PTX ISA 10, SM_50
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 shl(
     B32 a_reg,
     uint32_t b_reg);

shl.b64
^^^^^^^
.. code-block:: cuda

   // shl.b64 dest, a_reg, b_reg; // PTX ISA 10, SM_50
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 shl(
     B64 a_reg,
     uint32_t b_reg);
