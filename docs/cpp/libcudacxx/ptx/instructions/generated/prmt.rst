..
   This file was automatically generated. Do not edit.

prmt.b32
^^^^^^^^
.. code-block:: cuda

   // prmt.b32 dest, a_reg, b_reg, c_reg; // PTX ISA 20, SM_50
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline uint32_t prmt(
     B32 a_reg,
     B32 b_reg,
     uint32_t c_reg);

prmt.b32.f4e
^^^^^^^^^^^^
.. code-block:: cuda

   // prmt.b32.f4e dest, a_reg, b_reg, c_reg; // PTX ISA 20, SM_50
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline uint32_t prmt_f4e(
     B32 a_reg,
     B32 b_reg,
     uint32_t c_reg);

prmt.b32.b4e
^^^^^^^^^^^^
.. code-block:: cuda

   // prmt.b32.b4e dest, a_reg, b_reg, c_reg; // PTX ISA 20, SM_50
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline uint32_t prmt_b4e(
     B32 a_reg,
     B32 b_reg,
     uint32_t c_reg);

prmt.b32.rc8
^^^^^^^^^^^^
.. code-block:: cuda

   // prmt.b32.rc8 dest, a_reg, b_reg, c_reg; // PTX ISA 20, SM_50
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline uint32_t prmt_rc8(
     B32 a_reg,
     B32 b_reg,
     uint32_t c_reg);

prmt.b32.ecl
^^^^^^^^^^^^
.. code-block:: cuda

   // prmt.b32.ecl dest, a_reg, b_reg, c_reg; // PTX ISA 20, SM_50
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline uint32_t prmt_ecl(
     B32 a_reg,
     B32 b_reg,
     uint32_t c_reg);

prmt.b32.ecr
^^^^^^^^^^^^
.. code-block:: cuda

   // prmt.b32.ecr dest, a_reg, b_reg, c_reg; // PTX ISA 20, SM_50
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline uint32_t prmt_ecr(
     B32 a_reg,
     B32 b_reg,
     uint32_t c_reg);

prmt.b32.rc16
^^^^^^^^^^^^^
.. code-block:: cuda

   // prmt.b32.rc16 dest, a_reg, b_reg, c_reg; // PTX ISA 20, SM_50
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline uint32_t prmt_rc16(
     B32 a_reg,
     B32 b_reg,
     uint32_t c_reg);
