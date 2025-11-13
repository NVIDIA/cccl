..
   This file was automatically generated. Do not edit.

setmaxnreg.inc.sync.aligned.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // setmaxnreg.inc.sync.aligned.u32 imm_reg_count; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
   template <int N32>
   __device__ static inline void setmaxnreg_inc(
     cuda::ptx::n32_t<N32> imm_reg_count);

setmaxnreg.dec.sync.aligned.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // setmaxnreg.dec.sync.aligned.u32 imm_reg_count; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
   template <int N32>
   __device__ static inline void setmaxnreg_dec(
     cuda::ptx::n32_t<N32> imm_reg_count);
