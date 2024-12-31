
bfind
^^^^^
.. code:: cuda

   // bfind{.shiftamt}{.type}  dest, value;  // PTX ISA 20
   // .type      = { u32, .u64, .s32, .s64 }
   // .shiftamt  = { .inc }
   template <typename T>
   __device__ static inline uint32_t
   bfind(T value,
         cuda::ptx::bfind_shift_amount shiftamt = cuda::ptx::bfind_shift_amount::disable);
