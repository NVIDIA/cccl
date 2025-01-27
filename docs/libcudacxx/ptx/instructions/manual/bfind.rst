
bfind
^^^^^
.. code:: cuda

   // PTX ISA 20
   // bfind{.shiftamt}{.type}  dest, value;
   //   .type      = { u32, .u64, .s32, .s64 }
   //   .shiftamt  = { .inc }

   template <typename T>
   [[nodiscard]] __device__ static inline
   uint32_t bfind(T                             value,
                  cuda::ptx::bfind_shift_amount shiftamt = cuda::ptx::bfind_shift_amount::disable) noexcept;


- ``T`` is a 32-bit or 64-bit integer type.
- ``shiftamt`` is ``cuda::ptx::bfind_shift_amount::disable`` (default) or ``cuda::ptx::bfind_shift_amount::enable``.
