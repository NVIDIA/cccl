
shfl.sync
^^^^^^^^^

.. code:: cuda

   // PTX ISA 6.0
   // shfl.sync.mode.b32  d[|p], a, b, c, membermask;
   //   .mode = { .up, .down, .bfly, .idx };

   template<typename T>
   struct shfl_return_values {
       T    data;
       bool pred;
   };

   template<typename T>
   [[nodiscard]] __device__ static inline
   shfl_return_values<T> shfl_sync(shfl_mode_t shfl_mode,
                                   T           data,
                                   uint32_t    lane_idx_offset,
                                   uint32_t    clamp_segmask,
                                   uint32_t    lane_mask) noexcept;

- ``shfl_mode`` is ``shfl_mode_up`` or ``shfl_mode_down`` or ``shfl_mode_bfly`` or ``shfl_mode_idx``
