
shfl.sync
^^^^^^^^^

.. code:: cuda

   // PTX ISA 6.0
   // shfl.sync.mode.b32  d[|p], a, b, c, membermask;
   //   .mode = { .up, .down, .bfly, .idx };

   struct shfl_return_values {
       uint32_t data;
       bool     pred;
   };

   [[nodiscard]] __device__ static inline
   shfl_return_values shfl_sync(shfl_mode_t shfl_mode,
                                uint32_t    data,
                                uint32_t    lane_idx_offset,
                                uint32_t    clamp_segmask,
                                uint32_t    lane_mask) noexcept;

- ``shfl_mode`` is ``shfl_mode_up`` or ``shfl_mode_down`` or ``shfl_mode_bfly`` or ``shfl_mode_idx``
