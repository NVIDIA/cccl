
shfl.sync
^^^^^^^^^

.. code:: cuda

   // PTX ISA 6.0
   // shfl.sync.mode.b32  d[|p], a, b, c, membermask;
   //   .mode = { .up, .down, .bfly, .idx };

   template<typename T>
   [[nodiscard]] __device__ static inline
   T shfl_sync_idx(T        data,
                   uint32_t lane_idx_offset,
                   uint32_t clamp_segmask,
                   uint32_t lane_mask) noexcept;

   template<typename T>
   [[nodiscard]] __device__ static inline
   T shfl_sync_idx(T        data,
                   bool&    pred,
                   uint32_t lane_idx_offset,
                   uint32_t clamp_segmask,
                   uint32_t lane_mask) noexcept;

   template<typename T>
   [[nodiscard]] __device__ static inline
   T shfl_sync_up(T        data,
                  uint32_t lane_idx_offset,
                  uint32_t clamp_segmask,
                  uint32_t lane_mask) noexcept;

   template<typename T>
   [[nodiscard]] __device__ static inline
   T shfl_sync_up(T        data,
                  bool&    pred,
                  uint32_t lane_idx_offset,
                  uint32_t clamp_segmask,
                  uint32_t lane_mask) noexcept;

   template<typename T>
   [[nodiscard]] __device__ static inline
   T shfl_sync_down(T        data,
                    uint32_t lane_idx_offset,
                    uint32_t clamp_segmask,
                    uint32_t lane_mask) noexcept;

   template<typename T>
   [[nodiscard]] __device__ static inline
   T shfl_sync_down(T        data,
                    bool&    pred,
                    uint32_t lane_idx_offset,
                    uint32_t clamp_segmask,
                    uint32_t lane_mask) noexcept;

   template<typename T>
   [[nodiscard]] __device__ static inline
   T shfl_sync_bfly(T        data,
                    uint32_t lane_idx_offset,
                    uint32_t clamp_segmask,
                    uint32_t lane_mask) noexcept;

   template<typename T>
   [[nodiscard]] __device__ static inline
   T shfl_sync_bfly(T        data,
                    bool&    pred,
                    uint32_t lane_idx_offset,
                    uint32_t clamp_segmask,
                    uint32_t lane_mask) noexcept;

**Constrains and checks**

The following conditions are checked at *run-time* in debug mode:

- ``T`` must have 32-bit size (compile-time)
- ``lane_idx_offset`` must be less than the warp size
- ``clamp_segmask`` must use the bit positions [0:4] and [8:12])
- ``lane_mask`` must be a subset of the active mask
- All lanes must have the same value for ``lane_mask``
- The destination lane must be a member of the ``lane_mask``
