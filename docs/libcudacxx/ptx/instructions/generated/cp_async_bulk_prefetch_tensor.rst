..
   This file was automatically generated. Do not edit.

cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.1d.L2.src.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy; // PTX ISA 80, SM_90
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.2d.L2.src.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy; // PTX ISA 80, SM_90
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.3d.L2.src.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy; // PTX ISA 80, SM_90
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.4d.L2.src.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy; // PTX ISA 80, SM_90
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.5d.L2.src.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy; // PTX ISA 80, SM_90
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.2d.L2.src.tile::gather4.L2::cache_hint [tensorMap, tensorCoords], cache_policy; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_tile_gather4(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::evict_last
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.1d.L2.src.tile.L2::evict_last [tensorMap, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1]);

cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.1d.L2.src.tile.L2::cache_hint.override::global_address [tensorMap, gAddrToOverride, tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::evict_last.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.1d.L2.src.tile.L2::evict_last.override::global_address [tensorMap, gAddrToOverride, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1]);

cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.1d.L2.src.tile.L2::cache_hint.override::global_address.override::global_dim [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void cp_async_bulk_prefetch_tensor_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.1d.L2.src.tile.L2::evict_last.override::global_address.override::global_dim [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1]);

cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::evict_last
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.2d.L2.src.tile.L2::evict_last [tensorMap, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2]);

cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.2d.L2.src.tile.L2::cache_hint.override::global_address [tensorMap, gAddrToOverride, tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::evict_last.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.2d.L2.src.tile.L2::evict_last.override::global_address [tensorMap, gAddrToOverride, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2]);

cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.2d.L2.src.tile.L2::cache_hint.override::global_address.override::global_dim_stride [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void cp_async_bulk_prefetch_tensor_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.2d.L2.src.tile.L2::evict_last.override::global_address.override::global_dim_stride [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2]);

cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::evict_last
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.3d.L2.src.tile.L2::evict_last [tensorMap, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3]);

cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.3d.L2.src.tile.L2::cache_hint.override::global_address [tensorMap, gAddrToOverride, tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::evict_last.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.3d.L2.src.tile.L2::evict_last.override::global_address [tensorMap, gAddrToOverride, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3]);

cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.3d.L2.src.tile.L2::cache_hint.override::global_address.override::global_dim_stride [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void cp_async_bulk_prefetch_tensor_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.3d.L2.src.tile.L2::evict_last.override::global_address.override::global_dim_stride [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3]);

cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::evict_last
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.4d.L2.src.tile.L2::evict_last [tensorMap, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4]);

cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.4d.L2.src.tile.L2::cache_hint.override::global_address [tensorMap, gAddrToOverride, tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::evict_last.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.4d.L2.src.tile.L2::evict_last.override::global_address [tensorMap, gAddrToOverride, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4]);

cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.4d.L2.src.tile.L2::cache_hint.override::global_address.override::global_dim_stride [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void cp_async_bulk_prefetch_tensor_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.4d.L2.src.tile.L2::evict_last.override::global_address.override::global_dim_stride [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4]);

cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::evict_last
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.5d.L2.src.tile.L2::evict_last [tensorMap, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5]);

cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.5d.L2.src.tile.L2::cache_hint.override::global_address [tensorMap, gAddrToOverride, tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::evict_last.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.5d.L2.src.tile.L2::evict_last.override::global_address [tensorMap, gAddrToOverride, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5]);

cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.5d.L2.src.tile.L2::cache_hint.override::global_address.override::global_dim_stride [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void cp_async_bulk_prefetch_tensor_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.5d.L2.src.tile.L2::evict_last.override::global_address.override::global_dim_stride [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5]);

cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::evict_last
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.2d.L2.src.tile::gather4.L2::evict_last [tensorMap, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_tile_gather4_L2_evict_last(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5]);

cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.2d.L2.src.tile::gather4.L2::cache_hint.override::global_address [tensorMap, gAddrToOverride, tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_tile_gather4_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::evict_last.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.tensor.2d.L2.src.tile::gather4.L2::evict_last.override::global_address [tensorMap, gAddrToOverride, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_tensor_tile_gather4_L2_evict_last_override(
     cuda::ptx::space_global_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5]);
