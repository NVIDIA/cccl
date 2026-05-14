..
   This file was automatically generated. Do not edit.

cp.async.bulk.prefetch.L2.global.L2::cache_hint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.L2.global.L2::cache_hint [srcMem], size, cache_policy; // PTX ISA 80, SM_90
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch(
     const void* srcMem,
     uint32_t size,
     uint64_t cache_policy = 0x10F0000000000000);

cp.async.bulk.prefetch.L2.global.L2::evict_last
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.prefetch.L2.global.L2::evict_last [srcMem], size; // PTX ISA 94, SM_107a, SM_107f
   template <typename = void>
   __device__ static inline void cp_async_bulk_prefetch_L2_evict_last(
     const void* srcMem,
     uint32_t size);
