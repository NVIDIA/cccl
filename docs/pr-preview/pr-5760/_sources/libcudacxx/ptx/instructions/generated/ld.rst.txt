..
   This file was automatically generated. Do not edit.

ld.global.b8
^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.b8 dest, [addr]; // PTX ISA 10, SM_50
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.b16
^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.b16 dest, [addr]; // PTX ISA 10, SM_50
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.b32
^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.b32 dest, [addr]; // PTX ISA 10, SM_50
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.b64
^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.b64 dest, [addr]; // PTX ISA 10, SM_50
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.b128
^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.b128 dest, [addr]; // PTX ISA 83, SM_70
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.v4.b64
^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld(
     cuda::ptx::space_global_t,
     const B256* addr);

ld.global.L2::cache_hint.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.L2::cache_hint.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.L2::cache_hint.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.L2::cache_hint.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.L2::cache_hint.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.L2::cache_hint.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.L2::cache_hint.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.L2::cache_hint.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.L2::cache_hint.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.L2::cache_hint.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.L2::cache_hint.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B256* addr,
     uint64_t cache_policy);

ld.global.L1::evict_first.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.b8 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L1_evict_first(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.L1::evict_first.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L1_evict_first_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.L1::evict_first.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.b16 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L1_evict_first(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.L1::evict_first.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L1_evict_first_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.L1::evict_first.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.b32 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L1_evict_first(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.L1::evict_first.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L1_evict_first_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.L1::evict_first.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.b64 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L1_evict_first(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.L1::evict_first.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L1_evict_first_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.L1::evict_first.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.b128 dest, [addr]; // PTX ISA 83, SM_70
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L1_evict_first(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.L1::evict_first.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L1_evict_first_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.L1::evict_first.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_L1_evict_first(
     cuda::ptx::space_global_t,
     const B256* addr);

ld.global.L1::evict_first.L2::cache_hint.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.L1::evict_first.L2::cache_hint.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L1_evict_first_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.L1::evict_first.L2::cache_hint.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.L1::evict_first.L2::cache_hint.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L1_evict_first_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.L1::evict_first.L2::cache_hint.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.L1::evict_first.L2::cache_hint.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L1_evict_first_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.L1::evict_first.L2::cache_hint.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.L1::evict_first.L2::cache_hint.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L1_evict_first_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.L1::evict_first.L2::cache_hint.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.L1::evict_first.L2::cache_hint.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L1_evict_first_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.L1::evict_first.L2::cache_hint.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_first.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B256* addr,
     uint64_t cache_policy);

ld.global.L1::evict_last.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.b8 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L1_evict_last(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.L1::evict_last.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L1_evict_last_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.L1::evict_last.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.b16 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L1_evict_last(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.L1::evict_last.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L1_evict_last_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.L1::evict_last.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.b32 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L1_evict_last(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.L1::evict_last.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L1_evict_last_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.L1::evict_last.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.b64 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L1_evict_last(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.L1::evict_last.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L1_evict_last_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.L1::evict_last.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.b128 dest, [addr]; // PTX ISA 83, SM_70
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L1_evict_last(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.L1::evict_last.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L1_evict_last_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.L1::evict_last.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_L1_evict_last(
     cuda::ptx::space_global_t,
     const B256* addr);

ld.global.L1::evict_last.L2::cache_hint.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.L1::evict_last.L2::cache_hint.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L1_evict_last_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.L1::evict_last.L2::cache_hint.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.L1::evict_last.L2::cache_hint.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L1_evict_last_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.L1::evict_last.L2::cache_hint.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.L1::evict_last.L2::cache_hint.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L1_evict_last_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.L1::evict_last.L2::cache_hint.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.L1::evict_last.L2::cache_hint.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L1_evict_last_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.L1::evict_last.L2::cache_hint.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.L1::evict_last.L2::cache_hint.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L1_evict_last_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.L1::evict_last.L2::cache_hint.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::evict_last.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B256* addr,
     uint64_t cache_policy);

ld.global.L1::no_allocate.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.b8 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L1_no_allocate(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.L1::no_allocate.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L1_no_allocate_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.L1::no_allocate.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.b16 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L1_no_allocate(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.L1::no_allocate.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L1_no_allocate_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.L1::no_allocate.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.b32 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L1_no_allocate(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.L1::no_allocate.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L1_no_allocate_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.L1::no_allocate.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.b64 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L1_no_allocate(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.L1::no_allocate.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L1_no_allocate_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.L1::no_allocate.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.b128 dest, [addr]; // PTX ISA 83, SM_70
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L1_no_allocate(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.L1::no_allocate.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L1_no_allocate_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.L1::no_allocate.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_L1_no_allocate(
     cuda::ptx::space_global_t,
     const B256* addr);

ld.global.L1::no_allocate.L2::cache_hint.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.L1::no_allocate.L2::cache_hint.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_L1_no_allocate_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.L1::no_allocate.L2::cache_hint.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.L1::no_allocate.L2::cache_hint.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_L1_no_allocate_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.L1::no_allocate.L2::cache_hint.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.L1::no_allocate.L2::cache_hint.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_L1_no_allocate_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.L1::no_allocate.L2::cache_hint.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.L1::no_allocate.L2::cache_hint.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_L1_no_allocate_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.L1::no_allocate.L2::cache_hint.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.L1::no_allocate.L2::cache_hint.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_L1_no_allocate_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.L1::no_allocate.L2::cache_hint.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.L1::no_allocate.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B256* addr,
     uint64_t cache_policy);

ld.global.nc.b8
^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.b8 dest, [addr]; // PTX ISA 10, SM_50
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.nc.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.nc.b16
^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.b16 dest, [addr]; // PTX ISA 10, SM_50
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.nc.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.nc.b32
^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.b32 dest, [addr]; // PTX ISA 10, SM_50
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.nc.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.nc.b64
^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.b64 dest, [addr]; // PTX ISA 10, SM_50
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.nc.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.nc.b128
^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.b128 dest, [addr]; // PTX ISA 83, SM_70
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.nc.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.nc.v4.b64
^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_nc(
     cuda::ptx::space_global_t,
     const B256* addr);

ld.global.nc.L2::cache_hint.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.nc.L2::cache_hint.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.nc.L2::cache_hint.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.nc.L2::cache_hint.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.nc.L2::cache_hint.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.nc.L2::cache_hint.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.nc.L2::cache_hint.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.nc.L2::cache_hint.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.nc.L2::cache_hint.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.nc.L2::cache_hint.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.nc.L2::cache_hint.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_nc_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B256* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_first.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.b8 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L1_evict_first(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.nc.L1::evict_first.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L1_evict_first_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.nc.L1::evict_first.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.b16 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L1_evict_first(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.nc.L1::evict_first.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L1_evict_first_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.nc.L1::evict_first.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.b32 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L1_evict_first(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.nc.L1::evict_first.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L1_evict_first_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.nc.L1::evict_first.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.b64 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L1_evict_first(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.nc.L1::evict_first.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L1_evict_first_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.nc.L1::evict_first.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.b128 dest, [addr]; // PTX ISA 83, SM_70
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L1_evict_first(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.nc.L1::evict_first.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L1_evict_first_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.nc.L1::evict_first.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_nc_L1_evict_first(
     cuda::ptx::space_global_t,
     const B256* addr);

ld.global.nc.L1::evict_first.L2::cache_hint.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_first.L2::cache_hint.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_first.L2::cache_hint.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_first.L2::cache_hint.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_first.L2::cache_hint.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_first.L2::cache_hint.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_first.L2::cache_hint.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_first.L2::cache_hint.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_first.L2::cache_hint.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_first.L2::cache_hint.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_first.L2::cache_hint.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_first.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_nc_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B256* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_last.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.b8 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L1_evict_last(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.nc.L1::evict_last.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L1_evict_last_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.nc.L1::evict_last.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.b16 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L1_evict_last(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.nc.L1::evict_last.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L1_evict_last_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.nc.L1::evict_last.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.b32 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L1_evict_last(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.nc.L1::evict_last.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L1_evict_last_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.nc.L1::evict_last.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.b64 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L1_evict_last(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.nc.L1::evict_last.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L1_evict_last_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.nc.L1::evict_last.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.b128 dest, [addr]; // PTX ISA 83, SM_70
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L1_evict_last(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.nc.L1::evict_last.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L1_evict_last_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.nc.L1::evict_last.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_nc_L1_evict_last(
     cuda::ptx::space_global_t,
     const B256* addr);

ld.global.nc.L1::evict_last.L2::cache_hint.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_last.L2::cache_hint.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_last.L2::cache_hint.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_last.L2::cache_hint.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_last.L2::cache_hint.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.nc.L1::evict_last.L2::cache_hint.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::evict_last.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_nc_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B256* addr,
     uint64_t cache_policy);

ld.global.nc.L1::no_allocate.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.b8 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L1_no_allocate(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.nc.L1::no_allocate.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L1_no_allocate_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr);

ld.global.nc.L1::no_allocate.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.b16 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L1_no_allocate(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.nc.L1::no_allocate.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L1_no_allocate_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr);

ld.global.nc.L1::no_allocate.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.b32 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L1_no_allocate(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.nc.L1::no_allocate.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L1_no_allocate_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr);

ld.global.nc.L1::no_allocate.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.b64 dest, [addr]; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L1_no_allocate(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.nc.L1::no_allocate.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L1_no_allocate_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr);

ld.global.nc.L1::no_allocate.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.b128 dest, [addr]; // PTX ISA 83, SM_70
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L1_no_allocate(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.nc.L1::no_allocate.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L1_no_allocate_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr);

ld.global.nc.L1::no_allocate.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_nc_L1_no_allocate(
     cuda::ptx::space_global_t,
     const B256* addr);

ld.global.nc.L1::no_allocate.L2::cache_hint.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline B8 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B8* addr,
     uint64_t cache_policy);

ld.global.nc.L1::no_allocate.L2::cache_hint.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline B16 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B16* addr,
     uint64_t cache_policy);

ld.global.nc.L1::no_allocate.L2::cache_hint.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B32* addr,
     uint64_t cache_policy);

ld.global.nc.L1::no_allocate.L2::cache_hint.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B64* addr,
     uint64_t cache_policy);

ld.global.nc.L1::no_allocate.L2::cache_hint.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline B128 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
     cuda::ptx::space_global_t,
     const B128* addr,
     uint64_t cache_policy);

ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // ld.space.nc.L1::no_allocate.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline B256 ld_nc_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     const B256* addr,
     uint64_t cache_policy);
