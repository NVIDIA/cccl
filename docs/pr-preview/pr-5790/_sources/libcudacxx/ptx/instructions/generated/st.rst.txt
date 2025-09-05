..
   This file was automatically generated. Do not edit.

st.global.b8
^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.b8 [addr], src; // PTX ISA 10, SM_50
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void st(
     cuda::ptx::space_global_t,
     B8* addr,
     B8 src);

st.global.b16
^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.b16 [addr], src; // PTX ISA 10, SM_50
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void st(
     cuda::ptx::space_global_t,
     B16* addr,
     B16 src);

st.global.b32
^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.b32 [addr], src; // PTX ISA 10, SM_50
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st(
     cuda::ptx::space_global_t,
     B32* addr,
     B32 src);

st.global.b64
^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.b64 [addr], src; // PTX ISA 10, SM_50
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void st(
     cuda::ptx::space_global_t,
     B64* addr,
     B64 src);

st.global.b128
^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.b128 [addr], src; // PTX ISA 83, SM_70
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void st(
     cuda::ptx::space_global_t,
     B128* addr,
     B128 src);

st.global.v4.b64
^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.v4.b64 [addr], src; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline void st(
     cuda::ptx::space_global_t,
     B256* addr,
     B256 src);

st.global.L2::cache_hint.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L2::cache_hint.b8 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void st_L2_cache_hint(
     cuda::ptx::space_global_t,
     B8* addr,
     B8 src,
     uint64_t cache_policy);

st.global.L2::cache_hint.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L2::cache_hint.b16 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void st_L2_cache_hint(
     cuda::ptx::space_global_t,
     B16* addr,
     B16 src,
     uint64_t cache_policy);

st.global.L2::cache_hint.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L2::cache_hint.b32 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st_L2_cache_hint(
     cuda::ptx::space_global_t,
     B32* addr,
     B32 src,
     uint64_t cache_policy);

st.global.L2::cache_hint.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L2::cache_hint.b64 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void st_L2_cache_hint(
     cuda::ptx::space_global_t,
     B64* addr,
     B64 src,
     uint64_t cache_policy);

st.global.L2::cache_hint.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L2::cache_hint.b128 [addr], src, cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void st_L2_cache_hint(
     cuda::ptx::space_global_t,
     B128* addr,
     B128 src,
     uint64_t cache_policy);

st.global.L2::cache_hint.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L2::cache_hint.v4.b64 [addr], src, cache_policy; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline void st_L2_cache_hint(
     cuda::ptx::space_global_t,
     B256* addr,
     B256 src,
     uint64_t cache_policy);

st.global.L1::evict_first.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_first.b8 [addr], src; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void st_L1_evict_first(
     cuda::ptx::space_global_t,
     B8* addr,
     B8 src);

st.global.L1::evict_first.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_first.b16 [addr], src; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void st_L1_evict_first(
     cuda::ptx::space_global_t,
     B16* addr,
     B16 src);

st.global.L1::evict_first.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_first.b32 [addr], src; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st_L1_evict_first(
     cuda::ptx::space_global_t,
     B32* addr,
     B32 src);

st.global.L1::evict_first.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_first.b64 [addr], src; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void st_L1_evict_first(
     cuda::ptx::space_global_t,
     B64* addr,
     B64 src);

st.global.L1::evict_first.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_first.b128 [addr], src; // PTX ISA 83, SM_70
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void st_L1_evict_first(
     cuda::ptx::space_global_t,
     B128* addr,
     B128 src);

st.global.L1::evict_first.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_first.v4.b64 [addr], src; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline void st_L1_evict_first(
     cuda::ptx::space_global_t,
     B256* addr,
     B256 src);

st.global.L1::evict_first.L2::cache_hint.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_first.L2::cache_hint.b8 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void st_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     B8* addr,
     B8 src,
     uint64_t cache_policy);

st.global.L1::evict_first.L2::cache_hint.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_first.L2::cache_hint.b16 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void st_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     B16* addr,
     B16 src,
     uint64_t cache_policy);

st.global.L1::evict_first.L2::cache_hint.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_first.L2::cache_hint.b32 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     B32* addr,
     B32 src,
     uint64_t cache_policy);

st.global.L1::evict_first.L2::cache_hint.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_first.L2::cache_hint.b64 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void st_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     B64* addr,
     B64 src,
     uint64_t cache_policy);

st.global.L1::evict_first.L2::cache_hint.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_first.L2::cache_hint.b128 [addr], src, cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void st_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     B128* addr,
     B128 src,
     uint64_t cache_policy);

st.global.L1::evict_first.L2::cache_hint.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_first.L2::cache_hint.v4.b64 [addr], src, cache_policy; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline void st_L1_evict_first_L2_cache_hint(
     cuda::ptx::space_global_t,
     B256* addr,
     B256 src,
     uint64_t cache_policy);

st.global.L1::evict_last.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_last.b8 [addr], src; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void st_L1_evict_last(
     cuda::ptx::space_global_t,
     B8* addr,
     B8 src);

st.global.L1::evict_last.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_last.b16 [addr], src; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void st_L1_evict_last(
     cuda::ptx::space_global_t,
     B16* addr,
     B16 src);

st.global.L1::evict_last.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_last.b32 [addr], src; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st_L1_evict_last(
     cuda::ptx::space_global_t,
     B32* addr,
     B32 src);

st.global.L1::evict_last.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_last.b64 [addr], src; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void st_L1_evict_last(
     cuda::ptx::space_global_t,
     B64* addr,
     B64 src);

st.global.L1::evict_last.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_last.b128 [addr], src; // PTX ISA 83, SM_70
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void st_L1_evict_last(
     cuda::ptx::space_global_t,
     B128* addr,
     B128 src);

st.global.L1::evict_last.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_last.v4.b64 [addr], src; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline void st_L1_evict_last(
     cuda::ptx::space_global_t,
     B256* addr,
     B256 src);

st.global.L1::evict_last.L2::cache_hint.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_last.L2::cache_hint.b8 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void st_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     B8* addr,
     B8 src,
     uint64_t cache_policy);

st.global.L1::evict_last.L2::cache_hint.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_last.L2::cache_hint.b16 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void st_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     B16* addr,
     B16 src,
     uint64_t cache_policy);

st.global.L1::evict_last.L2::cache_hint.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_last.L2::cache_hint.b32 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     B32* addr,
     B32 src,
     uint64_t cache_policy);

st.global.L1::evict_last.L2::cache_hint.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_last.L2::cache_hint.b64 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void st_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     B64* addr,
     B64 src,
     uint64_t cache_policy);

st.global.L1::evict_last.L2::cache_hint.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_last.L2::cache_hint.b128 [addr], src, cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void st_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     B128* addr,
     B128 src,
     uint64_t cache_policy);

st.global.L1::evict_last.L2::cache_hint.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::evict_last.L2::cache_hint.v4.b64 [addr], src, cache_policy; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline void st_L1_evict_last_L2_cache_hint(
     cuda::ptx::space_global_t,
     B256* addr,
     B256 src,
     uint64_t cache_policy);

st.global.L1::no_allocate.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::no_allocate.b8 [addr], src; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void st_L1_no_allocate(
     cuda::ptx::space_global_t,
     B8* addr,
     B8 src);

st.global.L1::no_allocate.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::no_allocate.b16 [addr], src; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void st_L1_no_allocate(
     cuda::ptx::space_global_t,
     B16* addr,
     B16 src);

st.global.L1::no_allocate.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::no_allocate.b32 [addr], src; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st_L1_no_allocate(
     cuda::ptx::space_global_t,
     B32* addr,
     B32 src);

st.global.L1::no_allocate.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::no_allocate.b64 [addr], src; // PTX ISA 74, SM_70
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void st_L1_no_allocate(
     cuda::ptx::space_global_t,
     B64* addr,
     B64 src);

st.global.L1::no_allocate.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::no_allocate.b128 [addr], src; // PTX ISA 83, SM_70
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void st_L1_no_allocate(
     cuda::ptx::space_global_t,
     B128* addr,
     B128 src);

st.global.L1::no_allocate.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::no_allocate.v4.b64 [addr], src; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline void st_L1_no_allocate(
     cuda::ptx::space_global_t,
     B256* addr,
     B256 src);

st.global.L1::no_allocate.L2::cache_hint.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::no_allocate.L2::cache_hint.b8 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void st_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     B8* addr,
     B8 src,
     uint64_t cache_policy);

st.global.L1::no_allocate.L2::cache_hint.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::no_allocate.L2::cache_hint.b16 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void st_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     B16* addr,
     B16 src,
     uint64_t cache_policy);

st.global.L1::no_allocate.L2::cache_hint.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::no_allocate.L2::cache_hint.b32 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     B32* addr,
     B32 src,
     uint64_t cache_policy);

st.global.L1::no_allocate.L2::cache_hint.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::no_allocate.L2::cache_hint.b64 [addr], src, cache_policy; // PTX ISA 74, SM_80
   // .space     = { .global }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void st_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     B64* addr,
     B64 src,
     uint64_t cache_policy);

st.global.L1::no_allocate.L2::cache_hint.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::no_allocate.L2::cache_hint.b128 [addr], src, cache_policy; // PTX ISA 83, SM_80
   // .space     = { .global }
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void st_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     B128* addr,
     B128 src,
     uint64_t cache_policy);

st.global.L1::no_allocate.L2::cache_hint.v4.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.space.L1::no_allocate.L2::cache_hint.v4.b64 [addr], src, cache_policy; // PTX ISA 88, SM_100
   // .space     = { .global }
   template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
   __device__ static inline void st_L1_no_allocate_L2_cache_hint(
     cuda::ptx::space_global_t,
     B256* addr,
     B256 src,
     uint64_t cache_policy);
