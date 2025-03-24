..
   This file was automatically generated. Do not edit.

st.global.b8
^^^^^^^^^^^^
.. code:: cuda

   // st.global.b8 [addr], src; // PTX ISA 10, SM_50
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void st_global(
     B8* addr,
     B8 src);

st.global.b16
^^^^^^^^^^^^^
.. code:: cuda

   // st.global.b16 [addr], src; // PTX ISA 10, SM_50
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void st_global(
     B16* addr,
     B16 src);

st.global.b32
^^^^^^^^^^^^^
.. code:: cuda

   // st.global.b32 [addr], src; // PTX ISA 10, SM_50
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st_global(
     B32* addr,
     B32 src);

st.global.b64
^^^^^^^^^^^^^
.. code:: cuda

   // st.global.b64 [addr], src; // PTX ISA 10, SM_50
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void st_global(
     B64* addr,
     B64 src);

st.global.b128
^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.b128 [addr], src; // PTX ISA 83, SM_70
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void st_global(
     B128* addr,
     B128 src);

st.global.L1::evict_normal.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_normal.b8 [addr], src; // PTX ISA 74, SM_70
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void st_global_L1_evict_normal(
     B8* addr,
     B8 src);

st.global.L1::evict_normal.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_normal.b16 [addr], src; // PTX ISA 74, SM_70
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void st_global_L1_evict_normal(
     B16* addr,
     B16 src);

st.global.L1::evict_normal.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_normal.b32 [addr], src; // PTX ISA 74, SM_70
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st_global_L1_evict_normal(
     B32* addr,
     B32 src);

st.global.L1::evict_normal.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_normal.b64 [addr], src; // PTX ISA 74, SM_70
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void st_global_L1_evict_normal(
     B64* addr,
     B64 src);

st.global.L1::evict_normal.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_normal.b128 [addr], src; // PTX ISA 83, SM_70
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void st_global_L1_evict_normal(
     B128* addr,
     B128 src);

st.global.L1::evict_unchanged.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_unchanged.b8 [addr], src; // PTX ISA 74, SM_70
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void st_global_L1_evict_unchanged(
     B8* addr,
     B8 src);

st.global.L1::evict_unchanged.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_unchanged.b16 [addr], src; // PTX ISA 74, SM_70
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void st_global_L1_evict_unchanged(
     B16* addr,
     B16 src);

st.global.L1::evict_unchanged.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_unchanged.b32 [addr], src; // PTX ISA 74, SM_70
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st_global_L1_evict_unchanged(
     B32* addr,
     B32 src);

st.global.L1::evict_unchanged.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_unchanged.b64 [addr], src; // PTX ISA 74, SM_70
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void st_global_L1_evict_unchanged(
     B64* addr,
     B64 src);

st.global.L1::evict_unchanged.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_unchanged.b128 [addr], src; // PTX ISA 83, SM_70
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void st_global_L1_evict_unchanged(
     B128* addr,
     B128 src);

st.global.L1::evict_first.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_first.b8 [addr], src; // PTX ISA 74, SM_70
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void st_global_L1_evict_first(
     B8* addr,
     B8 src);

st.global.L1::evict_first.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_first.b16 [addr], src; // PTX ISA 74, SM_70
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void st_global_L1_evict_first(
     B16* addr,
     B16 src);

st.global.L1::evict_first.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_first.b32 [addr], src; // PTX ISA 74, SM_70
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st_global_L1_evict_first(
     B32* addr,
     B32 src);

st.global.L1::evict_first.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_first.b64 [addr], src; // PTX ISA 74, SM_70
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void st_global_L1_evict_first(
     B64* addr,
     B64 src);

st.global.L1::evict_first.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_first.b128 [addr], src; // PTX ISA 83, SM_70
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void st_global_L1_evict_first(
     B128* addr,
     B128 src);

st.global.L1::evict_last.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_last.b8 [addr], src; // PTX ISA 74, SM_70
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void st_global_L1_evict_last(
     B8* addr,
     B8 src);

st.global.L1::evict_last.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_last.b16 [addr], src; // PTX ISA 74, SM_70
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void st_global_L1_evict_last(
     B16* addr,
     B16 src);

st.global.L1::evict_last.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_last.b32 [addr], src; // PTX ISA 74, SM_70
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st_global_L1_evict_last(
     B32* addr,
     B32 src);

st.global.L1::evict_last.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_last.b64 [addr], src; // PTX ISA 74, SM_70
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void st_global_L1_evict_last(
     B64* addr,
     B64 src);

st.global.L1::evict_last.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::evict_last.b128 [addr], src; // PTX ISA 83, SM_70
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void st_global_L1_evict_last(
     B128* addr,
     B128 src);

st.global.L1::no_allocate.b8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::no_allocate.b8 [addr], src; // PTX ISA 74, SM_70
   template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
   __device__ static inline void st_global_L1_no_allocate(
     B8* addr,
     B8 src);

st.global.L1::no_allocate.b16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::no_allocate.b16 [addr], src; // PTX ISA 74, SM_70
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void st_global_L1_no_allocate(
     B16* addr,
     B16 src);

st.global.L1::no_allocate.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::no_allocate.b32 [addr], src; // PTX ISA 74, SM_70
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st_global_L1_no_allocate(
     B32* addr,
     B32 src);

st.global.L1::no_allocate.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::no_allocate.b64 [addr], src; // PTX ISA 74, SM_70
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void st_global_L1_no_allocate(
     B64* addr,
     B64 src);

st.global.L1::no_allocate.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.global.L1::no_allocate.b128 [addr], src; // PTX ISA 83, SM_70
   template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
   __device__ static inline void st_global_L1_no_allocate(
     B128* addr,
     B128 src);
