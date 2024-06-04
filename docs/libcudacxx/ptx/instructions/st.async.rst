.. _libcudacxx-ptx-instructions-st-async:

st.async
========

-  PTX ISA:
   `st.async <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async>`__
-  Used in: :ref:`How to use st.async <libcudacxx-ptx-examples-st-async>`

**NOTE.** Alignment of ``addr`` must be a multiple of vector size. For
instance, the ``addr`` supplied to the ``v2.b32`` variant must be
aligned to ``2 x 4 = 8`` bytes.

st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes{.type} [addr], value, [remote_bar];    // 1.  PTX ISA 81, SM_90
   // .type      = { .b32, .b64 }
   template <typename Type>
   __device__ static inline void st_async(
     Type* addr,
     const Type& value,
     uint64_t* remote_bar);

st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes{.type} [addr], value, [remote_bar];    // 1.  PTX ISA 81, SM_90
   // .type      = { .b32, .b64 }
   template <typename Type>
   __device__ static inline void st_async(
     Type* addr,
     const Type& value,
     uint64_t* remote_bar);

st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2{.type} [addr], value, [remote_bar]; // 2.  PTX ISA 81, SM_90
   // .type      = { .b32, .b64 }
   template <typename Type>
   __device__ static inline void st_async(
     Type* addr,
     const Type (&value)[2],
     uint64_t* remote_bar);

st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2{.type} [addr], value, [remote_bar]; // 2.  PTX ISA 81, SM_90
   // .type      = { .b32, .b64 }
   template <typename Type>
   __device__ static inline void st_async(
     Type* addr,
     const Type (&value)[2],
     uint64_t* remote_bar);

st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [addr], value, [remote_bar];    // 3.  PTX ISA 81, SM_90
   template <typename B32>
   __device__ static inline void st_async(
     B32* addr,
     const B32 (&value)[4],
     uint64_t* remote_bar);
