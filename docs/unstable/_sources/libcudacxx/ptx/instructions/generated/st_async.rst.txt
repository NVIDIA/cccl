..
   This file was automatically generated. Do not edit.

st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.type [addr], value, [remote_bar];    // 1.  PTX ISA 81, SM_90
   // .type      = { .b32, .b64 }
   template <typename Type>
   __device__ static inline void st_async(
     Type* addr,
     const Type& value,
     uint64_t* remote_bar);

st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.type [addr], value, [remote_bar];    // 1.  PTX ISA 81, SM_90
   // .type      = { .b32, .b64 }
   template <typename Type>
   __device__ static inline void st_async(
     Type* addr,
     const Type& value,
     uint64_t* remote_bar);

st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.type [addr], value, [remote_bar]; // 2.  PTX ISA 81, SM_90
   // .type      = { .b32, .b64 }
   template <typename Type>
   __device__ static inline void st_async(
     Type* addr,
     const Type (&value)[2],
     uint64_t* remote_bar);

st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.type [addr], value, [remote_bar]; // 2.  PTX ISA 81, SM_90
   // .type      = { .b32, .b64 }
   template <typename Type>
   __device__ static inline void st_async(
     Type* addr,
     const Type (&value)[2],
     uint64_t* remote_bar);

st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [addr], value, [remote_bar];    // 3.  PTX ISA 81, SM_90
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void st_async(
     B32* addr,
     const B32 (&value)[4],
     uint64_t* remote_bar);
