.. _libcudacxx-ptx-instructions-mbarrier-red-async:

red.async
=========

-  PTX ISA:
   `red.async <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red-async>`__

.. _red.async-1:

red.async
---------

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.inc.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .u32 }
   // .op        = { .inc }
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_inc_t,
     uint32_t* dest,
     const uint32_t& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.dec.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .u32 }
   // .op        = { .dec }
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_dec_t,
     uint32_t* dest,
     const uint32_t& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .u32 }
   // .op        = { .min }
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_min_t,
     uint32_t* dest,
     const uint32_t& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .u32 }
   // .op        = { .max }
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_max_t,
     uint32_t* dest,
     const uint32_t& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .u32 }
   // .op        = { .add }
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_add_t,
     uint32_t* dest,
     const uint32_t& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .s32 }
   // .op        = { .min }
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_min_t,
     int32_t* dest,
     const int32_t& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .s32 }
   // .op        = { .max }
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_max_t,
     int32_t* dest,
     const int32_t& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .s32 }
   // .op        = { .add }
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_add_t,
     int32_t* dest,
     const int32_t& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .b32 }
   // .op        = { .and }
   template <typename B32>
   __device__ static inline void red_async(
     cuda::ptx::op_and_op_t,
     B32* dest,
     const B32& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .b32 }
   // .op        = { .or }
   template <typename B32>
   __device__ static inline void red_async(
     cuda::ptx::op_or_op_t,
     B32* dest,
     const B32& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .b32 }
   // .op        = { .xor }
   template <typename B32>
   __device__ static inline void red_async(
     cuda::ptx::op_xor_op_t,
     B32* dest,
     const B32& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .u64 }
   // .op        = { .add }
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_add_t,
     uint64_t* dest,
     const uint64_t& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}.u64  [dest], value, [remote_bar]; // .u64 intentional PTX ISA 81, SM_90
   // .op        = { .add }
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_add_t,
     int64_t* dest,
     const int64_t& value,
     int64_t* remote_bar);

red.async ``.s64`` emulation
----------------------------

PTX does not currently (CTK 12.3) expose ``red.async.add.s64``. This
exposure is emulated in ``cuda::ptx`` using

.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}.u64  [dest], value, [remote_bar]; // .u64 intentional PTX ISA 81, SM_90
   // .op        = { .add }
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_add_t,
     int64_t* dest,
     const int64_t& value,
     int64_t* remote_bar);
