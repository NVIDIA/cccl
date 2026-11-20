.. _libcudacxx-ptx-instructions-mbarrier-red-async:

red.async
=========

-  PTX ISA:
   `red.async <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red-async>`__

.. _red.async-1:

red.async
---------

.. include:: generated/red_async.rst

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
