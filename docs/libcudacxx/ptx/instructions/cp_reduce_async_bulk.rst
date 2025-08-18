.. _libcudacxx-ptx-instructions-cp-reduce-async-bulk:

cp.reduce.async.bulk
====================

-  PTX ISA:
   `cp.reduce.async.bulk <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk>`__


Integer and floating point instructions
---------------------------------------

.. include:: generated/cp_reduce_async_bulk.rst

Emulation of ``.s64`` instruction
---------------------------------

PTX does not currently (CTK 12.3) expose
``cp.reduce.async.bulk.add.s64``. This exposure is emulated in
``cuda::ptx`` using:

.. code:: cuda

   // cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.u64 [dstMem], [srcMem], size, [rdsmem_bar]; // 2. PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .shared::cta }
   // .type      = { .s64 }
   // .op        = { .add }
   template <typename=void>
   __device__ static inline void cp_reduce_async_bulk(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_add_t,
     int64_t* dstMem,
     const int64_t* srcMem,
     uint32_t size,
     uint64_t* rdsmem_bar);

   // cp.reduce.async.bulk.dst.src.bulk_group.op.u64  [dstMem], [srcMem], size; // 6. PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .type      = { .s64 }
   // .op        = { .add }
   template <typename=void>
   __device__ static inline void cp_reduce_async_bulk(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_add_t,
     int64_t* dstMem,
     const int64_t* srcMem,
     uint32_t size);

FP16 instructions
-----------------

.. include:: generated/cp_reduce_async_bulk_f16.rst

BF16 instructions
-----------------

.. include:: generated/cp_reduce_async_bulk_bf16.rst
