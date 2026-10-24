..
   This file was automatically generated. Do not edit.

cp.reduce.async.bulk.global.shared::cta.bulk_group.min.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .type      = { .bf16 }
   // .op        = { .min }
   template <typename = void>
   __device__ static inline void cp_reduce_async_bulk(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_min_t,
     __nv_bfloat16* dstMem,
     const __nv_bfloat16* srcMem,
     uint32_t size);

cp.reduce.async.bulk.global.shared::cta.bulk_group.max.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .type      = { .bf16 }
   // .op        = { .max }
   template <typename = void>
   __device__ static inline void cp_reduce_async_bulk(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_max_t,
     __nv_bfloat16* dstMem,
     const __nv_bfloat16* srcMem,
     uint32_t size);

cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.dst.src.bulk_group.op.noftz.type  [dstMem], [srcMem], size; // 5. PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .type      = { .bf16 }
   // .op        = { .add }
   template <typename = void>
   __device__ static inline void cp_reduce_async_bulk(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_add_t,
     __nv_bfloat16* dstMem,
     const __nv_bfloat16* srcMem,
     uint32_t size);
