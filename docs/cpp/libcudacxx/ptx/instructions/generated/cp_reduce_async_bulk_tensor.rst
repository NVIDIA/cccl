..
   This file was automatically generated. Do not edit.

cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.min.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.max.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.inc.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.dec.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.and.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.or.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.xor.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.min.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.max.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.inc.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.dec.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.and.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.or.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.xor.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.min.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.max.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.inc.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.dec.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.and.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.or.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.xor.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.inc.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.dec.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.and.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.or.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.xor.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.inc.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.dec.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.and.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.or.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.xor.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.min.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.max.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.inc.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.dec.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.and.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.or.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.xor.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.override::global_address.override::global_dim.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.override::global_dim.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.min.tile.override::global_address.override::global_dim.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.override::global_dim.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.max.tile.override::global_address.override::global_dim.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.override::global_dim.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.inc.tile.override::global_address.override::global_dim.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.override::global_dim.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.dec.tile.override::global_address.override::global_dim.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.override::global_dim.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.and.tile.override::global_address.override::global_dim.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.override::global_dim.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.or.tile.override::global_address.override::global_dim.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.override::global_dim.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.1d.global.shared::cta.xor.tile.override::global_address.override::global_dim.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.override::global_dim.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.min.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.max.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.inc.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.dec.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.and.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.or.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.xor.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.min.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.max.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.inc.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.dec.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.and.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.or.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.2d.global.shared::cta.xor.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.min.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.max.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.inc.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.dec.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.and.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.or.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.xor.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.min.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.max.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.inc.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.dec.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.and.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.or.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.3d.global.shared::cta.xor.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.inc.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.dec.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.and.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.or.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.xor.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.inc.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.dec.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.and.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.or.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.4d.global.shared::cta.xor.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.inc.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.dec.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.and.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.or.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.xor.tile.override::global_address.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.inc.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.dec.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.and.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.or.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.reduce.async.bulk.tensor.5d.global.shared::cta.xor.tile.override::global_address.override::global_dim_stride.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_op Op>
   __device__ static inline void cp_reduce_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_t<Op> op,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);
