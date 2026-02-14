..
   This file was automatically generated. Do not edit.

cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1a. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1a. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1a. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1a. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1a. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1a. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1a. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1a. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1b. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1b. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1b. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1b. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1b. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1b. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1b. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1b. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1c. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1c. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1c. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1c. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1c. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1c. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1c. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1c. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1d. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1d. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1d. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1d. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1d. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1d. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1d. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1d. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1e. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1e. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1e. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1e. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1e. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1e. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1e. PTX ISA 80, SM_90
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

   // cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1e. PTX ISA 80, SM_90
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
