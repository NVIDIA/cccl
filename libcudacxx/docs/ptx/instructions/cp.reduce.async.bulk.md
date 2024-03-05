# cp.reduce.async.bulk

- PTX ISA: [`cp.reduce.async.bulk`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk)

PTX does not currently (CTK 12.3) expose `cp.reduce.async.bulk.add.s64`. This exposure is emulated in `cuda::ptx` using:

```cuda
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
```

**cp_reduce_async_bulk**:
```cuda
// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .b32 }
// .op        = { .and }
template <typename B32>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_and_op_t,
  B32* dstMem,
  const B32* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);

// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .b32 }
// .op        = { .or }
template <typename B32>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_or_op_t,
  B32* dstMem,
  const B32* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);

// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .b32 }
// .op        = { .xor }
template <typename B32>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_xor_op_t,
  B32* dstMem,
  const B32* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);

// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .min }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);

// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .max }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);

// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .add }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);

// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .inc }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_inc_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);

// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .dec }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_dec_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);

// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .s32 }
// .op        = { .min }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  int32_t* dstMem,
  const int32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);

// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .s32 }
// .op        = { .max }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  int32_t* dstMem,
  const int32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);

// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .s32 }
// .op        = { .add }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  int32_t* dstMem,
  const int32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);

// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .u64 }
// .op        = { .add }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  uint64_t* dstMem,
  const uint64_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);

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

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 3. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .b32, .b64 }
// .op        = { .and }
template <typename Type>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_and_op_t,
  Type* dstMem,
  const Type* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 3. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .b32, .b64 }
// .op        = { .or }
template <typename Type>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_or_op_t,
  Type* dstMem,
  const Type* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 3. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .b32, .b64 }
// .op        = { .xor }
template <typename Type>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_xor_op_t,
  Type* dstMem,
  const Type* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .min }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .max }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .add }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .inc }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_inc_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .dec }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_dec_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .s32 }
// .op        = { .min }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  int32_t* dstMem,
  const int32_t* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .s32 }
// .op        = { .max }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  int32_t* dstMem,
  const int32_t* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .s32 }
// .op        = { .add }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  int32_t* dstMem,
  const int32_t* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .u64 }
// .op        = { .min }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  uint64_t* dstMem,
  const uint64_t* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .u64 }
// .op        = { .max }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  uint64_t* dstMem,
  const uint64_t* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .u64 }
// .op        = { .add }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  uint64_t* dstMem,
  const uint64_t* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .s64 }
// .op        = { .min }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  int64_t* dstMem,
  const int64_t* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .s64 }
// .op        = { .max }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  int64_t* dstMem,
  const int64_t* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .f32 }
// .op        = { .add }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  float* dstMem,
  const float* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .f64 }
// .op        = { .add }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  double* dstMem,
  const double* srcMem,
  uint32_t size);

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
```

**cp_reduce_async_bulk_f16**:
```cuda
// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .f16 }
// .op        = { .min }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  __half* dstMem,
  const __half* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .f16 }
// .op        = { .max }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  __half* dstMem,
  const __half* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.noftz.type  [dstMem], [srcMem], size; // 5. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .f16 }
// .op        = { .add }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  __half* dstMem,
  const __half* srcMem,
  uint32_t size);
```

**cp_reduce_async_bulk_bf16**:
```cuda
// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .bf16 }
// .op        = { .min }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  __nv_bfloat16* dstMem,
  const __nv_bfloat16* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .bf16 }
// .op        = { .max }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  __nv_bfloat16* dstMem,
  const __nv_bfloat16* srcMem,
  uint32_t size);

// cp.reduce.async.bulk.dst.src.bulk_group.op.noftz.type  [dstMem], [srcMem], size; // 5. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .bf16 }
// .op        = { .add }
template <typename=void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  __nv_bfloat16* dstMem,
  const __nv_bfloat16* srcMem,
  uint32_t size);
```
