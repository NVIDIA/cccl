# cp.reduce.async.bulk

- PTX ISA: [`cp.reduce.async.bulk`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk)

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>


## Integer and floating point instructions

| C++ | PTX |
| [(0)](#0-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.and.b32` |
| [(1)](#1-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.or.b32` |
| [(2)](#2-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.xor.b32` |
| [(3)](#3-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.min.u32` |
| [(4)](#4-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.max.u32` |
| [(5)](#5-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u32` |
| [(6)](#6-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.inc.u32` |
| [(7)](#7-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.dec.u32` |
| [(8)](#8-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.min.s32` |
| [(9)](#9-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.max.s32` |
| [(10)](#10-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.s32` |
| [(11)](#11-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u64` |
| [(12)](#12-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u64` |
| [(13)](#13-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.and.b32` |
| [(13)](#13-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.and.b64` |
| [(14)](#14-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.or.b32` |
| [(14)](#14-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.or.b64` |
| [(15)](#15-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b32` |
| [(15)](#15-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b64` |
| [(16)](#16-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.min.u32` |
| [(17)](#17-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.max.u32` |
| [(18)](#18-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32` |
| [(19)](#19-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.inc.u32` |
| [(20)](#20-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.dec.u32` |
| [(21)](#21-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s32` |
| [(22)](#22-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s32` |
| [(23)](#23-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.s32` |
| [(24)](#24-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.min.u64` |
| [(25)](#25-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.max.u64` |
| [(26)](#26-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u64` |
| [(27)](#27-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s64` |
| [(28)](#28-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s64` |
| [(29)](#29-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32` |
| [(30)](#30-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64` |
| [(31)](#31-cp_reduce_async_bulk) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u64` |


### [(0)](#0-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
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
```

### [(1)](#1-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(2)](#2-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(3)](#3-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(4)](#4-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(5)](#5-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(6)](#6-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(7)](#7-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(8)](#8-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(9)](#9-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(10)](#10-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(11)](#11-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(12)](#12-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
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
```

### [(13)](#13-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(14)](#14-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(15)](#15-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(16)](#16-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(17)](#17-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(18)](#18-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(19)](#19-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(20)](#20-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(21)](#21-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(22)](#22-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(23)](#23-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(24)](#24-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(25)](#25-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(26)](#26-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(27)](#27-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(28)](#28-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(29)](#29-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(30)](#30-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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
```

### [(31)](#31-cp_reduce_async_bulk) `cp_reduce_async_bulk`
{: .no_toc }
```cuda
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

## Emulation of `.s64` instruction

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

## FP16 instructions

| C++ | PTX |
| [(0)](#0-cp_reduce_async_bulk_f16) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.min.f16` |
| [(1)](#1-cp_reduce_async_bulk_f16) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.max.f16` |
| [(2)](#2-cp_reduce_async_bulk_f16) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16` |


### [(0)](#0-cp_reduce_async_bulk_f16) `cp_reduce_async_bulk_f16`
{: .no_toc }
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
```

### [(1)](#1-cp_reduce_async_bulk_f16) `cp_reduce_async_bulk_f16`
{: .no_toc }
```cuda
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
```

### [(2)](#2-cp_reduce_async_bulk_f16) `cp_reduce_async_bulk_f16`
{: .no_toc }
```cuda
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

## BF16 instructions

| C++ | PTX |
| [(0)](#0-cp_reduce_async_bulk_bf16) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.min.bf16` |
| [(1)](#1-cp_reduce_async_bulk_bf16) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.max.bf16` |
| [(2)](#2-cp_reduce_async_bulk_bf16) `cuda::ptx::cp_reduce_async_bulk`| `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16` |


### [(0)](#0-cp_reduce_async_bulk_bf16) `cp_reduce_async_bulk_bf16`
{: .no_toc }
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
```

### [(1)](#1-cp_reduce_async_bulk_bf16) `cp_reduce_async_bulk_bf16`
{: .no_toc }
```cuda
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
```

### [(2)](#2-cp_reduce_async_bulk_bf16) `cp_reduce_async_bulk_bf16`
{: .no_toc }
```cuda
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
