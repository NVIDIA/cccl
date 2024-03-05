# cp.async.bulk.tensor

- PTX ISA: [`cp.async.bulk.tensor`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Unicast

| C++ | PTX |
| [(0)](#0-cp_async_bulk_tensor) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes` |
| [(1)](#1-cp_async_bulk_tensor) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group` |
| [(2)](#2-cp_async_bulk_tensor) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes` |
| [(3)](#3-cp_async_bulk_tensor) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group` |
| [(4)](#4-cp_async_bulk_tensor) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes` |
| [(5)](#5-cp_async_bulk_tensor) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group` |
| [(6)](#6-cp_async_bulk_tensor) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes` |
| [(7)](#7-cp_async_bulk_tensor) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group` |
| [(8)](#8-cp_async_bulk_tensor) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes` |
| [(9)](#9-cp_async_bulk_tensor) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group` |


### [(0)](#0-cp_async_bulk_tensor) `cp_async_bulk_tensor`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1a. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar);
```

### [(1)](#1-cp_async_bulk_tensor) `cp_async_bulk_tensor`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.1d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3a. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  const void* srcMem);
```

### [(2)](#2-cp_async_bulk_tensor) `cp_async_bulk_tensor`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1b. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar);
```

### [(3)](#3-cp_async_bulk_tensor) `cp_async_bulk_tensor`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.2d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3b. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  const void* srcMem);
```

### [(4)](#4-cp_async_bulk_tensor) `cp_async_bulk_tensor`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1c. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar);
```

### [(5)](#5-cp_async_bulk_tensor) `cp_async_bulk_tensor`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.3d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3c. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  const void* srcMem);
```

### [(6)](#6-cp_async_bulk_tensor) `cp_async_bulk_tensor`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1d. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar);
```

### [(7)](#7-cp_async_bulk_tensor) `cp_async_bulk_tensor`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.4d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3d. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  const void* srcMem);
```

### [(8)](#8-cp_async_bulk_tensor) `cp_async_bulk_tensor`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1e. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar);
```

### [(9)](#9-cp_async_bulk_tensor) `cp_async_bulk_tensor`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.5d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3e. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  const void* srcMem);
```

## Multicast

| C++ | PTX |
| [(0)](#0-cp_async_bulk_tensor_multicast) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster` |
| [(1)](#1-cp_async_bulk_tensor_multicast) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster` |
| [(2)](#2-cp_async_bulk_tensor_multicast) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster` |
| [(3)](#3-cp_async_bulk_tensor_multicast) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster` |
| [(4)](#4-cp_async_bulk_tensor_multicast) `cuda::ptx::cp_async_bulk_tensor`| `cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster` |


### [(0)](#0-cp_async_bulk_tensor_multicast) `cp_async_bulk_tensor_multicast`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2a. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
```

### [(1)](#1-cp_async_bulk_tensor_multicast) `cp_async_bulk_tensor_multicast`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2b. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
```

### [(2)](#2-cp_async_bulk_tensor_multicast) `cp_async_bulk_tensor_multicast`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2c. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
```

### [(3)](#3-cp_async_bulk_tensor_multicast) `cp_async_bulk_tensor_multicast`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2d. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
```

### [(4)](#4-cp_async_bulk_tensor_multicast) `cp_async_bulk_tensor_multicast`
{: .no_toc }
```cuda
// cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2e. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
```
