# cp.async.bulk

-  PTX ISA: [`cp.async.bulk`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Implementation notes

**NOTE.** Both `srcMem` and `dstMem` must be 16-byte aligned, and `size` must be a multiple of 16.

## Unicast

| C++ | PTX |
| [(0)](#0-cp_async_bulk) `cuda::ptx::cp_async_bulk`| `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` |
| [(1)](#1-cp_async_bulk) `cuda::ptx::cp_async_bulk`| `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes` |
| [(2)](#2-cp_async_bulk) `cuda::ptx::cp_async_bulk`| `cp.async.bulk.global.shared::cta.bulk_group` |


### [(0)](#0-cp_async_bulk) `cp_async_bulk`
{: .no_toc }
```cuda
// cp.async.bulk.dst.src.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [smem_bar]; // 1a. unicast PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  uint64_t* smem_bar);
```

### [(1)](#1-cp_async_bulk) `cp_async_bulk`
{: .no_toc }
```cuda
// cp.async.bulk.dst.src.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [rdsmem_bar]; // 2.  PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  uint64_t* rdsmem_bar);
```

### [(2)](#2-cp_async_bulk) `cp_async_bulk`
{: .no_toc }
```cuda
// cp.async.bulk.dst.src.bulk_group [dstMem], [srcMem], size; // 3.  PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size);
```

## Multicast

| C++ | PTX |
| [(0)](#0-cp_async_bulk_multicast) `cuda::ptx::cp_async_bulk`| `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster` |


### [(0)](#0-cp_async_bulk_multicast) `cp_async_bulk_multicast`
{: .no_toc }
```cuda
// cp.async.bulk{.dst}{.src}.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem], size, [smem_bar], ctaMask; // 1.  PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
```
