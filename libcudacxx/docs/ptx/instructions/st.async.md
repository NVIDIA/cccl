# st.async

-  PTX ISA: [`st.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async)
- Used in: [How to use st.async](../examples/st.async.md)

**NOTE.** Alignment of `addr` must be a multiple of vector size. For instance,
the `addr` supplied to the `v2.b32` variant must be aligned to `2 x 4 = 8` bytes.

| C++ | PTX |
| [(0)](#0-st_async) `cuda::ptx::st_async`| `st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32` |
| [(0)](#0-st_async) `cuda::ptx::st_async`| `st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b64` |
| [(1)](#1-st_async) `cuda::ptx::st_async`| `st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b32` |
| [(1)](#1-st_async) `cuda::ptx::st_async`| `st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b64` |
| [(2)](#2-st_async) `cuda::ptx::st_async`| `st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32` |


### [(0)](#0-st_async) `st_async`
{: .no_toc }
```cuda
// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes{.type} [addr], value, [remote_bar];    // 1.  PTX ISA 81, SM_90
// .type      = { .b32, .b64 }
template <typename Type>
__device__ static inline void st_async(
  Type* addr,
  const Type& value,
  uint64_t* remote_bar);
```

### [(1)](#1-st_async) `st_async`
{: .no_toc }
```cuda
// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2{.type} [addr], value, [remote_bar]; // 2.  PTX ISA 81, SM_90
// .type      = { .b32, .b64 }
template <typename Type>
__device__ static inline void st_async(
  Type* addr,
  const Type (&value)[2],
  uint64_t* remote_bar);
```

### [(2)](#2-st_async) `st_async`
{: .no_toc }
```cuda
// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [addr], value, [remote_bar];    // 3.  PTX ISA 81, SM_90
template <typename B32>
__device__ static inline void st_async(
  B32* addr,
  const B32 (&value)[4],
  uint64_t* remote_bar);
```
