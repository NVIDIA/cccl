# getctarank

- PTX ISA: [`getctarank`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-getctarank)

| C++ | PTX |
| [(0)](#0-getctarank) `cuda::ptx::getctarank`| `getctarank.shared::cluster.u32` |


### [(0)](#0-getctarank) `getctarank`
{: .no_toc }
```cuda
// getctarank{.space}.u32 dest, addr; // PTX ISA 78, SM_90
// .space     = { .shared::cluster }
template <typename=void>
__device__ static inline uint32_t getctarank(
  cuda::ptx::space_cluster_t,
  const void* addr);
```
