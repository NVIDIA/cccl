# cp.async.bulk.wait_group

- PTX ISA: [`cp.async.bulk.wait_group`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)

| C++ | PTX |
| [(0)](#0-cp_async_bulk_wait_group) `cuda::ptx::cp_async_bulk_wait_group`| `cp.async.bulk.wait_group` |
| [(1)](#1-cp_async_bulk_wait_group) `cuda::ptx::cp_async_bulk_wait_group_read`| `cp.async.bulk.wait_group.read` |


### [(0)](#0-cp_async_bulk_wait_group) `cp_async_bulk_wait_group`
{: .no_toc }
```cuda
// cp.async.bulk.wait_group N; // PTX ISA 80, SM_90
template <int N32>
__device__ static inline void cp_async_bulk_wait_group(
  cuda::ptx::n32_t<N32> N);
```

### [(1)](#1-cp_async_bulk_wait_group) `cp_async_bulk_wait_group`
{: .no_toc }
```cuda
// cp.async.bulk.wait_group.read N; // PTX ISA 80, SM_90
template <int N32>
__device__ static inline void cp_async_bulk_wait_group_read(
  cuda::ptx::n32_t<N32> N);
```
