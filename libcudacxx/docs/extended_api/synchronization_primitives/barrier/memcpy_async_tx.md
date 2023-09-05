---
grand_parent: Extended API
parent: Barriers
---

# `cuda::device::memcpy_async_tx`

Defined in header `<cuda/barrier>`:

```cuda
template <typename T, size_t Alignment>
inline __device__
async_contract_fulfillment
cuda::device::memcpy_async_tx(
  T* dest,
  const T* src,
  cuda::aligned_size_t<Alignment> size,
  cuda::barrier<cuda::thread_scope_block>& bar);
```

Copies `size` bytes from global memory `src` to shared memory `dest` and arrives
on a shared memory barrier `bar`, updating its transaction count by `size`
bytes.
 
## Notes

The behavior is undefined if any of the following conditions hold: 
- `bar` is not in `__shared__` memory;
- `dest` is not 16-byte aligned
- `src` is not 16-byte aligned
- `size` is not a multiple of 16.

This function can only be used under CUDA Compute Capability 9.0 (Hopper) or
higher.

There is no feature flag to check if `cuda::device::memcpy_async_tx` is
available.

## Return Value

Returns `async_contract_fulfillment::async`.

## Example

```cuda
#include <cuda/barrier>
#include <cuda/std/utility> // cuda::std::move

#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Insufficient CUDA Compute Capability: cuda::device::memcpy_async_tx is not available.");
#endif // __CUDA_MINIMUM_ARCH__

__device__ alignas(16) int gmem_x[2048];

__global__ void example_kernel() {
  __shared__ alignas(16) int smem_x[1024];
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    cuda::device::memcpy_async_tx(smem_x, gmem_x, cuda::aligned_size_t<16>(sizeof(smem_x)), bar);
    token = cuda::device::arrive_tx(bar, 1, sizeof(smem_x));
  } else {
    auto token = bar.arrive(1);
  } 
  bar.wait(cuda::std::move(token));

  // smem_x contains the contents of gmem_x[0], ..., gmem_x[1023]
  smem_x[threadIdx.x] += 1;
}
```

[See it on Godbolt](https://godbolt.org/z/nTv558sK7){: .btn }


[`cuda::thread_scope`]: ./memory_model.md
[Tracking asynchronous operations by the mbarrier object]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tracking-asynchronous-operations-by-the-mbarrier-object

[`cp.async.bulk` PTX instruction]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk

[thread.barrier.class paragraph 12]: https://eel.is/c++draft/thread.barrier.class#12



