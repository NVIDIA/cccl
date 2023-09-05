---
grand_parent: Extended API
parent: Barriers
---

# `cuda::device::arrive_tx`

Defined in header `<cuda/barrier>`:

```cuda
__device__
cuda::barrier<cuda::thread_scope_block>::arrival_token
cuda::device::arrive_tx(
  cuda::barrier<cuda::thread_scope_block>& bar,
  ptrdiff_t arrive_count_update,
  ptrdiff_t transaction_count_update);
```

Arrives at a barrier in shared memory, updating both the arrival count and
transaction count.

## Notes

If `bar` is not in `__shared__` memory, the behavior is undefined. This function
can only be used under CUDA Compute Capability 9.0 (Hopper) or higher.

To check if `cuda::device::arive_tx` is available, use the
`__cccl_lib_local_barrier_arrive_tx` feature flag, as shown in the example code below.

## Return Value

An arrival_token, just like the one returned from `cuda::barrier:arrive`.

## Example

```cuda
#include <cuda/barrier>
#include <cuda/std/utility> // cuda::std::move

#if defined(__CUDA_MINIMUM_ARCH__) && 900 <= __CUDA_MINIMUM_ARCH__
#ifndef  __cccl_lib_local_barrier_arrive_tx
static_assert(false, "Insufficient libcu++ version: cuda::device::arrive_tx is not yet available.");
#endif // __cccl_lib_local_barrier_arrive_tx
#endif // __CUDA_MINIMUM_ARCH__

__global__ void example_kernel() {
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }

  auto token = cuda::device::arrive_tx(bar, 1, 0);

  bar.wait(cuda::std::move(token));
}
```

[See it on Godbolt](https://godbolt.org/z/joja75jK8){: .btn }


[`cuda::thread_scope`]: ./memory_model.md
[Tracking asynchronous operations by the mbarrier object]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tracking-asynchronous-operations-by-the-mbarrier-object
[thread.barrier.class paragraph 12]: https://eel.is/c++draft/thread.barrier.class#12

