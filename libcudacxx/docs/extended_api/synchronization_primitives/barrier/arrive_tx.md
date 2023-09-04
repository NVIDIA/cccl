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

## Return Value

An arrival_token, just like the one returned from `cuda::barrier:arrive`.

## Example

```cuda
#include <cuda/barrier>
#include <cuda/std/utility> // cuda::std::move

__global__ void example_kernel() {
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }

  auto token = cuda::device::arrive_tx(bar, 1, 0);

  bar.wait(cuda::std::move(token));
}
```

[See it on Godbolt](https://godbolt.org/z/GdTrGo3Kx){: .btn }


[`cuda::thread_scope`]: ./memory_model.md
[Tracking asynchronous operations by the mbarrier object]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tracking-asynchronous-operations-by-the-mbarrier-object
[thread.barrier.class paragraph 12]: https://eel.is/c++draft/thread.barrier.class#12

