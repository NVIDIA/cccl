---
grand_parent: Extended API
parent: Barriers
---

# `cuda::device::barrier_native_handle`

Defined in header `<cuda/barrier>`:

```cuda
__device__ cuda::std::uint64_t* cuda::device::barrier_native_handle(
  cuda::barrier<cuda::thread_scope_block>& bar);
```

Returns a pointer to the native handle of a [`cuda::barrier`] if its scope
  is `cuda::thread_scope_block` and it is allocated in shared memory.
The pointer is suitable for use with PTX instructions.

## Notes

If `bar` is not in `__shared__` memory, the behavior is undefined.

## Return Value

A pointer to the PTX "mbarrier" subobject of the `cuda::barrier` object.

## Example

```cuda
#include <cuda/barrier>

__global__ void example_kernel(cuda::barrier<cuda::thread_scope_block>& bar) {
  auto ptr = cuda::device::barrier_native_handle(bar);

  asm volatile (
      "mbarrier.arrive.b64 _, [%0];"
      :
      : "l" (ptr)
      : "memory");
  // Equivalent to: `(void)b.arrive()`.
}
```

[See it on Godbolt](https://godbolt.org/z/dr4798Y76){: .btn }


[`cuda::thread_scope`]: ./memory_model.md

[thread.barrier.class paragraph 12]: https://eel.is/c++draft/thread.barrier.class#12

[coalesced threads]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#coalesced-group-cg

[`concurrentManagedAccess` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b
[`hostNativeAtomicSupported` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f
