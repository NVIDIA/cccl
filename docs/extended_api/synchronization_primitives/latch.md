---
grand_parent: Extended API
parent: Synchronization Primitives
nav_order: 1
---

# `cuda::latch`

Defined in header `<cuda/latch>`:

```cuda
template <cuda::thread_scope Scope>
class cuda::latch;
```

The class template `cuda::latch` is an extended form of [`cuda::std::latch`]
  takes an additional [`cuda::thread_scope`] argument.
It has the same interface and semantics as [`cuda::std::latch`].

## Concurrency Restrictions

An object of type `cuda::latch` or [`cuda::std::latch`] shall not be accessed
  concurrently by CPU and GPU threads unless:
- it is in unified memory and the [`concurrentManagedAccess` property] is 1, or
- it is in CPU memory and the [`hostNativeAtomicSupported` property] is 1.

Note, for objects of scopes other than `cuda::thread_scope_system` this is a
  data-race, and thefore also prohibited regardless of memory characteristics.

Under CUDA Compute Capability 6 (Pascal) or prior, an object of type
  `cuda::latch` or [`cuda::std::latch`] may not be used.

## Implementation-Defined Behavior

For each [`cuda::thread_scope`] `S`, the value of `cuda::latch<S>::max()` is as
  follows:

| [`cuda::thread_scope`] `S` | `cuda::latch<S>::max()`                                  |
|----------------------------|----------------------------------------------------------|
| Any                        | `cuda::std::numeric_limits<cuda::std::ptrdiff_t>::max()` |

## Example

```cuda
#include <cuda/latch>

__global__ void example_kernel() {
  // This latch is suitable for all threads in the system.
  cuda::latch<cuda::thread_scope_system> a(10);

  // This latch has the same type as the previous one (`a`).
  cuda::std::latch b(10);

  // This latch is suitable for all threads on the current processor (e.g. GPU).
  cuda::latch<cuda::thread_scope_device> c(10);

  // This latch is suitable for all threads in the same thread block.
  cuda::latch<cuda::thread_scope_block> d(10);
}
```

[See it on Godbolt](https://godbolt.org/z/8v4dcK7fa){: .btn }


[`cuda::thread_scope`]: ../memory_model.md

[`cuda::std::latch`]: https://en.cppreference.com/w/cpp/thread/latch

[`concurrentManagedAccess` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b
[`hostNativeAtomicSupported` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f
