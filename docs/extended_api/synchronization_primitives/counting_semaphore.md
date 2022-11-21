---
grand_parent: Extended API
parent: Synchronization Primitives
nav_order: 3
---

# `cuda::counting_semaphore`

Defined in header `<cuda/semaphore>`:

```cuda
template <cuda::thread_scope Scope,
          cuda::std::ptrdiff_t LeastMaxValue = /* implementation-defined */>
class cuda::counting_semaphore;
```

The class template `cuda::counting_semaphore` is an extended form of
  [`cuda::std::counting_semaphore`] that takes an additional
  [`cuda::thread_scope`] argument.
`cuda::counting_semaphore` has the same interface and semantics as
  [`cuda::std::counting_semaphore`].

## Concurrency Restrictions

An object of type `cuda::counting_semaphore` or `cuda::std::counting_semaphore`,
  shall not be accessed concurrently by CPU and GPU threads unless:
- it is in unified memory and the [`concurrentManagedAccess` property] is 1, or
- it is in CPU memory and the [`hostNativeAtomicSupported` property] is 1.

Note, for objects of scopes other than `cuda::thread_scope_system` this is a
  data-race, and thefore also prohibited regardless of memory characteristics.

Under CUDA Compute Capability 6 (Pascal) or prior, an object of type
  `cuda::counting_semaphore` or `cuda::std::counting_semaphore` may not be used.

## Implementation-Defined Behavior

For each [`cuda::thread_scope`] `S` and least maximum value `V`,
  `counting_semaphore<S,V>::max()` is as follows:

| [`cuda::thread_scope`] `S` | Least Maximum Value `V` | `cuda::counting_semaphore<S,V>::max()`                   |
|----------------------------|-------------------------|----------------------------------------------------------|
| Any                        | Any                     | `cuda::std::numeric_limits<cuda::std::ptrdiff_t>::max()` |

## Example

```cuda
#include <cuda/semaphore>

__global__ void example_kernel() {
  // This semaphore is suitable for all threads in the system.
  cuda::counting_semaphore<cuda::thread_scope_system> a;

  // This semaphore has the same type as the previous one (`a`).
  cuda::std::counting_semaphore<> b;

  // This semaphore is suitable for all threads on the current processor (e.g. GPU).
  cuda::counting_semaphore<cuda::thread_scope_device> c;

  // This semaphore is suitable for all threads in the same thread block.
  cuda::counting_semaphore<cuda::thread_scope_block> d;
}
```

[See it on Godbolt](https://godbolt.org/z/3YrjjTvG6){: .btn }


[`cuda::thread_scope`]: ../memory_model.md

[`cuda::std::counting_semaphore`]: https://en.cppreference.com/w/cpp/thread/counting_semaphore

[`concurrentManagedAccess` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b
[`hostNativeAtomicSupported` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f
