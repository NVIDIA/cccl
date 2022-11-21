---
grand_parent: Extended API
parent: Barriers
---

# `cuda::barrier::init`

Defined in header `<cuda/barrier>`:

```cuda
template <cuda::thread_scope Scope,
          typename CompletionFunction = /* unspecified */>
class barrier {
public:
  // ...

  __host__ __device__
  friend void init(cuda::std::barrier* bar,
                   cuda::std::ptrdiff_t expected,
                   CompletionFunction cf = CompletionFunction{});
};
```

The friend function `cuda::barrier::init` may be used to initialize an
  `cuda::barrier` that has not been initialized.

When using libcu++ with NVCC,  `__shared__` `cuda::barrier` will not have its
  constructors run because `__shared__` variables are not initialized.
`cuda::barrier::init` should be use to properly initialize such a
  `cuda::barrier`.

An NVCC diagnostic warning about the ignored constructor will be emitted:

```
warning: dynamic initialization is not supported for a function-scope static
__shared__ variable within a __device__/__global__ function
```

It can be silenced using `#pragma diag_suppress static_var_with_dynamic_init`.

## Example

```cuda
#include <cuda/barrier>

// Disables `pipeline_shared_state` initialization warning.
#pragma diag_suppress static_var_with_dynamic_init

__global__ void example_kernel() {
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  init(&bar, 1);
}
```

[See it on Godbolt](https://godbolt.org/z/jG8se6Kd8){: .btn }


[`cuda::thread_scope`]: ./memory_model.md

[thread.barrier.class paragraph 12]: https://eel.is/c++draft/thread.barrier.class#12

[coalesced threads]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#coalesced-group-cg

[`concurrentManagedAccess` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b
[`hostNativeAtomicSupported` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f
